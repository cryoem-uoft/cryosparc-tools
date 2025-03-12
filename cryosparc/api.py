import json
import re
import urllib.parse
import warnings
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Type, TypedDict, Union

import httpx

from . import registry
from .errors import APIError
from .json_util import api_default, api_encode, api_object_hook
from .models.auth import Token
from .stream import Streamable

_BASE_RESPONSE_TYPES = ("string", "integer", "number", "boolean")

Auth = Union[str, Tuple[str, str]]
"""
Auth token or email/password.
"""

FormEncoding = Literal["urlencoded", "json"]
"""
How form data should be transmitted over the API. Use "json" for normal
connections to the CryoSPARC instance over the base port (e.g., 39000).
"""


class APIRequest(TypedDict):
    params: Dict[str, Any]
    headers: Dict[str, str]
    content: Any
    data: Optional[Dict[str, Any]]  # form data
    files: Optional[Dict[str, Any]]


class APINamespace:
    """
    Collection of API methods that call a certain namespace, e.g., only the
    methods under http://master:39004/pipelines/
    """

    _client: httpx.Client
    _form_encoding: FormEncoding

    def __init__(self, http_client: httpx.Client, form_encoding: FormEncoding = "urlencoded"):
        self._client = http_client
        self._form_encoding = form_encoding

    def _set_headers(self, update: Dict[str, str]):
        """For testing only, reset client headers"""
        self._client.headers.update(update)

    def _prepare_request_stream(self, streamable):
        return streamable.stream()

    def _construct_request(self, _path: str, _schema, *args, **kwargs) -> Tuple[str, APIRequest]:
        args = list(args)
        query_params = {}
        func_name = _schema["summary"]
        headers = {}
        client_headers = {h for h in self._client.headers.keys()}
        request_body = None
        data = None
        files = None

        for param_schema in sort_params_schema(_path, _schema.get("parameters", [])):
            # Compile function params
            param_name: str = param_schema["name"]
            param_in: str = param_schema["in"]
            assert param_in in (
                "path",
                "query",
                "header",
            ), f"[API] Param specification for '{param_name}' in {param_in} NOT supported."

            if param_in == "path" and args:
                # path param must be specified positionally
                param, args = args[0], args[1:]
                _path = _path.replace("{%s}" % param_name, _uriencode(param))
            elif param_in == "query" and param_name in kwargs and (value := kwargs.pop(param_name)) is not None:
                # query param must be in kwargs
                query_params[param_name] = api_encode(value)
            elif (
                param_in == "header"
                and (header_name := param_name.replace("-", "_")) in kwargs
                and (value := kwargs.pop(header_name)) is not None
            ):
                # header must be in kwargs
                headers[param_name] = api_encode(value)
            elif param_in == "header" and param_name in client_headers:
                pass  # in default headers, no action required
            elif param_schema["required"]:
                raise TypeError(f"[API] {func_name}() missing required argument: '{param_name}'")

        if "requestBody" in _schema:
            content_schema = _schema["requestBody"].get("content", {})
            if stream_mime_type := registry.first_streamable_mime(content_schema.keys()):
                body_name = _get_schema_param_name(content_schema[stream_mime_type]["schema"], "body")
                headers["Content-Type"] = stream_mime_type
                if args:
                    streamable, args = args[0], args[1:]
                elif body_name in kwargs:
                    streamable = kwargs.pop(body_name)
                elif content_schema.get("required", False):
                    raise TypeError(f"[API] {func_name}() missing required argument: {body_name}")
                else:
                    streamable = None

                if streamable is not None:
                    if not isinstance(streamable, Streamable):
                        raise TypeError(f"[API] {func_name}() invalid argument {streamable}; expected Streamable type")
                    request_body = self._prepare_request_stream(streamable)
            elif "application/json" in content_schema:
                body_name = _get_schema_param_name(content_schema["application/json"]["schema"], "body")
                headers["Content-Type"] = "application/json"
                if args:
                    request_body, args = args[0], args[1:]
                    request_body = json.dumps(request_body, default=api_default)
                elif body_name in kwargs:
                    request_body = kwargs.pop(body_name)
                    request_body = json.dumps(request_body, default=api_default)
                elif content_schema.get("required", False):
                    raise TypeError(f"[API] {func_name}() missing required argument: {body_name}")
            elif "application/x-www-form-urlencoded" in content_schema:
                assert kwargs, (
                    f"[API] {func_name}() requires x-www-form-urlencoded which "
                    "does not yet support positional arguments for the content body."
                )
                if self._form_encoding == "json":
                    request_body, kwargs = json.dumps(kwargs, default=api_default), {}
                else:
                    data, kwargs = kwargs, {}
            elif "multipart/form-data" in content_schema:
                assert kwargs, (
                    f"[API] {func_name}() requires multipart/form-data which "
                    "does not yet support positional arguments for the content body."
                )
                files, kwargs = kwargs, {}
            else:
                raise TypeError(f"[API] Does not yet support request body with content schema {content_schema}")

        # Fail if any extra parameters
        if args:
            raise TypeError(f"[API] {func_name}() given extra positional arguments ({', '.join(map(str, args))})")
        if kwargs:
            raise TypeError(f"[API] {func_name}() given unknown keyword arguments ({', '.join(kwargs.keys())})")

        return _path, APIRequest(params=query_params, headers=headers, content=request_body, data=data, files=files)

    def _handle_response(self, schema, res: httpx.Response):
        responses_schema = schema.get("responses", {})
        response_schema = responses_schema.get(str(res.status_code))
        if not response_schema:
            res.raise_for_status()
            raise APIError("Received unknown response", res=res)

        if "content" not in response_schema:
            res.raise_for_status()
            return None

        content_schema = response_schema["content"]
        stream_mime_type = registry.first_streamable_mime(content_schema.keys())
        if stream_mime_type is not None:  # This is a streaming type
            stream_class = registry.get_stream_class(stream_mime_type)
            assert stream_class
            return stream_class.from_iterator(
                res.iter_bytes(),
                media_type=res.headers.get("Content-Type", stream_mime_type),
            )
        elif "text/plain" in content_schema:
            return res.text
        elif "application/json" in content_schema:
            data = res.json(object_hook=api_object_hook)
            if res.status_code >= 400:
                raise APIError("Received error response", res=res, data=data)
            return _decode_json_response(data, content_schema["application/json"]["schema"])
        else:
            raise APIError(f"Received unimplemented response type in {content_schema.keys()}", res=res)

    @contextmanager
    def _request(self, method: str, url: str, *args, **kwargs) -> Iterator[httpx.Response]:
        attempt = 0
        max_attempts = 3
        while attempt < max_attempts:
            attempt += 1
            try:
                request = self._client.request(method, url, *args, **kwargs)
                yield request
                return
            except httpx.TransportError as err:
                warnings.warn(
                    f"*** API client {method.upper()} {url} failed due to {err} (attempt {attempt}/{max_attempts})"
                )
                if attempt == max_attempts:
                    raise

    @contextmanager
    def _request_stream(self, method: str, url: str, *args, **kwargs) -> Iterator[httpx.Response]:
        attempt = 0
        max_attempts = 3
        while attempt < max_attempts:
            attempt += 1
            try:
                with self._client.stream(method, url, *args, **kwargs) as response:
                    yield response
                    return
            except httpx.TransportError as err:
                warnings.warn(
                    f"*** API client {method.upper()} {url} stream failed due to {err} (attempt {attempt}/{max_attempts})"
                )
                if attempt == max_attempts:
                    raise

        raise RuntimeError(f"[API] Could not complete request {method.upper()} {url}")

    def _call(self, _method: str, _path: str, _schema, *args, **kwargs):
        """Meta-call method that runs whenever a named function is called on
        this namespace"""
        responses_schema = _schema.get("responses", {})
        _path, req = self._construct_request(_path, _schema, *args, **kwargs)

        # Submit a response and get a generator for it. Resulting res_gen should
        # only yield a single item. Cleaned up at the end of this function call.
        content_schema = responses_schema.get("201", {}).get("content", {})
        stream_mime_type = registry.first_streamable_mime(content_schema.keys())

        try:
            ctx = (
                self._request(_method, _path, **req)
                if stream_mime_type is None
                else self._request_stream(_method, _path, **req)
            )
            with ctx as res:
                return self._handle_response(_schema, res)
        except httpx.HTTPStatusError as err:
            raise APIError("received error response", res=err.response) from err


class APIClient(APINamespace):
    """Root API namespace interface for an OpenAPI server"""

    _namespace_class: Type[APINamespace] = APINamespace

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        auth: Optional[Auth] = None,  # token or email/password
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 300,
        http_client: Optional[httpx.Client] = None,
        form_encoding: FormEncoding = "urlencoded",
    ):
        if base_url and http_client:
            raise TypeError(f"Cannot specify both base_url ({base_url}) and http_client ({http_client})")
        if not http_client:
            if not base_url:
                raise TypeError(f"Must specify either base_url ({base_url}) or http_client ({http_client})")
            http_client = httpx.Client(base_url=base_url, timeout=timeout)
        http_client.headers.update(headers)
        super().__init__(http_client, form_encoding=form_encoding)
        self._attrs = set()
        self(auth=auth)  # query the OpenAPI server and populate endpoint functions

    def __del__(self):
        # Clean up client when it gets garbage collected
        self._client.close()

    def __call__(self, auth: Optional[Auth] = None):
        """
        Re-generate all routes and internal namespaces. Optionally accepts
        either an authentication bearer token or an email/password tuple. Note
        that the password should be encoded in sha256.
        """
        self._reset()
        try:
            with self._request("get", "/openapi.json") as res:
                res.raise_for_status()
                schema = res.json()
        except json.JSONDecodeError as e:
            raise ValueError("Error reading JSON response") from e
        self._process_schema(schema)
        if auth:
            self._authorize(auth)
        return schema

    def _reset(self):
        for attr in self._attrs:
            delattr(self, attr)
        self._attrs.clear()

    def _process_schema(self, schema):
        assert isinstance(schema, dict), "[API] Invalid OpenAPI schema response: Not a dictionary"
        for key in {"info", "paths", "components"}:
            assert key in schema, (
                f"[API] Invalid OpenAPI schema response: Missing '{key}' key, got keys {list(schema.keys())}"
            )

        for path, path_schema in schema["paths"].items():
            for method, endpoint_schema in path_schema.items():
                self._register_endpoint(method, path, endpoint_schema)

    def _register_endpoint(self, method: str, path: str, schema):
        """
        CryoSPARC's OpenAPI server is configured such that the "summary"
        property in the provided endpoint schemas at /openapi.json is the name
        of the function in python. '.' used to delimit namespaced endpoints.
        """
        namespace = self
        func_name: str = schema["summary"]
        if "." in func_name:
            namespace_name, func_name = func_name.split(".", 1)
            namespace = self._get_namespace(namespace_name)
        else:
            self._attrs.add(func_name)

        setattr(namespace, func_name, self._generate_endpoint(func_name, namespace, method, path, schema))

    def _generate_endpoint(self, func_name: str, namespace, method: str, path: str, schema):
        def endpoint(*args, **kwargs):
            return namespace._call(method, path, schema, *args, **kwargs)

        endpoint.__name__ = func_name
        return endpoint

    def _get_namespace(self, name: str):
        if not hasattr(self, name):
            setattr(self, name, self._namespace_class(self._client, form_encoding=self._form_encoding))
            self._attrs.add(name)
        namespace = getattr(self, name)
        assert isinstance(namespace, self._namespace_class), (
            f"{self} name conflict with namespace '{name}'. This is likely a bug"
        )
        return namespace

    def _authorize(self, auth: Auth):
        token = (
            Token(access_token=auth, token_type="bearer")
            if isinstance(auth, str)
            else self.login(grant_type="password", username=auth[0], password=auth[1])  # type: ignore
        )
        self._client.headers["Authorization"] = f"{token.token_type.title()} {token.access_token}"


def sort_params_schema(path: str, param_schema: List[dict]):
    """
    Sort the OpenAPI endpoint parameters schema in order that path params appear
    in the given URI.
    """
    path_params = {p["name"]: p for p in param_schema if p["in"] == "path"}
    known_path_params = re.findall(r"{([^}]*)}", path)
    return [path_params[name] for name in known_path_params] + [p for p in param_schema if p["in"] != "path"]


def _get_schema_param_name(schema: dict, default: str = "param") -> str:
    """
    Given a parameter schema, convert its title to a valid python argument
    identifier. Used to determine kwarg name of body arguments.
    """
    return schema.get("title", default).lower().replace(" ", "_")


def _matches_schema_type(value: Any, schema: dict) -> bool:
    if "$ref" in schema:
        model_class = registry.model_for_ref(schema["$ref"])
        if model_class:
            return True
    schema_type = schema.get("type")
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type in _BASE_RESPONSE_TYPES:
        return isinstance(value, (str, float, int, bool))
    return False


def _decode_json_response(value: Any, schema: dict):
    # Check for empty schema/value, means just return JSON-decoded value as-is
    if value is None or not schema:
        return value

    # Don't attempt parsing unions without a model
    if "anyOf" in schema:
        for subschema in schema["anyOf"]:
            if _matches_schema_type(value, subschema):
                try:
                    return _decode_json_response(value, subschema)
                except (TypeError, ValueError):
                    continue
        warnings.warn("[API] Warning: No union schemas matched response. Returning API result as plain object.")
        return value

    # Check for schema that links to one of our existing models
    if "$ref" in schema:
        model_class = registry.model_for_ref(schema["$ref"])
        if model_class and issubclass(model_class, Enum):
            return model_class(value)
        elif model_class and issubclass(model_class, dict):  # typed dict
            return model_class(**value)
        elif model_class:  # pydantic model
            # use model_validate in case validator result derives from subtype, e.g., Event model
            return model_class.model_validate(value)  # type: ignore
        warnings.warn(
            f"[API] Warning: Received API response with unregistered schema type {schema['$ref']}. "
            "Returning as plain object."
        )
        return value

    # Check for Base case types
    if "type" in schema and schema["type"] in _BASE_RESPONSE_TYPES:
        return value

    # Recursively decode list or tuple
    if "type" in schema and schema["type"] == "array":
        collection_type, items_key = (tuple, "prefixItems") if "prefixItems" in schema else (list, "items")
        return collection_type(_decode_json_response(item, schema[items_key]) for item in value)

    # Recursively decode object
    if "type" in schema and schema["type"] == "object":
        prop_schemas = schema.get("properties", {})
        default_prop_schema = schema.get("additionalProperties", {})
        return {
            key: _decode_json_response(val, prop_schemas.get(key, default_prop_schema)) for key, val in value.items()
        }

    # No other result found, return as plain JSON
    return value


def _uriencode(val: Any):
    # Encode any string-compatible value so that it may be used in a URI.
    return urllib.parse.quote(val if isinstance(val, (str, bytes)) else str(val))
