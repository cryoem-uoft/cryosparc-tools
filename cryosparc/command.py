"""
Provides classes and functions for communicating with CryoSPARC's command
servers. Generally should not be used directly.
"""
from contextlib import contextmanager
import json
import uuid
from typing import Optional, Type
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


class CommandClient:
    """
    Class for communicating with CryoSPARC's ``command_core`` and
    ``command_vis`` HTTP services.

    Upon initialization, gets a list of available JSONRPC_ endpoints and creates
    corresponding instance methods for each one. Reference of available methods
    for the ``command_core`` service (a.k.a. "cli") is  available in the
    `CryoSPARC Guide`_.

    Args:
        service (str, optional): Label for CryoSPARC Command service that this
            instance connects to and communicates with, e.g., ``command_core``,
            ``command_vis`` or ``command_rtp``
        host (str, optional): Domain name or IP address of CryoSPARC master.
            Defaults to "localhost".
        port (int, optional): Command server base port. Defaults to 39002.
        url (str, optional): Base URL path prefix for all requests (e.g., "/v1").
            Defaults to "".
        timeout (int, optional): How long to wait for a request to complete
            before timing out, in seconds. Defaults to 300.
        headers (dict, optional): Default HTTP headers to send with every
            request. Defaults to {}.
        cls (Type[JSONEncoder], optional): Class to handle JSON encoding of
            special Python objects, such as numpy arrays. Defaults to None.

    Attributes:

        service (str): label of CryoSPARC Command service this instance connects to
            and communicates with

    Examples:

        Connect to ``command_core``

        >>> from cryosparc.command import CommandClient
        >>> cli = CommandClient(
        ...     host="csmaster",
        ...     port=39002,
        ...     headers={"License-ID": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"}
        ... )

        Queue a job

        >>> cli.enqueue_job(project_uid="P3", job_uid="J42", lane="csworker")
        "launched"

    .. _JSONRPC:
        https://www.jsonrpc.org

    .. _CryoSPARC Guide:
        https://guide.cryosparc.com/setup-configuration-and-management/management-and-monitoring/cli

    """

    service: str

    class Error(BaseException):
        def __init__(self, parent: "CommandClient", reason: str, *args: object, url: str = "") -> None:
            msg = f"*** {type(parent).__name__}: ({url}) {reason}"
            super().__init__(msg, *args)

    def __init__(
        self,
        service: str = "command",
        host: str = "localhost",
        port: int = 39002,
        url: str = "",
        timeout: int = 300,
        headers: dict = {},
        cls: Optional[Type[json.JSONEncoder]] = None,
    ):
        self.service = service
        self._url = f"http://{host}:{port}{url}"
        self._cls = cls
        self._timeout = timeout
        self._headers = headers
        self._reload()  # attempt connection immediately to gather methods

    def _get_callable(self, key):
        def func(*args, **kwargs):
            params = kwargs if len(kwargs) else args
            data = {
                "jsonrpc": "2.0",
                "method": key,
                "params": params,
                "id": str(uuid.uuid4()),
            }
            res = None
            try:
                with make_json_request(self, "/api", data=data) as request:
                    res = json.loads(request.read())
            except CommandClient.Error as err:
                raise CommandClient.Error(
                    self, f'Did not receive a JSON response from method "{key}" with params {params}', url=self._url
                ) from err

            assert res, f'JSON response not received for method "{key}" with params {params}'
            assert "error" not in res, f'Error for "{key}" with params {params}:\n' + format_server_error(res["error"])
            return res["result"]

        return func

    def _reload(self):
        system = self._get_callable("system.describe")()
        self._endpoints = [p["name"] for p in system["procs"]]
        for key in self._endpoints:
            setattr(self, key, self._get_callable(key))

    def __call__(self):
        self._reload()


@contextmanager
def make_request(
    client: CommandClient, method: str = "post", url: str = "", query: dict = {}, data=None, headers: dict = {}
):
    """
    Create a raw HTTP request/response context with the given command client.

    Args:
        client (CommandClient): command client instance
        method (str, optional): HTTP method. Defaults to "post".
        url (str, optional): URL to append to the client's initialized URL. Defaults to "".
        query (dict, optional): Query string parameters. Defaults to {}.
        data (any, optional): Request body data. Usually in binary. Defaults to None.
        headers (dict, optional): HTTP headers. Defaults to {}.

    Raises:
        CommandClient.Error: General error such as timeout, URL or HTTP

    Yields:
        http.client.HTTPResponse:  Use with a context manager to get HTTP response

    Example:

        >>> from cryosparc.command import CommandClient, make_request
        >>> cli = CommandClient()
        >>> with make_request(cli, url="/download_file", query={'path': '/file.txt'}) as response:
        ...     data = response.read()

    """
    url = f"{client._url}{url}{'?' + urlencode(query) if query else ''}"
    headers = {"Originator": "client", **client._headers, **headers}
    attempt = 1
    max_attempts = 3
    error_reason = "<unknown>"
    while attempt < max_attempts:
        request = Request(url, data=data, headers=headers, method=method)
        response = None
        try:
            with urlopen(request, timeout=client._timeout) as response:
                yield response
                return
        except TimeoutError:
            error_reason = "Timeout Error"
            print(
                f"*** {type(client).__name__}: command ({url}) "
                f"did not reply within timeout of {client._timeout} seconds, "
                f"attempt {attempt} of {max_attempts}"
            )
            attempt += 1
        except HTTPError as error:
            error_reason = (
                f"HTTP Error {error.code} {error.reason}; "
                f"please check cryosparcm log {client.service} for additional information."
            )
            if error.readable():
                error_reason += "\nResponse from server: " + str(error.read())

            print(f"*** {type(client).__name__}: ({url}) {error_reason}")
            break
        except URLError as error:
            error_reason = f"URL Error {error.reason}"
            print(f"*** {type(client).__name__}: ({url}) {error_reason}")
            break

    raise CommandClient.Error(client, error_reason, url=url)


def make_json_request(client: CommandClient, url="", query={}, data=None, headers={}):
    """
    Similar to ``make_request``, except sends request body data JSON and
    receives arbitrary response.

    Args:
        client (CommandClient): command client instance
        url (str, optional): URL path to append to the client's initialized root
            URL. Defaults to "".
        query (dict, optional): Query string parameters. Defaults to {}.
        data (any, optional): JSON-encodable request body. Defaults to None.
        headers (dict, optional): HTTP headers. Defaults to {}.

    Yields:
        http.client.HTTPResponse: Use with a context manager to get HTTP response

    Example:

        >>> from cryosparc.command import CommandClient, make_json_request
        >>> cli = CommandClient()
        >>> with make_json_request(cli, url="/download_file", data={'path': '/file.txt'}) as response:
        ...     data = response.read()

    """
    headers = {"Content-Type": "application/json", **headers}
    data = json.dumps(data, cls=client._cls).encode()
    return make_request(client, url=url, query=query, data=data, headers=headers)


def format_server_error(error):
    err = error["message"] if "message" in error else str(error)
    if "data" in error and error["data"]:
        if isinstance(error["data"], dict) and "traceback" in error["data"]:
            err += "\n" + error["data"]["traceback"]
        else:
            err += "\n" + str(error["data"])
    return err
