from contextlib import contextmanager
import json
import uuid
from typing import Generator, Optional, Type
from http.client import HTTPResponse
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


class RequestClient:
    """
    Simple HTTP POST request client that sends data and receives an arbitary
    response. Supports retrying connection upon failing the first few times.
    """

    class Error(BaseException):
        def __init__(self, parent: "RequestClient", reason: str, *args: object, url: str = "") -> None:
            msg = f"*** {type(parent).__name__}: ({url}) {reason}"
            super().__init__(msg, *args)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 39000,
        url: str = "",
        headers: dict = {},
        timeout: int = 300,
        cls: Optional[Type[json.JSONEncoder]] = None,
    ):
        self.url = f"http://{host}:{port}{url}"
        self.cls = cls
        self.timeout = timeout
        self.headers = headers

    @contextmanager
    def _request(self, url="", query: dict = {}, data=None, headers={}) -> Generator[HTTPResponse, None, None]:
        url = f"{self.url}{url}{'?' + urlencode(query) if query else ''}"
        headers = {"Originator": "client", **self.headers, **headers}
        attempt = 1
        max_attempts = 3
        error_reason = "<unknown>"
        while attempt < max_attempts:
            request = Request(
                url,
                data=data,
                headers=headers,
            )
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    yield response
                    return
            except HTTPError as error:
                error_reason = f"HTTP Error {error.code} {error.reason}"
                print(f"*** {type(self).__name__}: ({url}) {error_reason}")
                raise
            except URLError as error:
                error_reason = f"URL Error {error.reason}"
                print(f"*** {type(self).__name__}: ({url}) {error_reason}")
                raise
            except TimeoutError as error:
                error_reason = "Timeout Error"
                print(
                    f"*** {type(self).__name__}: command ({url}) did not reply within timeout of {self.timeout} seconds, "
                    f"attempt {attempt} of {max_attempts}"
                )
                attempt += 1

        raise RequestClient.Error(self, error_reason, url=url)

    def _json_request(self, url="", query={}, data=None, headers={}):
        """
        Sends data in JSON, receives, arbitrary response
        """
        headers = {"Content-Type": "application/json", **headers}
        data = json.dumps(data, cls=self.cls).encode()
        return self._request(url=url, query=query, data=data, headers=headers)


class CommandClient(RequestClient):
    """
    Class of communicating with cryoSPARC's command_core and command_rtp
    HTTP services
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 39002,
        url: str = "",
        timeout: int = 300,
        headers: dict = {},
        cls: Optional[Type[json.JSONEncoder]] = None,
    ):
        super().__init__(host, port, url, headers, timeout, cls)
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
                with self._json_request("/api", data=data) as request:
                    res = json.loads(request.read())
            except RequestClient.Error as err:
                raise RequestClient.Error(
                    self, f'Did not receive a JSON response from method "{key}" with params {params}', url=self.url
                ) from err

            assert res, f'JSON response not received for method "{key}" with params {params}'
            assert "error" not in res, f'Error for "{key}" with params {params}:\n' + str(res["error"])
            return res["result"]

        return func

    def _reload(self):
        system = self._get_callable("system.describe")()
        self.endpoints = [p["name"] for p in system["procs"]]
        for key in self.endpoints:
            setattr(self, key, self._get_callable(key))

    def __call__(self):
        self._reload()
