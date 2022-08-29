from contextlib import contextmanager
import json
import uuid
from typing import Generator, Optional, Type
from http.client import HTTPResponse
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


class CommandClient:
    """
    Class of communicating with cryoSPARC's command_core and command_rtp
    HTTP services
    """

    class Error(BaseException):
        def __init__(self, parent: "CommandClient", reason: str, *args: object, url: str = "") -> None:
            msg = f"*** {type(parent).__name__}: ({url}) {reason}"
            super().__init__(msg, *args)

    def __init__(
        self,
        host: str = "localhost",
        port: int = 39002,
        url: str = "",
        timeout: int = 300,
        headers: dict = {},
        cls: Optional[Type[json.JSONEncoder]] = None,
    ):
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
            assert "error" not in res, f'Error for "{key}" with params {params}:\n' + str(res["error"])
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
    client: CommandClient, url="", query: dict = {}, data=None, headers={}
) -> Generator[HTTPResponse, None, None]:
    """
    Create a raw request/response context with the given command client.

    Usage:

        cli = CommandClient()
        with make_request(cli, url="/download_file", query={'path': '/file.txt'}) as response:
            data = response.read()

    """
    url = f"{client._url}{url}{'?' + urlencode(query) if query else ''}"
    headers = {"Originator": "client", **client._headers, **headers}
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
            with urlopen(request, timeout=client._timeout) as response:
                yield response
                return
        except HTTPError as error:
            error_reason = f"HTTP Error {error.code} {error.reason}"
            print(f"*** {type(client).__name__}: ({url}) {error_reason}")
            raise
        except URLError as error:
            error_reason = f"URL Error {error.reason}"
            print(f"*** {type(client).__name__}: ({url}) {error_reason}")
            raise
        except TimeoutError as error:
            error_reason = "Timeout Error"
            print(
                f"*** {type(client).__name__}: command ({url}) did not reply within timeout of {client._timeout} seconds, "
                f"attempt {attempt} of {max_attempts}"
            )
            attempt += 1

    raise CommandClient.Error(client, error_reason, url=url)


def make_json_request(client: CommandClient, url="", query={}, data=None, headers={}):
    """
    Similar to make_request, except sends request body data JSON and receives
    arbitrary response.

    Usage:

        cli = CommandClient()
        with make_json_request(cli, url="/download_file", data={'path': '/file.txt'}) as response:
            data = response.read()

    """
    headers = {"Content-Type": "application/json", **headers}
    data = json.dumps(data, cls=client._cls).encode()
    return make_request(client, url=url, query=query, data=data, headers=headers)
