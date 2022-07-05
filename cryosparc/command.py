import json
import uuid
from typing import Optional, Type
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request


class CommandClient:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 39002,
        url: str = "/api",
        timeout: int = 300,
        cls: Optional[Type[json.JSONEncoder]] = None,
    ):
        """Attempts connection immediately"""
        self.url = f"http://{host}:{port}{url}"
        self.timeout = timeout
        self.cls = cls
        self.__reload__()

    def __get_callable__(self, key):
        def func(*args, **kwargs):
            params = kwargs if len(kwargs) else args
            data = {
                "jsonrpc": "2.0",
                "method": key,
                "params": params,
                "id": str(uuid.uuid4()),
            }
            headers = {"Content-Type": "application/json", "Originator": "client"}
            attempt = 1
            max_attempts = 3
            done = False
            res = None
            error_reason = "<unknown>"
            while not done:
                request = Request(
                    self.url,
                    data=json.dumps(data, cls=self.cls).encode(),
                    headers=headers,
                )
                try:
                    with urlopen(request, timeout=self.timeout) as response:
                        res = json.loads(response.read())
                        done = True
                except HTTPError as error:
                    error_reason = f"HTTP Error {error.code} {error.reason}"
                    print(f"*** CommandClient: ({self.url}) {error_reason}")
                    done = True
                except URLError as error:
                    error_reason = f"URL Error {error.reason}"
                    print(f"*** CommandClient: ({self.url}) {error_reason}")
                    done = True
                except TimeoutError as error:
                    error_reason = "Timeout Error"
                    print(
                        f"*** CommandClient: command ({self.url}) did not reply within timeout of {self.timeout} seconds, attempt {attempt} of {max_attempts}"
                    )
                    if attempt < max_attempts:
                        attempt += 1
                    else:
                        raise
            assert (
                res
            ), f'Did not receive a JSON response from method "{key}" with params {params} due to {error_reason}'
            assert (
                "error" not in res
            ), f"Encountered error for method \"{key}\" with params {params}:\n{res['error']['message'] if 'message' in res['error'] else res['error']}"
            return res["result"]

        return func

    def __reload__(self):
        system = self.__get_callable__("system.describe")()
        self.endpoints = [p["name"] for p in system["procs"]]
        for key in self.endpoints:
            setattr(self, key, self.__get_callable__(key))

    def __call__(self):
        self.__reload__()
