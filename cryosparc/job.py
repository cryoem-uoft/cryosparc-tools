from typing import Iterable, Optional, Union
from typing_extensions import Literal

from .spec import Datatype, Datafield, Datatype
from .dataset import Dataset
from .client import CommandClient

class Job:
    def __init__(self, cli: CommandClient, project_uid: str, uid: str) -> None:
        self.cli = cli
        self.puid = project_uid
        self.juid = uid

    def load_input(self, name: str, fields: Optional[Iterable[str]] = None) -> Dataset:
        pass

    def load_output(self, name: str, fields: Optional[Iterable[str]] = None) -> Dataset:
        pass


class CustomJob(Job):
    def add_input(self, name: str, type: Datatype, fields=Iterable[Union[str, Datafield]], min: int = 0, max: Union[int, Literal['inf']] = 'inf', title: Optional[str] = None):
        pass

    def add_output(self, name: str, type: Optional[Datatype] = None, inherits: Optional[str] = None, fields=Iterable[Union[str, Datafield]] = [], min: int = 0, max: Union[int, Literal['inf']] = 'inf', title: Optional[str] = None):
        pass


    def connect(self, input: str, job_uid: str, output: str):
        pass
