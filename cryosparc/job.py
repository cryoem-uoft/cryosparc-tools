from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Optional, Union
from typing_extensions import Literal

from .spec import Datatype, Datafield, Datatype
from .dataset import Dataset
from .command import CommandClient


class Job:
    """
    Immutable reference to a job in cryoSPARC with ability to load inputs and
    outputs
    """

    def __init__(self, cli: CommandClient, project_uid: str, uid: str) -> None:
        self.cli = cli
        self.puid = project_uid
        self.juid = uid

    def dir(self) -> Path:
        """
        Get the path to the job directory
        """
        pass

    def load_input(self, name: str, fields: Optional[Iterable[str]] = None) -> Dataset:
        pass

    def load_output(self, name: str, fields: Optional[Iterable[str]] = None) -> Dataset:
        pass

    def save_output(self, name: str, dataset: Dataset):
        pass

    def log(self, text, level: Literal["text", "warning", "error"] = "text"):
        """
        Append to a job's event log
        """
        pass

class CustomJob(Job):
    """
    Mutable custom job with customizeble input slots and saveable results
    """

    def add_input(
        self,
        name: str,
        type: Datatype,
        title: Optional[str] = None,
        fields: Iterable[Union[str, Datafield]] = [],
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
    ):
        """
        Add an input slot to the current job.
        """
        pass

    def add_output(
        self,
        name: str,
        type: Optional[Datatype] = None,
        inherits: Optional[str] = None,
        title: Optional[str] = None,
        fields: Iterable[Union[str, Datafield]] = [],
        min: int = 0,
        max: Union[int, Literal["inf"]] = "inf",
    ):
        """
        Add an output slot to the current job.

        One of `type` or `inherits` must be specified, where `inherits` is the
        name of an existing input.
        """
        pass

    def connect(self, input: str, job_uid: str, output: str):
        """
        Connect the given input to an output with given job UID and name
        """
        pass

    @contextmanager
    def run(self):
        # TODO: Set job to running status
        try:
            yield self
            # TODO: Set job to completed status
        except:
            # TODO: Set job to error status
            pass
