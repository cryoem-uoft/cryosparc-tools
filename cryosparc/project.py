from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, List, Optional, Tuple, Union

from .job import Job, ExternalJob
from .dataset import Dataset
from .row import R
from .spec import Datafield, Datatype

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore
    from .tools import CryoSPARC


class Project:
    """
    Accessor class for CryoSPARC projects with ability to add workspaces, jobs
    and upload/download project files.
    """

    def __init__(self, cs: "CryoSPARC", uid: str) -> None:
        self.cs = cs
        self.uid = uid
        self._doc = {}

    @property
    def doc(self) -> dict:
        if not self._doc:
            self.refresh()
        return self._doc

    def refresh(self):
        self._doc = self.cs.cli.get_project(self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory
        """
        path: str = self.cs.cli.get_project_dir_abs(self.uid)  # type: ignore
        return PurePosixPath(path)

    def find_job(self, job_uid: str) -> Job:
        job = Job(self.cs, self.uid, job_uid)
        job.refresh()
        return job

    def find_external_job(self, job_uid: str) -> ExternalJob:
        job = ExternalJob(self.cs, self.uid, job_uid)
        job.refresh()
        if job.doc["job_type"] != "snowflake":
            raise TypeError(f"Job {self.uid}-{job_uid} is not an external job")
        return job

    def create_workspace(self, title: str, desc: Optional[str] = None, user: Optional[str] = None) -> str:
        """
        Create a new empty workspace in this project. At least a title must be
        provided.
        """
        user_id: str = self.cs.cli.get_user_id(user) if user else None  # type: ignore
        return self.cs.cli.create_empty_workspace(  # type: ignore
            project_uid=self.uid, created_by_user_id=user_id, title=title, desc=desc
        )

    def create_external_job(
        self,
        workspace_uid: Optional[str],
        user: Optional[str] = None,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> ExternalJob:
        job_uid: str = self.cs.vis.create_external_job(  # type: ignore
            project_uid=self.uid, workspace_uid=workspace_uid, user=user, title=title, desc=desc
        )
        return self.find_external_job(job_uid)

    def save_external_result(
        self,
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[Union[str, Datafield]]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        workspace_uid: Optional[str] = None,
        user: Optional[str] = None,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> str:
        """
        Save the given result dataset to the project. Specify at least the
        dataset to save and the type of data.

        If `workspace_uid` or `passthrough` input is not specified, saves the
        result to the project's newest workspace. If `passthrough` is
        specified but `workspace_uid` is not, saves to the passthrough job's
        newest workspace.

        Returns UID of the External job where the results were saved.

        Examples:
            Save all particle data
            >>> particles = Dataset()
            >>> juid = project.save_external_result(particles, 'particle', workspace_uid='W1')

            Save particle locations that inherit passthrough slots from a parent job
            >>> dataset = Dataset()
            >>> juid = project.save_external_result(
            ...     dataset=particles,
            ...     type='particle',
            ...     name='particles',
            ...     slots=['location'],
            ...     passthrough=('J42', 'selected_particles'),
            ...     user='ali@example.com'
            ...     title='Re-centered particles'
            ... )
        """
        # Check slot names if present. If not provided, use all slots specified
        # in the dataset prefixes.
        prefixes = dataset.prefixes()
        if slots is None:
            slots = list(prefixes)
        slot_names = {s if isinstance(s, str) else s["prefix"] for s in slots}
        assert slot_names.intersection(prefixes) == slot_names, f"Given dataset missing required slots"

        passthrough_str = ".".join(passthrough) if passthrough else None
        job_uid, output = self.cs.vis.create_external_result(  # type: ignore
            project_uid=self.uid,
            workspace_uid=workspace_uid,
            type=type,
            name=name,
            slots=slots,
            passthrough=passthrough_str,
            user=user,
            title=title,
            desc=desc,
        )

        job = self.find_external_job(job_uid)
        with job.run():
            job.save_output(output, dataset)

        return job.uid

    def download(self, path: Union[str, PurePosixPath]):
        return self.cs.download(self.uid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        return self.cs.download_file(self.uid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        return self.cs.download_dataset(self.uid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        return self.cs.download_mrc(self.uid, path)

    def upload(self, path: Union[str, PurePosixPath], file: Union[str, PurePath, IO[bytes]]):
        return self.cs.upload(self.uid, path, file)

    def upload_dataset(self, path: Union[str, PurePosixPath], dset: Dataset):
        return self.cs.upload_dataset(self.uid, path, dset)

    def upload_mrc(self, path: Union[str, PurePosixPath], data: "NDArray", psize: float):
        return self.cs.upload_mrc(self.uid, path, data, psize)
