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
    and upload/download project files. Should be instantiated through
    `CryoSPARC.find_project`_.

    .. _CryoSPARC.find_project:
        tools.html#cryosparc.tools.CryoSPARC.find_project
    """

    _doc: Optional[dict] = None

    def __init__(self, cs: "CryoSPARC", uid: str) -> None:
        self.cs = cs
        self.uid = uid

    @property
    def doc(self) -> dict:
        """
        doc: Raw project document from CryoSPARC database.
        """
        if not self._doc:
            self.refresh()
        assert self._doc, "Could not refresh project document"
        return self._doc

    def refresh(self):
        """
        Reload this project from the CryoSPARC database.

        Returns:
            Project: self
        """
        self._doc = self.cs.cli.get_project(self.uid)  # type: ignore
        return self

    def dir(self) -> PurePosixPath:
        """
        Get the path to the project directory.

        Returns:
            Path: project directory Pure Path instance
        """
        path: str = self.cs.cli.get_project_dir_abs(self.uid)  # type: ignore
        return PurePosixPath(path)

    def find_job(self, job_uid: str) -> Job:
        """
        Get a job accessor instance for the job in this project with the given
        UID. Fails with an error if job does not exist.

        Args:
            job_uid (str): Job unique ID, e.g., "J42"

        Returns:
            Job: accessor class
        """
        job = Job(self.cs, self.uid, job_uid)
        job.refresh()
        return job

    def find_external_job(self, job_uid: str) -> ExternalJob:
        """
        Get the External job accessor instance for an External job in this
        project with the given UID. Fails if the job does not exist or is not an
        external job.

        Args:
            job_uid (str): Job unique ID, e.g,. "J42"

        Raises:
            TypeError: If job is not an external job

        Returns:
            ExternalJob: accessor class
        """
        job = ExternalJob(self.cs, self.uid, job_uid)
        job.refresh()
        if job.doc["job_type"] != "snowflake":
            raise TypeError(f"Job {self.uid}-{job_uid} is not an external job")
        return job

    def create_workspace(self, title: str, desc: Optional[str] = None, user: Optional[str] = None) -> str:
        """
        Create a new empty workspace in this project. At least a title must be
        provided.

        Args:
            title (str): Title of new workspace
            desc (str, optional): Markdown text description. Defaults to None.
            user (str, optional): Unique identifier (email, _id) for user
                account creating this workspace. Set to project owner if not
                specified. Defaults to None.

        Returns:
            str: UID of created workspace.
        """
        user_id: str = self.cs.cli.get_user_id(user) if user else None  # type: ignore
        return self.cs.cli.create_empty_workspace(  # type: ignore
            project_uid=self.uid, created_by_user_id=user_id, title=title, desc=desc
        )

    def create_external_job(
        self,
        workspace_uid: Optional[str] = None,
        user: Optional[str] = None,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> ExternalJob:
        """
        Add a new External job to this project to save generated outputs to.

        Args:
            workspace_uid (str, optional): Workspace UID to create job in.
                Created in latest workspace if not specified. Defaults to None.
            user (str, optional): Unique identifier (email, _id) for user
                account creating this job. Set to project owner if not
                specified. Defaults to None.
            title (str, optional): Title for external job (recommended).
                Defaults to None.
            desc (str, optional): Markdown description for external job.
                Defaults to None.

        Returns:
            ExternalJob: created external job instance
        """
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
            >>> project.save_external_result(particles, 'particle', workspace_uid='W1')
            "J1"

            Save particle locations that inherit passthrough slots from a parent job

            >>> particles = Dataset()
            >>> project.save_external_result(
            ...     dataset=particles,
            ...     type='particle',
            ...     name='particles',
            ...     slots=['location'],
            ...     passthrough=('J42', 'selected_particles'),
            ...     user='ali@example.com',
            ...     title='Re-centered particles'
            ... )
            "J2"

        Args:
            dataset (Dataset): Result dataset.
            type (Datatype): Type of output dataset.
            name (str, optional): Name of output on created External job. Same
                as type if unspecified. Defaults to None.
            slots (list[str | Datafield], optional): List of slots expected to
                be created for this output such as ``location`` or ``blob``. Do
                not specify any slots that were passed through from an input
                unless those slots are modified in the output. Defaults to None.
            passthrough (tuple[str, str], optional): Indicates that this output
                inherits slots from the specified output. e.g.,
                ``("J1", "particles")``. Defaults to None.
            workspace_uid (str, optional): Workspace UID to save results into.
                If not specified, uses same workspace as passthrough or most
                recent workspace. Defaults to None.
            user (str, optional): Unique identifier (email, _id) of user account
                saving this output. Defaults to None.
            title (str, optional): Human-readable title for this output.
                Defaults to None.
            desc (str, optional): Markdown description for this output. Defaults
                to None.

        Returns:
            str: UID of created job where this output was saved
        """
        # Check slot names if present. If not provided, use all slots specified
        # in the dataset prefixes.
        prefixes = dataset.prefixes()
        if slots is None:
            slots = list(prefixes)
        slot_names = {s if isinstance(s, str) else s["prefix"] for s in slots}
        assert slot_names.intersection(prefixes) == slot_names, "Given dataset missing required slots"

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

    def download(self, path_rel: Union[str, PurePosixPath]):
        """
        Open a file in the current project for reading. Use to get files from a
        remote CryoSPARC instance where the project directory is not available
        on the client file system.

        Args:
            path (str | Path): Relative path of file in project directory.

        Yields:
            HTTPResponse: Use a context manager to read the file from the
            request body
        """
        return self.cs.download(self.uid, path_rel)

    def download_file(self, path_rel: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]]):
        """
        Download a file from the project directory to the given writeable target.

        Args:
            path_rel (str | Path): Relative path of file in project directory.
            target (str | Path | IO): Relative local path or writeable file
                handle to write response file into.

        Returns:
            str | Path | IO: resulting target path or file handle.
        """
        return self.cs.download_file(self.uid, path_rel, target)

    def download_dataset(self, path_rel: Union[str, PurePosixPath]):
        """
        Download a .cs dataset file frmo the given relative path in the project
        directory.

        Args:
            path_rel (str | Path): Realtive path to .cs file in project
            directory.

        Returns:
            Dataset: Loaded dataset instance
        """
        return self.cs.download_dataset(self.uid, path_rel)

    def download_mrc(self, path_rel: Union[str, PurePosixPath]):
        """
        Download a .mrc file from the given relative path in the project
        directory.

        Args:
            path_rel (str | Path): Relative path to .mrc file in project
                directory.

        Returns:
            tuple[Header, NDArray]: MRC file header and data as a numpy array
        """
        return self.cs.download_mrc(self.uid, path_rel)

    def upload(self, target_path_rel: Union[str, PurePosixPath], source: Union[str, PurePath, IO[bytes]]):
        """
        Upload the given file to the project directory at the given relative
        path.

        Args:
            target_path_rel (str | Path): Relative target path in project
                directory.
            source (str | Path | IO): Local path or file handle to upload.
        """
        return self.cs.upload(self.uid, target_path_rel, source)

    def upload_dataset(self, target_path_rel: Union[str, PurePosixPath], dset: Dataset):
        """
        Upload a dataset as a CS file into the project directory.

        Args:
            target_path_rel (str | Path): relative path to save dataset in
                project directory. Should have a ``.cs`` extension.
            dset (Dataset): dataset to save.
        """
        return self.cs.upload_dataset(self.uid, target_path_rel, dset)

    def upload_mrc(self, target_path_rel: Union[str, PurePosixPath], data: "NDArray", psize: float):
        """
        Upload a numpy 2D or 3D array to the job directory as an MRC file.

        Args:
            target_path_rel (str | Path): filename or relative path. Should have
                ``.mrc`` extension.
            data (NDArray): Numpy array with MRC file data.
            psize (float): Pixel size to include in MRC header.
        """
        return self.cs.upload_mrc(self.uid, target_path_rel, data, psize)
