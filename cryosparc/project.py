from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union


from .workspace import Workspace
from .job import Job, ExternalJob
from .dataset import Dataset, DEFAULT_FORMAT
from .row import R
from .spec import MongoController, Datafield, Datatype, ProjectDocument

if TYPE_CHECKING:
    from numpy.typing import NDArray  # type: ignore
    from .tools import CryoSPARC


class Project(MongoController[ProjectDocument]):
    """
    Accessor instance for CryoSPARC projects with ability to add workspaces, jobs
    and upload/download project files. Should be instantiated through
    `CryoSPARC.find_project`_.

    .. _CryoSPARC.find_project:
        tools.html#cryosparc.tools.CryoSPARC.find_project
    """

    def __init__(self, cs: "CryoSPARC", uid: str) -> None:
        self.cs = cs
        self.uid = uid

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

    def find_workspace(self, workspace_uid) -> Workspace:
        """
        Get a workspace accessor instance for the workspace in this project
        with the given UID. Fails with an error if workspace does not exist.

        Args:
            workspace_uid (str): Workspace unique ID, e.g., "W1"

        Returns:
            Workspace: accessor instance
        """
        workspace = Workspace(self.cs, self.uid, workspace_uid)
        return workspace.refresh()

    def find_job(self, job_uid: str) -> Job:
        """
        Get a job accessor instance for the job in this project with the given
        UID. Fails with an error if job does not exist.

        Args:
            job_uid (str): Job unique ID, e.g., "J42"

        Returns:
            Job: accessor instance
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
            ExternalJob: accessor instance
        """
        return self.cs.find_external_job(self.uid, job_uid)

    def create_workspace(self, title: str, desc: Optional[str] = None) -> Workspace:
        """
        Create a new empty workspace in this project. At least a title must be
        provided.

        Args:
            title (str): Title of new workspace
            desc (str, optional): Markdown text description. Defaults to None.

        Returns:
            Workspace: created workspace instance
        """
        return self.cs.create_workspace(self.uid, title, desc)

    def create_job(
        self,
        workspace_uid: str,
        type: str,
        connections: Dict[str, Tuple[str, str]] = {},
        params: Dict[str, Any] = {},
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> Job:
        """
        Create a new job with the given type. Use `CryoSPARC.get_job_sections`_
        to query available job types on the connected CryoSPARC instance.

        Args:
            project_uid (str): Project UID to create job in, e.g., "P3"
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            type (str): Job type identifier, e.g., "homo_abinit"
            connections (dict[str, tuple[str, str]]): Initial input connections.
                Each key is an input name and each value is a (job uid, output
                name) tuple. Defaults to {}
            params (dict[str, any], optional): Specify parameter values.
                Defaults to {}.
            title (str, optional): Job title. Defaults to None.
            desc (str, optional): Job markdown description. Defaults to None.

        Returns:
            Job: created job instance. Raises error if job cannot be created.

        Examples:

            Create an Import Movies job.

            >>> from cryosparc.tools import CryoSPARC
            >>> cs = CryoSPARC()
            >>> project = cs.find_project("P3")
            >>> import_job = project.create_job("W1", "import_movies")
            >>> import_job.set_param("blob_paths", "/bulk/data/t20s/*.tif")
            True

            Create a 3-class ab-initio job connected to existing particles.

            >>> abinit_job = project.create_job("W1", "homo_abinit"
            ...     connections={"particles": ("J20", "particles_selected")}
            ...     params={"abinit_K": 3}
            ... )

        .. _CryoSPARC.get_job_sections:
            tools.html#cryosparc.tools.CryoSPARC.get_job_sections
        """
        return self.cs.create_job(
            self.uid, workspace_uid, type, connections=connections, params=params, title=title, desc=desc
        )

    def create_external_job(
        self,
        workspace_uid: str,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> ExternalJob:
        """
        Add a new External job to this project to save generated outputs to.

        Args:
            workspace_uid (str): Workspace UID to create job in, e.g., "W3".
            title (str, optional): Title for external job (recommended).
                Defaults to None.
            desc (str, optional): Markdown description for external job.
                Defaults to None.

        Returns:
            ExternalJob: created external job instance
        """
        job_uid: str = self.cs.vis.create_external_job(  # type: ignore
            project_uid=self.uid, workspace_uid=workspace_uid, user=self.cs.user_id, title=title, desc=desc
        )
        return self.find_external_job(job_uid)

    def save_external_result(
        self,
        workspace_uid: Optional[str],
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[Union[str, Datafield]]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> str:
        """
        Save the given result dataset to the project. Specify at least the
        dataset to save and the type of data.

        If ``workspace_uid`` or ``passthrough`` input is not specified or is
        None, saves the result to the project's newest workspace. If
        ``passthrough`` is specified but ``workspace_uid`` is not, saves to the
        passthrough job's newest workspace.

        Returns UID of the External job where the results were saved.

        Examples:

            Save all particle data

            >>> particles = Dataset()
            >>> project.save_external_result("W1", particles, 'particle')
            "J43"

            Save new particle locations that inherit passthrough slots from a
            parent job

            >>> particles = Dataset()
            >>> project.save_external_result(
            ...     workspace_uid='W1',
            ...     dataset=particles,
            ...     type='particle',
            ...     name='particles',
            ...     slots=['location'],
            ...     passthrough=('J42', 'selected_particles'),
            ...     title='Re-centered particles'
            ... )
            "J44"

            Save a result with multiple slots of the same type.

            >>> project.save_external_result(
            ...     workspace_uid="P1",
            ...     dataset=particles,
            ...     type="particle",
            ...     name="particle_alignments",
            ...     slots=[
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_0", "required": True},
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_1", "required": True},
            ...         {"dtype": "alignments3D", "prefix": "alignments_class_2", "required": True},
            ...     ]
            ... )
            "J45"

        Args:
            workspace_uid (str | None): Workspace UID to save results into.
                Specify ``None`` to auto-select a workspace.
            dataset (Dataset): Result dataset.
            type (Datatype): Type of output dataset.
            name (str, optional): Name of output on created External job. Same
                as type if unspecified. Defaults to None.
            slots (list[str | Datafield], optional): List of slots expected to
                be created for this output such as ``location`` or ``blob``. Do
                not specify any slots that were passed through from an input
                unless those slots are modified in the output. Defaults to None.
            passthrough (tuple[str, str], optional): Indicates that this output
                inherits slots from the specified output. e.g., ``("J1",
                "particles")``. Defaults to None.
            title (str, optional): Human-readable title for this output.
                Defaults to None.
            desc (str, optional): Markdown description for this output. Defaults
                to None.

        Returns:
            str: UID of created job where this output was saved
        """
        return self.cs.save_external_result(
            self.uid,
            workspace_uid,
            dataset=dataset,
            type=type,
            name=name,
            slots=slots,
            passthrough=passthrough,
            title=title,
            desc=desc,
        )

    def list_files(self, prefix: Union[str, PurePosixPath] = "", recursive: bool = False) -> List[str]:
        """
        Get a list of files inside the project directory.

        Args:
            prefix (str | Path, optional): Subdirectory inside project to list.
                Defaults to "".
            recursive (bool, optional): If True, lists files recursively.
                Defaults to False.

        Returns:
            list[str]: List of file paths relative to the project directory.
        """
        return self.cs.list_files(self.uid, prefix=prefix, recursive=recursive)

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
            target (str | Path | IO): Local file path, directory path or writeable
                file handle to write response data.

        Returns:
            Path | IO: resulting target path or file handle.
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

    def upload(self, target_path_rel: Union[str, PurePosixPath], source: Union[str, bytes, PurePath, IO]):
        """
        Upload the given file to the project directory at the given relative
        path.

        Args:
            target_path_rel (str | Path): Relative target path in project
                directory.
            source (str | bytes | Path | IO): Local path or file handle to
                upload. May also specified as raw bytes.
        """
        return self.cs.upload(self.uid, target_path_rel, source)

    def upload_dataset(self, target_path_rel: Union[str, PurePosixPath], dset: Dataset, format: int = DEFAULT_FORMAT):
        """
        Upload a dataset as a CS file into the project directory.

        Args:
            target_path_rel (str | Path): relative path to save dataset in
                project directory. Should have a ``.cs`` extension.
            dset (Dataset): dataset to save.
            format (int): format to save in from ``cryosparc.dataset.*_FORMAT``,
                defaults to NUMPY_FORMAT)

        """
        return self.cs.upload_dataset(self.uid, target_path_rel, dset, format=format)

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

    def mkdir(
        self,
        target_path_rel: Union[str, PurePosixPath],
        parents: bool = False,
        exist_ok: bool = False,
    ):
        """
        Create a directory in the given project.

        Args:
            target_path_rel (str | Path): Relative path to create inside project
                directory.
            parents (bool, optional): If True, any missing parents are created
                as needed. Defaults to False.
            exist_ok (bool, optional): If True, does not raise an error for
                existing directories. Still raises if the target path is not a
                directory. Defaults to False.
        """
        self.cs.mkdir(
            project_uid=self.uid,
            target_path_rel=target_path_rel,
            parents=parents,
            exist_ok=exist_ok,
        )

    def cp(self, source_path_rel: Union[str, PurePosixPath], target_path_rel: Union[str, PurePosixPath]):
        """
        Copy a file or folder within a project to another location within that
        same project. Note that argument order is reversed from
        equivalent ``cp`` command.

        Args:
            source_path_rel (str | Path): Relative path in project of source
                file or folder to copy.
            target_path_rel (str | Path): Relative path in project to copy to.
        """
        self.cs.cp(
            project_uid=self.uid,
            source_path_rel=source_path_rel,
            target_path_rel=target_path_rel,
        )

    def symlink(self, source_path_rel: Union[str, PurePosixPath], target_path_rel: Union[str, PurePosixPath]):
        """
        Create a symbolic link in the given project. May only create links for
        files within the project. Note that argument order is reversed from
        ``ln -s``.

        Args:
            project_uid (str): Target project UID, e.g., "P3".
            target_path_rel (str | Path): Relative path in project to new
                symlink.
            source_path_rel (str | Path): Relative path in project to file from
                which to create symlink.
        """
        self.cs.symlink(
            project_uid=self.uid,
            source_path_rel=source_path_rel,
            target_path_rel=target_path_rel,
        )
