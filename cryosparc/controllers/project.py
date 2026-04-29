import time
import warnings
from pathlib import PurePath, PurePosixPath
from typing import IO, TYPE_CHECKING, Any, Dict, Iterable, List, Literal, Optional, Tuple, Union

from typing_extensions import Unpack

from ..dataset import DEFAULT_FORMAT, Dataset
from ..dataset.row import R
from ..errors import APIError, ProjectError
from ..models.project import Project, ProjectSymlink
from ..search import In, JobSearch
from ..spec import Datatype, SlotSpec
from ..util import PurePosixPathProperty
from . import Controller, as_output_slot
from .job import ExternalJobController, FileOrFigure, JobController
from .workspace import WorkspaceController

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..tools import CryoSPARC


class ProjectController(Controller[Project]):
    """
    Accessor instance for CryoSPARC projects with ability to add workspaces, jobs
    and upload/download project files. Should be created with
    :py:meth:`cs.find_project() <cryosparc.tools.CryoSPARC.find_project>`.

    Arguments:
        project (str | Project): either Project UID or Project model, e.g. ``"P3"``

    Attributes:
        model (Project): All project data from the CryoSPARC database. Contents
            may change over time, use :py:meth:`refresh` to update.
    """

    uid: str
    """
    Project unique ID, e.g., "P3"
    """

    def __init__(self, cs: "CryoSPARC", project: Union[str, Project]) -> None:
        self.cs = cs
        if isinstance(project, str):
            self.uid = project
            self.refresh()
        else:
            self.uid = project.uid
            self.model = project

    def refresh(self):
        """
        Reload this project from the CryoSPARC database.

        Returns:
            ProjectController: self
        """
        self.model = self.cs.api.projects.find_one(self.uid)
        return self

    @property
    def title(self) -> str:
        """Project title"""
        return self.model.title

    @property
    def desc(self) -> str:
        """Project description"""
        return self.model.description

    @property
    def dir(self) -> PurePosixPath:
        """Full path to project directory."""
        return PurePosixPathProperty(self.cs.api.projects.get_directory(self.uid))

    def set_title(self, title: str):
        """
        Set the project title.

        Args:
            title (str): New project title
        """
        self.model = self.cs.api.projects.set_title(self.uid, title=title)

    def set_description(self, desc: str):
        """
        Set the project description. May include Markdown formatting.

        Args:
            desc (str): New project description
        """
        self.model = self.cs.api.projects.set_description(self.uid, description=desc)

    def find_workspaces(self, *, order: Literal[1, -1] = 1) -> Iterable[WorkspaceController]:
        """
        Get all workspaces available in the current project.

        Returns:
            Iterable[WorkspaceController]: workspace accessor objects
        """
        return self.cs.find_workspaces(self.uid, order=order)

    def find_workspace(self, workspace_uid: str) -> WorkspaceController:
        """
        Get a workspace accessor instance for the workspace in this project
        with the given UID. Fails with an error if workspace does not exist.

        Args:
            workspace_uid (str): Workspace unique ID, e.g., "W1"

        Returns:
            WorkspaceController: workspace accessor object
        """
        return WorkspaceController(self.cs, (self.uid, workspace_uid))

    def find_jobs(
        self,
        workspace_uid: Optional[In[str]] = None,
        *,
        order: Literal[1, -1] = 1,
        **search: Unpack[JobSearch],
    ) -> Iterable[JobController]:
        """
        Search jobs available in the current project.

        Example:
            >>> jobs = project.find_jobs("W3")
            >>> jobs = project.find_jobs(["W3", "W4"])
            >>> jobs = project.find_jobs(
            ...     type="homo_reconstruct",
            ...     completed_at=(datetime(2025, 3, 1), datetime(2025, 3, 31)),
            ...     order=-1,
            ... )
            >>> for job in jobs:
            ...     print(job.uid)

        Args:
            workspace_uid (str | list[str] | None): Workspace unique ID, e.g.,
                "W1". If not specified, returns jobs from all workspaces.
                Defaults to None.
            **search (JobSearch): Additional search parameters to filter jobs,
                specified as keyword arguments.

        Returns:
            Iterable[JobController]: job accessor objects
        """
        return self.cs.find_jobs(self.uid, workspace_uid, order=order, **search)

    def find_job(self, job_uid: str) -> JobController:
        """
        Get a job accessor instance for the job in this project with the given
        UID. Fails with an error if job does not exist.

        Args:
            job_uid (str): Job unique ID, e.g., "J42"

        Returns:
            JobController: job accessor instance
        """
        return JobController(self.cs, (self.uid, job_uid))

    def find_external_job(self, job_uid: str) -> ExternalJobController:
        """
        Get the External job accessor instance for an External job in this
        project with the given UID. Fails if the job does not exist or is not an
        external job.

        Args:
            job_uid (str): Job unique ID, e.g,. "J42"

        Raises:
            TypeError: If job is not an external job

        Returns:
            ExternalJobController: external job accessor object
        """
        return self.cs.find_external_job(self.uid, job_uid)

    def find_symlinks(self) -> List[ProjectSymlink]:
        """
        Get a list of symlinks for this project. Symlinks are files or folders
        that are linked into the project directory but stored elsewhere on the
        file system.

        Returns:
            list[ProjectSymlink]: List of symlinks for this project, each
                including their source paths and target paths.
        """
        return self.cs.api.projects.get_symlinks(self.uid)

    def move(self, path: Union[str, PurePath], *, wait: bool = False):
        """
        Move this project to a new location on the file system. This is useful
        for moving projects to long-term storage locations. The project directory
        will be moved to the given path and the project will continue to be
        accessible through the same project UID.

        Args:
            path (str | Path): New file system path for the project directory.
            wait (bool, optional): If True, wait for the move operation to
            complete before returning. Defaults to False.
        """
        self.cs.api.projects.move(self.uid, path=str(path))
        self.model.import_status = "moving"  # backend will set this on next refresh
        while wait and self.model.import_status == "moving":
            time.sleep(3)
            self.refresh()

    def archive(self, *, wait: bool = False):
        """
        Archive this project. Archived projects are hidden from the CryoSPARC UI
        and cannot be modified. An admin may move their directories to a long-
        term storage location. Project may be restored with the :py:meth:`unarchive`
        function, which requires the path to its current location.

        Args:
            wait (bool, optional): If False, waits for the archive operation to
                complete before returning. Defaults to False.
        """
        self.cs.api.projects.archive(self.uid)
        while wait and not self.model.archived:
            time.sleep(3)
            self.refresh()

    def unarchive(self, path: Union[str, PurePath]):
        """
        Unarchive this project. See :py:meth:`archive` for details.

        Args:
            path (str | Path): Current file system path to the project directory.
        """
        self.model = self.cs.api.projects.unarchive(self.uid, path=str(path))

    def attach(self, *, wait: bool = False):
        """
        Attach this project to the CryoSPARC instance. Attached projects are
        accessible through the UI and can be modified.

        Args:
            wait (bool, optional): If True, wait for the attach operation to
                complete before returning. Defaults to False.
        """
        new_project = self.cs.attach_project(self.dir, wait=wait)
        self.uid = new_project.uid
        self.model = new_project.model

    def detach(self, *, wait: bool = False):
        """
        Detach this project from the CryoSPARC instance. Detached projects are
        not accessible through the UI and may be attached to other instances.

        See :py:meth:`cs.attach_project() <cryosparc.tools.CryoSPARC.attach_project>`
        to re-attach.

        Args:
            wait (bool, optional): If True, wait for the detach operation to
                complete before returning. Defaults to False.
        """
        self.cs.api.projects.detach(self.uid)
        while wait and not self.model.detached:
            time.sleep(3)
            self.refresh()

    def delete(self, *, wait: bool = False):
        """
        Delete this project from disk. Deleted projects cannot be recovered.

        Args:
            wait (bool, optional): If True, wait for the delete operation to
                complete before returning. Defaults to False.

        Raises:
            ProjectError: If project could not be deleted.
                See cryosparcm log api for details.
        """
        self.cs.api.projects.delete(self.uid)
        self.model.deleting = True  # backend will set this on next refresh
        while wait and self.model.deleting:
            time.sleep(3)
            try:
                self.refresh()
            except APIError as err:
                if err.code == 404:
                    return  # not found, project successfully deleted
                raise
        if not self.model.deleted and not self.model.deleting:
            raise ProjectError("Could not be deleted. See cryosparcm log api for details.", project=self)

    def accept_failed_attach(self):
        """
        Accept a failed attach for this project. This allows the project to be
        visible and modifiable again after a failed attach attempt. Some
        job or workspace data may be inaccessible or lost after a failed attach,
        so use with caution.
        """
        self.model = self.cs.api.projects.accept_failed_attach(self.uid)

    def create_workspace(self, title: str, desc: Optional[str] = None) -> WorkspaceController:
        """
        Create a new empty workspace in this project. At least a title must be
        provided.

        Args:
            title (str): Title of new workspace
            desc (str, optional): Markdown text description. Defaults to None.

        Returns:
            WorkspaceController: created workspace accessor object

        Raises:
            APIError: Workspace cannot be created.
        """
        return self.cs.create_workspace(self.uid, title, desc)

    def create_job(
        self,
        workspace_uid: str,
        type: str,
        connections: Dict[str, Union[Tuple[str, str], List[Tuple[str, str]]]] = {},
        params: Dict[str, Any] = {},
        title: str = "",
        desc: str = "",
    ) -> JobController:
        """
        Create a new job with the given type. Use
        :py:attr:`cs.job_register <cryosparc.tools.CryoSPARC.job_register>`
        to find available job types on the connected CryoSPARC instance.

        Args:
            project_uid (str): Project UID to create job in, e.g., "P3"
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            type (str): Job type identifier, e.g., "homo_abinit"
            connections (dict[str, tuple[str, str] | list[tuple[str, str]]]):
                Initial input connections. Each key is an input name and each
                value is a (job uid, output name) tuple. Defaults to {}
            params (dict[str, any], optional): Specify parameter values.
                Defaults to {}.
            title (str, optional): Job title. Defaults to "".
            desc (str, optional): Job markdown description. Defaults to "".

        Returns:
            JobController: created job accessor object.

        Raises:
            APIError: Job cannot be created.

        Examples:

            Create an Import Movies job.

            >>> from cryosparc.tools import CryoSPARC
            >>> cs = CryoSPARC("http://localhost:61000")
            >>> project = cs.find_project("P3")
            >>> import_job = project.create_job("W1", "import_movies")
            >>> import_job.set_param("blob_paths", "/bulk/data/t20s/*.tif")
            True

            Create a 3-class ab-initio job connected to existing particles.

            >>> abinit_job = project.create_job("W1", "homo_abinit"
            ...     connections={"particles": ("J20", "particles_selected")}
            ...     params={"abinit_K": 3}
            ... )
        """
        return self.cs.create_job(
            self.uid, workspace_uid, type, connections=connections, params=params, title=title, desc=desc
        )

    def create_external_job(
        self,
        workspace_uid: str,
        title: str = "",
        desc: str = "",
    ) -> ExternalJobController:
        """
        Add a new External job to this project to save generated outputs to.

        Args:
            workspace_uid (str): Workspace UID to create job in, e.g., "W3".
            title (str, optional): Title for external job (recommended).
                Defaults to "".
            desc (str, optional): Markdown description for external job.
                Defaults to "".

        Returns:
            ExternalJob: created external job instance
        """
        return self.cs.create_external_job(self.uid, workspace_uid=workspace_uid, title=title, desc=desc)

    def import_job(self, workspace_uid: str, path: Union[str, PurePosixPath]):
        """
        Import a job from a location on disk

        Args:
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            path (str | Path): Path to job directory, must be in the project
                directory. If the CryoSPARC instance is hosted remotely,
                this should be a path available on the server file system.
                e.g., ``"/projects/CS-project/imports/jobs/J134_homo_abinit"``
                or ``"imports/jobs/J134_homo_abinit"``

        Raises:
            APIError: Job cannot be imported.
        """
        return self.cs.import_job(self.uid, workspace_uid, path)

    def save_external_result(
        self,
        workspace_uid: Optional[str],
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[SlotSpec]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        title: str = "",
        desc: str = "",
        image: Optional[FileOrFigure] = None,
        savefig_kw: dict = dict(bbox_inches="tight", pad_inches=0),
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
            ...     workspace_uid="W1",
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
            slots (list[SlotSpec], optional): List of slots expected to
                be created for this output such as ``location`` or ``blob``. Do
                not specify any slots that were passed through from an input
                unless those slots are modified in the output. Defaults to None.
            passthrough (tuple[str, str], optional): Indicates that this output
                inherits slots from the specified output. e.g., ``("J1",
                "particles")``. Defaults to None.
            title (str, optional): Human-readable title for this output.
                Defaults to "".
            desc (str, optional): Markdown description for this output. Defaults
                to "".
            image (str | Path | IO | Figure, optional): Optional image file
                or matplotlib Figure to set as the image for this output.
                Defaults to None.
            savefig_kw (dict, optional): Additional keyword arguments to pass
                to ``figure.savefig()`` when saving matplotlib Figures. Defaults
                to ``dict(bbox_inches="tight", pad_inches=0)``.

        Returns:
            str: UID of created job where this output was saved
        """
        if slots and any(isinstance(s, dict) and "prefix" in s for s in slots):
            warnings.warn("'prefix' slot key is deprecated. Use 'name' instead.", DeprecationWarning, stacklevel=2)
            slots = [as_output_slot(slot) for slot in slots]  # type: ignore
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
            image=image,
            savefig_kw=savefig_kw,
        )

    def set_default_param(self, param: str, value: Any):
        """
        Set a default parameter value for this project. Default parameters are
        used to pre-populate parameter fields when creating new jobs in this
        project. This is useful for setting parameters that are commonly used
        across many jobs in the same project, such as file paths.

        Args:
            param (str): Name of the parameter to set a default value for,
                e.g., "compute_num_gpus".
            value (any): Default value to set for the given parameter.
        """
        self.cs.api.projects.set_default_param(self.uid, param, value)

    def clear_default_param(self, param: str):
        """
        Clear a default parameter value for this project.

        Args:
            param (str): Name of the parameter to clear the default value for,
                e.g., "compute_num_gpus".
        """
        self.cs.api.projects.clear_default_param(self.uid, param)

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

    def download(self, path: Union[str, PurePosixPath]):
        """
        Open a file in the current project for reading. Use to get files from a
        remote CryoSPARC instance where the project directory is not available
        on the client file system.

        Args:
            path (str | Path): Name or path of file in project directory.

        Yields:
            HTTPResponse: Use a context manager to read the file from the
            request body.

        Examples:

            Download a project's metadata

            >>> cs = CryoSPARC("http://localhost:61000")
            >>> project = cs.find_project("P3")
            >>> with project.download("project.json") as res:
            >>>     project_data = json.loads(res.read())

        """
        return self.cs.download(self.uid, path)

    def download_file(self, path: Union[str, PurePosixPath], target: Union[str, PurePath, IO[bytes]] = ""):
        """
        Download a file from the project directory to the given target path or
        writeable file handle.

        Args:
            path (str | Path): Name or path of file in project directory.
            target (str | Path | IO, optional): Local file path, directory path or
                writeable file handle to write response data. If not specified,
                downloads to current working directory with same file name.
                Defaults to "".

        Returns:
            Path | IO: resulting target path or file handle.
        """
        return self.cs.download_file(self.uid, path, target)

    def download_dataset(self, path: Union[str, PurePosixPath]):
        """
        Download a .cs dataset file from the given relative path in the project
        directory.

        Args:
            path (str | Path): Name or path to .cs file in project directory.

        Returns:
            Dataset: Loaded dataset instance
        """
        return self.cs.download_dataset(self.uid, path)

    def download_mrc(self, path: Union[str, PurePosixPath]):
        """
        Download a .mrc file from the given relative path in the project
        directory.

        Args:
            path (str | Path): Name or path to .mrc file in project directory.

        Returns:
            tuple[Header, NDArray]: MRC file header and data as a numpy array
        """
        return self.cs.download_mrc(self.uid, path)

    def upload(
        self,
        target_path: Union[str, PurePosixPath],
        source: Union[str, bytes, PurePath, IO],
        *,
        overwrite: bool = False,
    ):
        """
        Upload the given file to the project directory at the given relative
        path. Fails if target already exists.

        Args:
            target_path (str | Path): Name or path of file to write in project
                directory.
            source (str | bytes | Path | IO): Local path or file handle to
                upload. May also specified as raw bytes.
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        return self.cs.upload(self.uid, target_path, source, overwrite=overwrite)

    def upload_dataset(
        self,
        target_path: Union[str, PurePosixPath],
        dset: Dataset,
        *,
        format: int = DEFAULT_FORMAT,
        overwrite: bool = False,
    ):
        """
        Upload a dataset as a CS file into the project directory. Fails if
        target already exists.

        Args:
            target_path (str | Path): Name or path of dataset to save in the
                project directory. Should have a ``.cs`` extension.
            dset (Dataset): Dataset to save.
            format (int): Format to save in from ``cryosparc.dataset.*_FORMAT``,
                defaults to NUMPY_FORMAT)
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        return self.cs.upload_dataset(self.uid, target_path, dset, format=format, overwrite=overwrite)

    def upload_mrc(
        self,
        target_path: Union[str, PurePosixPath],
        data: "NDArray",
        psize: float,
        *,
        overwrite: bool = False,
    ):
        """
        Upload a numpy 2D or 3D array to the project directory as an MRC file. Fails
        if target already exists.

        Args:
            target_path (str | Path): Name or path of MRC file to save in the
                project directory. Should have a ``.mrc`` extension.
            data (NDArray): Numpy array with MRC file data.
            psize (float): Pixel size to include in MRC header.
            overwrite (bool, optional): If True, overwrite existing files.
                Defaults to False.
        """
        return self.cs.upload_mrc(self.uid, target_path, data, psize, overwrite=overwrite)

    def mkdir(
        self,
        target_path: Union[str, PurePosixPath],
        parents: bool = False,
        exist_ok: bool = False,
    ):
        """
        Create a directory in the given project.

        Args:
            target_path (str | Path): Name or path of folder to create inside
                the project directory.
            parents (bool, optional): If True, any missing parents are created
                as needed. Defaults to False.
            exist_ok (bool, optional): If True, does not raise an error for
                existing directories. Still raises if the target path is not a
                directory. Defaults to False.
        """
        self.cs.mkdir(
            project_uid=self.uid,
            target_path=target_path,
            parents=parents,
            exist_ok=exist_ok,
        )

    def cp(self, source_path: Union[str, PurePosixPath], target_path: Union[str, PurePosixPath] = ""):
        """
        Copy a file or folder into the job direcotry.

        Args:
            source_path (str | Path): Relative or absolute path of source file
                or folder to copy. If relative, assumed to be within the project
                directory.
            target_path (str | Path, optional): Name or path in the project
                directory to copy into. If not specified, uses the same file
                name as the source. Defaults to "".
        """
        self.cs.cp(
            project_uid=self.uid,
            source_path=source_path,
            target_path=target_path,
        )

    def symlink(self, source_path: Union[str, PurePosixPath], target_path: Union[str, PurePosixPath] = ""):
        """
        Create a symbolic link in the given project. May only create links for
        files within the project.

        Args:
            source_path (str | Path): Relative or absolute path of source file
                or folder to create a link to. If relative, assumed to be within
                the project directory.
            target_path (str | Path): Name or path of new symlink in the project
                directory. If not specified, creates link with the same file
                name as the source. Defaults to "".
        """
        self.cs.symlink(
            project_uid=self.uid,
            source_path=source_path,
            target_path=target_path,
        )
