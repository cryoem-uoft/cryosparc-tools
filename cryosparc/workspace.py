from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .dataset import Dataset
from .row import R
from .job import Job, ExternalJob
from .spec import MongoController, Datafield, Datatype, WorkspaceDocument

if TYPE_CHECKING:
    from .tools import CryoSPARC


class Workspace(MongoController[WorkspaceDocument]):
    """
    Accessor class to a workspace in CryoSPARC with ability create jobs and save
    results. Should be instantiated through `CryoSPARC.find_workspace`_ or
    `Project.find_workspace`_.

    .. _CryoSPARC.find_workspace:
        tools.html#cryosparc.tools.CryoSPARC.find_workspace

    .. _Project.find_workspace:
        project.html#cryosparc.project.Project.find_workspace
    """

    def __init__(self, cs: "CryoSPARC", project_uid: str, uid: str) -> None:
        self.cs = cs
        self.project_uid = project_uid
        self.uid = uid

    def refresh(self):
        """
        Reload this workspace from the CryoSPARC database.

        Returns:
            Workspace: self
        """
        self._doc = self.cs.cli.get_workspace(self.project_uid, self.uid)  # type: ignore
        return self

    def create_job(
        self,
        type: str,
        connections: Dict[str, Tuple[str, str]] = {},
        params: Dict[str, Any] = {},
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> Job:
        """
        Create a new job with the given type. Use the
        `CryoSPARC.get_job_sections`_ method to query available job types on
        the connected CryoSPARC instance.

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
            >>> workspace = cs.find_workspace("P3", "W3")
            >>> import_job = workspace.create_job("W1", "import_movies")
            >>> import_job.set_param("blob_paths", "/bulk/data/t20s/*.tif")
            True

            Create a 3-class ab-initio job connected to existing particles.

            >>> abinit_job = workspace.create_job("homo_abinit"
            ...     connections={"particles": ("J20", "particles_selected")}
            ...     params={"abinit_K": 3}
            ... )

        .. _CryoSPARC.get_job_sections:
            tools.html#cryosparc.tools.CryoSPARC.get_job_sections
        """
        return self.cs.create_job(
            self.project_uid, self.uid, type, connections=connections, params=params, title=title, desc=desc
        )

    def create_external_job(
        self,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> ExternalJob:
        """
        Add a new External job to this workspace to save generated outputs to.

        Args:
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            title (str, optional): Title for external job (recommended).
                Defaults to None.
            desc (str, optional): Markdown description for external job.
                Defaults to None.

        Returns:
            ExternalJob: created external job instance
        """
        return self.cs.create_external_job(self.project_uid, self.uid, title, desc)

    def save_external_result(
        self,
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[Union[str, Datafield]]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        title: Optional[str] = None,
        desc: Optional[str] = None,
    ) -> str:
        """
        Save the given result dataset to a workspace.

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
                inherits slots from the specified output. e.g., ``("J1",
                "particles")``. Defaults to None.
            title (str, optional): Human-readable title for this output.
                Defaults to None.
            desc (str, optional): Markdown description for this output. Defaults
                to None.

        Returns:
            str: UID of created job where this output was saved.

        Examples:

            Save all particle data

            >>> particles = Dataset()
            >>> workspace.save_external_result(particles, 'particle')
            "J43"

            Save new particle locations that inherit passthrough slots from a
            parent job

            >>> particles = Dataset()
            >>> workspace.save_external_result(
            ...     dataset=particles,
            ...     type='particle',
            ...     name='particles',
            ...     slots=['location'],
            ...     passthrough=('J42', 'selected_particles'),
            ...     title='Re-centered particles'
            ... )
            "J44"

            Save a result with multiple slots of the same type.

            >>> workspace.save_external_result(
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
        """
        return self.cs.save_external_result(
            self.project_uid,
            self.uid,
            dataset=dataset,
            type=type,
            name=name,
            slots=slots,
            passthrough=passthrough,
            title=title,
            desc=desc,
        )
