import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from .controller import Controller, as_output_slot
from .dataset import Dataset
from .job import ExternalJobController, JobController
from .models.workspace import Workspace
from .row import R
from .spec import Datatype, SlotSpec

if TYPE_CHECKING:
    from .tools import CryoSPARC


class WorkspaceController(Controller[Workspace]):
    """
    Accessor class to a workspace in CryoSPARC with ability create jobs and save
    results. Should be created with`
    :py:meth:`cs.find_workspace() <cryosparc.tools.CryoSPARC.find_workspace>` or
    :py:meth:`project.find_workspace() <cryosparc.project.ProjectController.find_workspace>`.

    Arguments:
        workspace (tuple[str, str] | Workspace): either _(Project UID, Workspace UID)_
            tuple or Workspace model, e.g. ``("P3", "W4")``

    Attributes:
        model (Workspace): All workspace data from the CryoSPARC database.
            Contents may change over time, use :py:method:`refresh` to update.
    """

    uid: str
    """
    Workspace unique ID, e.g., "W42"
    """
    project_uid: str
    """
    Project unique ID, e.g., "P3"
    """

    def __init__(self, cs: "CryoSPARC", workspace: Union[Tuple[str, str], Workspace]) -> None:
        self.cs = cs
        if isinstance(workspace, tuple):
            self.project_uid, self.uid = workspace
            self.refresh()
        else:
            self.project_uid = workspace.project_uid
            self.uid = workspace.uid
            self.model = workspace

    def refresh(self):
        """
        Reload this workspace from the CryoSPARC database.

        Returns:
            WorkspaceController: self
        """
        self.model = self.cs.api.workspaces.find_one(self.project_uid, self.uid)
        return self

    def create_job(
        self,
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
            JobController: created job instance. Raises error if job cannot be created.

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
        """
        return self.cs.create_job(
            self.project_uid, self.uid, type, connections=connections, params=params, title=title, desc=desc
        )

    def create_external_job(
        self,
        title: str = "",
        desc: str = "",
    ) -> ExternalJobController:
        """
        Add a new External job to this workspace to save generated outputs to.

        Args:
            workspace_uid (str): Workspace UID to create job in, e.g., "W1"
            title (str, optional): Title for external job (recommended).
                Defaults to "".
            desc (str, optional): Markdown description for external job.
                Defaults to "".

        Returns:
            ExternalJobController: created external job instance
        """
        return self.cs.create_external_job(self.project_uid, self.uid, title, desc)

    def save_external_result(
        self,
        dataset: Dataset[R],
        type: Datatype,
        name: Optional[str] = None,
        slots: Optional[List[SlotSpec]] = None,
        passthrough: Optional[Tuple[str, str]] = None,
        title: str = "",
        desc: str = "",
    ) -> str:
        """
        Save the given result dataset to a workspace.

        Args:
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
        if slots and any(isinstance(s, dict) and "prefix" in s for s in slots):
            warnings.warn("'prefix' slot key is deprecated. Use 'name' instead.", DeprecationWarning, stacklevel=2)
            # convert to prevent from warning again
            slots = [as_output_slot(slot) for slot in slots]  # type: ignore
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
