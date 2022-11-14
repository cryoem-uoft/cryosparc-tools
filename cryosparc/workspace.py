from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from .dataset import Dataset
from .row import R
from .job import ExternalJob
from .spec import DatabaseEntity, Datafield, Datatype, WorkspaceDocument

if TYPE_CHECKING:
    from .tools import CryoSPARC


class Workspace(DatabaseEntity[WorkspaceDocument]):
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
