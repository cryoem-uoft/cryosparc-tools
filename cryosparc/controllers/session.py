from typing import TYPE_CHECKING, Optional, Tuple, Union

from ..models.session import Session
from . import Controller

if TYPE_CHECKING:
    from ..tools import CryoSPARC


class SessionController(Controller[Session]):
    """
    Accessor class to a CryoSPARC Live session with the ability to manage
    session lifecycle. Should be initialized with
    :py:meth:`cs.find_session() <cryosparc.tools.CryoSPARC.find_session>`
    or :py:meth:`project.find_session() <cryosparc.controllers.project.ProjectController.find_session>`
    """

    uid: str
    """
    Session unique ID, e.g., "S3"
    """
    workspace_uid: str
    """
    Unique ID of workspace associated with the session, e.g., "W42"
    """
    project_uid: str
    """
    Project unique ID, e.g., "P3"
    """

    def __init__(self, cs: "CryoSPARC", session: Union[Tuple[str, str], Session]) -> None:
        self.cs = cs
        if isinstance(session, tuple):
            self.project_uid, self.uid = session
            self.refresh()
            self.workspace_uid = self.model.uid
        else:
            self.uid = session.session_uid
            self.workspace_uid = session.uid
            self.project_uid = session.project_uid
            self.model = session

    @property
    def title(self) -> Optional[str]:
        """Workspace title"""
        return self.model.title

    @property
    def desc(self) -> Optional[str]:
        """Workspace description"""
        return self.model.description

    def refresh(self):
        """
        Reload this session from the CryoSPARC database.

        Returns:
            SessionController: self
        """
        self.model = self.cs.api.sessions.find_one(self.project_uid, self.uid)
        return self

    def set_title(self, title: str):
        """
        Set the workspace title.

        Args:
            title (str): New workspace title
        """
        self.model = self.cs.api.workspaces.set_title(self.project_uid, self.workspace_uid, title=title)

    def set_description(self, desc: str):
        """
        Set the workspace description. May include Markdown formatting.

        Args:
            desc (str): New workspace description
        """
        self.model = self.cs.api.workspaces.set_description(self.project_uid, self.workspace_uid, description=desc)
