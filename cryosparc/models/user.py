# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .file_browser import FileBrowserPrefixes


class Bookmark(BaseModel):
    """
    Bookmarked file path
    """

    id: str
    """
    """
    path: str
    """
    """
    name: str
    """
    """
    description: str
    """
    """
    color: str
    """
    """
    last_accessed: datetime.datetime
    """
    """
    created_at: datetime.datetime
    """
    """


class Email(BaseModel):
    """
    Email address information associated with a user account
    """

    address: str
    """
    """
    verified: bool = False
    """
    """


class RecentPath(BaseModel):
    """
    Recently accessed file path
    """

    path: str
    """
    """
    last_accessed: datetime.datetime
    """
    """


class FileBrowserState(BaseModel):
    """
    State of the user's file browser
    """

    recentPaths: List[RecentPath] = []
    """
    """
    bookmarks: List[Bookmark] = []
    """
    """


class LoginToken(BaseModel):
    """
    Login token for resuming sessions
    """

    hashedToken: str
    """
    """
    when: datetime.datetime
    """
    """


class LoginService(BaseModel):
    """
    Service information for login/resume functionality
    """

    loginTokens: List[LoginToken] = []
    """
    """


class PasswordService(BaseModel):
    """
    Service information for password-based authentication
    """

    bcrypt: str
    """
    The bcrypt-hashed password value
    """


class Profile(BaseModel):
    """
    Profile information for a user account
    """

    name: Union[str, Dict[str, Any]] = ""
    """
    """


class RecentJob(BaseModel):
    """
    Recently accessed job
    """

    project_uid: str
    """
    """
    workspace_uid: str
    """
    """
    job_uid: str
    """
    """


class RecentSession(BaseModel):
    """
    Recently accessed session
    """

    project_uid: str
    """
    """
    session_uid: str
    """
    """


class RecentWorkspace(BaseModel):
    """
    Recently accessed workspace
    """

    project_uid: str
    """
    """
    workspace_uid: str
    """
    """


class Services(BaseModel):
    """
    Authentication services associated with a user account
    """

    password: Optional[PasswordService] = None
    """
    Service information for password-based authentication
    """
    resume: LoginService = LoginService()
    """
    Service information for resuming sessions
    """


class UserState(BaseModel):
    """
    State information for a user account
    """

    licenseAccepted: bool = False
    """
    """
    userCanSetJobPriority: bool = False
    """
    """
    recentProjects: List[str] = []
    """
    """
    recentWorkspaces: List[RecentWorkspace] = []
    """
    """
    recentSessions: List[RecentSession] = []
    """
    """
    recentJobs: List[RecentJob] = []
    """
    """
    browserPath: Optional[str] = None
    """
    """
    defaultJobPriority: Optional[int] = None
    """
    """
    userFileBrowserState: FileBrowserState = FileBrowserState()
    """
    """


class User(BaseModel):
    """
    User account information
    """

    id: str = Field("000000000000000000000000", alias="_id")
    """
    """
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    emails: List[Email] = []
    """
    """
    created_by_user_id: Optional[str] = None
    """
    ID of the admin user who created this account, if applicable
    """
    name: str
    """
    Username
    """
    first_name: str
    """
    First/given name
    """
    last_name: str
    """
    Last/family name
    """
    status: Literal["invited"] = "invited"
    """
    Registration status
    """
    profile: Profile = Profile()
    """
    """
    roles: Dict[str, List[Literal["user", "admin"]]] = {}
    """
    """
    register_token: Optional[str] = None
    """
    """
    reset_token: Optional[str] = None
    """
    """
    last_password_changed_at: Optional[datetime.datetime] = None
    """
    """
    services: Services = Services()
    """
    """
    state: UserState = UserState()
    """
    """
    preferences: Dict[str, Any] = {}
    """
    """
    file_browser_settings: FileBrowserPrefixes = FileBrowserPrefixes()
    """
    """
    lanes: List[str] = []
    """
    List of scheduler lane names that a user is allowed to queue to. Admins
    may queue to all lanes, regardless of this value.
    """
