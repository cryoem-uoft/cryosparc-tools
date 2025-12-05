# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .file_browser import FileBrowserPrefixes


class Bookmark(BaseModel):
    id: str
    path: str
    name: str
    description: str
    color: str
    last_accessed: datetime.datetime
    created_at: datetime.datetime


class Email(BaseModel):
    address: str
    verified: bool = False


class RecentPath(BaseModel):
    path: str
    last_accessed: datetime.datetime


class FileBrowserState(BaseModel):
    recentPaths: List[RecentPath] = []
    bookmarks: List[Bookmark] = []


class LoginToken(BaseModel):
    hashedToken: str
    when: datetime.datetime


class LoginService(BaseModel):
    loginTokens: List[LoginToken] = []


class PasswordService(BaseModel):
    bcrypt: str


class Profile(BaseModel):
    name: Union[str, Dict[str, Any]] = ""


class RecentJob(BaseModel):
    project_uid: str
    workspace_uid: str
    job_uid: str


class RecentSession(BaseModel):
    project_uid: str
    session_uid: str


class RecentWorkspace(BaseModel):
    project_uid: str
    workspace_uid: str


class Services(BaseModel):
    password: Optional[PasswordService] = None
    resume: LoginService = LoginService()


class UserState(BaseModel):
    licenseAccepted: bool = False
    userCanSetJobPriority: bool = False
    recentProjects: List[str] = []
    recentWorkspaces: List[RecentWorkspace] = []
    recentSessions: List[RecentSession] = []
    recentJobs: List[RecentJob] = []
    browserPath: Optional[str] = None
    defaultJobPriority: Optional[int] = None
    userFileBrowserState: FileBrowserState = FileBrowserState()


class User(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
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
    created_by_user_id: Optional[str] = None
    name: str
    first_name: str
    last_name: str
    status: Literal["invited"] = "invited"
    profile: Profile = Profile()
    roles: Dict[str, List[Literal["user", "admin"]]] = {}
    register_token: Optional[str] = None
    reset_token: Optional[str] = None
    last_password_changed_at: Optional[datetime.datetime] = None
    services: Services = Services()
    state: UserState = UserState()
    preferences: Dict[str, Any] = {}
    file_browser_settings: FileBrowserPrefixes = FileBrowserPrefixes()
    lanes: List[str] = []
    """
    List of lane names that a user is allowed to access
    """
