# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class Email(BaseModel):
    address: str
    verified: bool = False


class LoginToken(BaseModel):
    hashedToken: str
    when: datetime.datetime


class LoginService(BaseModel):
    loginTokens: List[LoginToken] = []


class PasswordService(BaseModel):
    bcrypt: str


class Profile(BaseModel):
    name: Union[str, dict] = ""


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
    userCanModifyLiveDataManagement: bool = False
    recentProjects: List[str] = []
    recentWorkspaces: List[RecentWorkspace] = []
    recentSessions: List[RecentSession] = []
    recentJobs: List[RecentJob] = []
    browserPath: Optional[str] = None
    defaultJobPriority: int = 0


class User(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    emails: List[Email]
    created_by_user_id: Optional[str] = None
    name: str
    first_name: str
    last_name: str
    status: str = "invited"
    profile: Profile = Profile()
    roles: Dict[str, List[Literal["user", "admin"]]] = {}
    register_token: Optional[str] = None
    reset_token: Optional[str] = None
    services: Services = Services()
    state: UserState = UserState()
    preferences: dict = {}
    allowed_prefix_dir: str = "/"
    lanes: List[str] = []
