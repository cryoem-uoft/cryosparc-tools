# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GenerateIntermediateResultsSettings(BaseModel):
    class_2D: bool = False
    class_3D: bool = False
    var_3D_disp: bool = False


class ProjectLastAccessed(BaseModel):
    name: str = ""
    accessed_at: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


class ProjectStats(BaseModel):
    workspace_count: int = 0
    session_count: int = 0
    job_count: int = 0
    job_types: Dict[str, int] = {}
    job_sections: Dict[str, int] = {}
    job_status: Dict[str, int] = {}
    updated_at: Optional[datetime.datetime] = None


class Project(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    dumped_at: Optional[datetime.datetime] = None
    last_dumped_version: Optional[str] = None
    autodump: bool = True
    uid: str
    project_dir: str
    owner_user_id: str
    title: str
    description: str = ""
    project_params_pdef: dict = {}
    queue_paused: bool = False
    deleted: bool = False
    users_with_access: List[str] = []
    size: int = 0
    size_last_updated: datetime.datetime
    last_accessed: Optional[ProjectLastAccessed] = None
    archived: bool = False
    detached: bool = False
    hidden: bool = False
    generate_intermediate_results_settings: GenerateIntermediateResultsSettings = GenerateIntermediateResultsSettings()
    last_exp_group_id_used: Optional[int] = None
    develop_run_as_user: Optional[str] = None
    imported_at: Optional[datetime.datetime] = None
    import_status: Optional[Literal["importing", "complete", "failed"]] = None
    project_stats: ProjectStats = ProjectStats()
    last_archived_version: Optional[str] = None
    last_detached_version: Optional[str] = None
    is_cleanup_in_progress: bool = False
    tags: List[str] = []
    starred_by: List[str] = []
    uid_num: int


class ProjectSymlink(BaseModel):
    path: str
    target: str
    exists: bool
