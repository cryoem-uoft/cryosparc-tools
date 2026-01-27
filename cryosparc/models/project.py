# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class GenerateIntermediateResultsSettings(BaseModel):
    """
    Settings for generating intermediate results for different job types in a project.
    """

    class_2D_new: bool = False
    """
    """
    class_3D: bool = False
    """
    """
    var_3D_disp: bool = False
    """
    """


class ProjectLastAccessed(BaseModel):
    """
    Record for when a user last accessed a project.
    """

    name: str = ""
    """
    Username that last accessed the project.
    """
    accessed_at: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    Timestamp of the last access.
    """


class ProjectStats(BaseModel):
    """
    Statistics about the contents of a project.
    """

    workspace_count: int = 0
    """
    """
    session_count: int = 0
    """
    """
    job_count: int = 0
    """
    """
    job_types: Dict[str, int] = {}
    """
    Number of jobs per type.
    """
    job_sections: Dict[str, int] = {}
    """
    Number of jobs per section.
    """
    job_status: Dict[str, int] = {}
    """
    Number of jobs per status.
    """
    updated_at: Optional[datetime.datetime] = None
    """
    """


class ProjectWorkflowInfo(BaseModel):
    """
    Information about workflows used in a project.
    """

    latest_workflow_uid: str = "WF1"
    """
    UID of the latest workflow applied in a project.
    """
    runs: Dict[str, int] = {}
    """
    Map from workflow ID to the number of runs of that workflow.
    """


class Project(BaseModel):
    """
    Representation of a CryoSPARC Project, a container of data-processing
    results.
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
    dumped_at: Optional[datetime.datetime] = None
    """
    When the model was last dumped to disk
    """
    last_dumped_version: Optional[str] = None
    """
    The version of CryoSPARC last dumped at
    """
    autodump: bool = True
    """
    Whether the model was updated recently and must be dumped

    :meta private:
    """
    uid: str
    """
    Unique identifier for the project, e.g. "P1".
    """
    project_dir: str
    """
    Absolute path to the project directory on disk.
    """
    owner_user_id: str
    """
    User ID of the project owner.
    """
    title: str
    """
    User-specified project title.
    """
    description: str = ""
    """
    User-specified detailed project description.
    """
    project_params_pdef: Dict[str, Any] = {}
    """
    Default job parameter definitions to be used within the project.
    For example ``{'output_f16': True}`` to enable half-precision output for
    all motion correction jobs.
    """
    queue_paused: bool = False
    """
    Currently unused :meta private:
    """
    deleted: bool = False
    """
    """
    deleting: bool = False
    """
    """
    users_with_access: List[str] = []
    """
    List of user IDs with access to the project, including the owner.
    """
    size: int = 0
    """
    Size of the project on disk, in bytes.
    """
    size_last_updated: datetime.datetime
    """
    """
    last_accessed: Optional[ProjectLastAccessed] = None
    """
    """
    archived: bool = False
    """
    """
    detached: bool = False
    """
    """
    generate_intermediate_results_settings: GenerateIntermediateResultsSettings = GenerateIntermediateResultsSettings()
    """
    """
    last_exp_group_id_used: Optional[int] = None
    """
    Last exposure group ID used in the project.
    """
    develop_run_as_user: Optional[str] = None
    """
    :meta private:
    """
    imported_at: Optional[datetime.datetime] = None
    """
    """
    import_status: Optional[Literal["importing", "complete", "failed"]] = None
    """
    """
    project_stats: ProjectStats = ProjectStats()
    """
    """
    created_at_version: str = "unknown"
    """
    """
    last_archived_version: Optional[str] = None
    """
    """
    last_detached_version: Optional[str] = None
    """
    """
    is_cleanup_in_progress: bool = False
    """
    """
    tags: List[str] = []
    """
    """
    starred_by: List[str] = []
    """
    List of user IDs who have starred the project.
    """
    autodump_failed: bool = False
    """
    Whether the last automatic disk dump operation failed.
    """
    autodump_errors: List[str] = []
    """
    List of error messages from the last automatic disk dump operation.
    """
    workflows: Optional[ProjectWorkflowInfo] = None
    """
    """
    uid_num: int
    """
    Numeric part of the project UID.
    """


class ProjectSymlink(BaseModel):
    """
    Information about a symlink in a project directory.
    """

    path: str
    """
    Path of the symlink.
    """
    target: str
    """
    Target of the symlink.
    """
    exists: bool
    """
    Whether the target of the symlink exists.
    """
