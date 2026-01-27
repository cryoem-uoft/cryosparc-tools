# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field

from .instance import InstanceInformation
from .job_spec import JobBuildError, JobSpec
from .resource import FixedResourceSlots, ResourceSlots
from .scheduler_target import SchedulerTarget


class AllocatedResources(BaseModel):
    """
    Scheduler target resources allocated for a job the scheduler at launch time.
    """

    lane: Optional[str] = None
    """
    Name of lane the job is queued to.
    """
    lane_type: Optional[str] = None
    """
    Type of allocated lane ("node" or "cluster")
    """
    hostname: str
    """
    For node targets, hostname of selected worker node.
    """
    target: Optional[SchedulerTarget] = None
    """
    Allocated scheduler target full details.
    """
    slots: ResourceSlots = ResourceSlots()
    """
    Listings of compute resources allocated for this job.
    """
    fixed: FixedResourceSlots = FixedResourceSlots()
    """
    Fixed compute resources allocated for this job.
    """
    licenses_acquired: int = 0
    """
    Number of GPU tokens allocated for this job.
    """


JobStatus = Literal["building", "queued", "launched", "started", "running", "waiting", "completed", "killed", "failed"]


class JobLastAccessed(BaseModel):
    """
    Record for when a user last accessed a job.
    """

    name: str
    """
    Username that last accessed the job.
    """
    accessed_at: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    Timestamp of the last access.
    """


class RunError(BaseModel):
    """
    Details about a runtime error encountered during job execution.
    """

    message: str
    """
    """
    warning: bool = False
    """
    """


class UiTileImage(BaseModel):
    """
    Details about a job tile image shown in the UI.
    """

    name: str
    """
    """
    fileid: str
    """
    """
    num_rows: Optional[int] = None
    """
    """
    num_cols: Optional[int] = None
    """
    """


class JobWorkflowInfo(BaseModel):
    """
    Information about the workflow a job is part of.
    """

    id: str
    """
    """
    jobId: str
    """
    """
    run: int = 0
    """
    """


class Job(BaseModel):
    """
    Specification for a CryoSPARC Job object. Access parameters, inputs and
    outputs from `spec` field.
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
    Job unique ID, e.g., "J42".
    """
    project_uid: str
    """
    Project unique ID, e.g., "P3".
    """
    workspace_uids: List[str] = ["W1"]
    """
    List of workspace UIDs this job belongs to. Must have at least one item
    """
    spec: JobSpec
    """
    Job type-specific settings, including params, inputs, outputs, resources, etc.
    """
    job_dir: str
    """
    Job directory relative to project directory, usually the same as uid
    """
    job_dir_size: int = 0
    """
    Job directory size
    """
    job_dir_size_last_updated: datetime.datetime
    """
    Job directory last updated timestamp
    """
    run_as_user: Optional[str] = None
    """
    Developer job register to use (internal use only)
    """
    title: str = ""
    """
    Human-readable job title.
    """
    description: str = ""
    """
    Human-readable job markdown description.
    """
    status: JobStatus = "building"
    """
    Job scheduling status, e.g., "building", "queued", "running".
    """
    created_by_user_id: Optional[str] = None
    """
    Object ID of user account that created this job.
    """
    created_by_job_uid: Optional[str] = None
    """
    Job ID of job that created this job.
    """
    cloned_from: Optional[str] = None
    """
    Job this job was cloned from
    """
    queued_at: Optional[datetime.datetime] = None
    """
    When added to the queue
    """
    started_at: Optional[datetime.datetime] = None
    """
    When internal scheduler launches/submits
    """
    launched_at: Optional[datetime.datetime] = None
    """
    When job initial monitor process starts
    """
    running_at: Optional[datetime.datetime] = None
    """
    When job monitor process forks into main process
    """
    waiting_at: Optional[datetime.datetime] = None
    """
    When job last went into waiting for interaction status
    """
    completed_at: Optional[datetime.datetime] = None
    """
    When job finished
    """
    killed_at: Optional[datetime.datetime] = None
    """
    When job was killed
    """
    failed_at: Optional[datetime.datetime] = None
    """
    When job failed
    """
    heartbeat_at: Optional[datetime.datetime] = None
    """
    Last heartbeat received
    """
    tokens_acquired_at: Optional[datetime.datetime] = None
    """
    When job got confirmation from the licensing server to run
    """
    tokens_requested_at: Optional[datetime.datetime] = None
    """
    When job started waiting for tokens to become available
    """
    last_scheduled_at: Optional[datetime.datetime] = None
    """
    When scheduler last tried to launch this job
    """
    last_accessed: Optional[JobLastAccessed] = None
    """
    When a specific user last opened this job in the user interface
    """
    has_error: bool = False
    """
    If True, there's an error in the streamlog
    """
    has_warning: bool = False
    """
    If True, there's a warning in the streamlog
    """
    version_created: Optional[str] = None
    """
    Version of CryoSPARC used to create this job
    """
    version: Optional[str] = None
    """
    Version of CryoSPARC that last ran this job
    """
    priority: int = 0
    """
    Queue priority set by user or default for system. higher number = higher priority.
    """
    deleted: bool = False
    """
    True if the job has been marked as deleted.
    """
    deleting: bool = False
    """
    True if the job is being deleted
    """
    parents: List[str] = []
    """
    Set of parent jobs UIDs based on input connections.
    """
    children: List[str] = []
    """
    Set of child job UIDs based on output connections.
    """
    resources_allocated: Optional[AllocatedResources] = None
    """
    Set by scheduler before launching
    """
    queued_by_user_id: Optional[str] = None
    """
    ID of user account that queued this job.
    """
    queued_to_lane: Optional[str] = None
    """
    set at queue time based on params
    """
    queued_to_hostname: Optional[str] = None
    """
    NOTE: database field is sometimes ``False`` in older CryoSPARC jobs, cast validator prevents this
    """
    queued_to_gpu: Optional[List[int]] = None
    """
    NOTE: database field is sometimes ``False`` in older CryoSPARC jobs, cast validator prevents this
    """
    queue_status: Optional[
        Literal[
            "waiting_inputs",
            "project_paused",
            "actively_queued",
            "launched",
            "waiting_maintenance",
            "waiting_licenses",
            "waiting_resources",
        ]
    ] = None
    """
    waiting_inputs, project_paused, actively_queued, launched
    """
    queue_message: Optional[str] = None
    """
    e.g., "Waiting for resource in lane xxx: GPU"
    """
    queued_job_hash: Optional[int] = None
    """
    hash of puid, juid, # tokens requested, job type and user for scheduler
    """
    num_tokens: int = 0
    """
    number of (GPU) tokens required to run the job
    """
    job_sig: Optional[str] = None
    """
    Job signature created by licensing module
    """
    errors_run: List[RunError] = []
    """
    Runtime errors, limited to last 10 errors.
    """
    interactive_port: Optional[int] = None
    """
    Port assigned to a job's interactive server by operation system, only
    available after an interactive job enters waiting status.
    """
    PID_monitor: Optional[int] = None
    """
    Process ID number of job monitor process
    """
    PID_main: Optional[int] = None
    """
    Process ID number of main job process
    """
    PID_workers: List[int] = []
    """
    Process ID numbers of worker processes spawned by main job process
    """
    cluster_job_id: Optional[str] = None
    """
    """
    cluster_job_status: Optional[str] = None
    """
    """
    cluster_job_status_code: Optional[str] = None
    """
    """
    cluster_job_monitor_event_id: Optional[str] = None
    """
    """
    cluster_job_monitor_retries: int = 0
    """
    """
    cluster_job_monitor_last_run_at: Optional[datetime.datetime] = None
    """
    """
    cluster_job_submission_script: Optional[str] = None
    """
    """
    cluster_job_custom_vars: Dict[str, str] = {}
    """
    """
    ui_tile_images: List[UiTileImage] = []
    """
    Job tile images in app
    """
    is_experiment: bool = False
    """
    Is the job part of an experiment?
    """
    enable_bench: bool = False
    """
    Enable benchmarking for this job. Used by extensive workflow
    """
    bench: Dict[str, Union[float, int, Dict[str, Any]]] = {}
    """
    Benchmarking data
    """
    bench_timings: Dict[str, List[Tuple[datetime.datetime, Optional[datetime.datetime]]]] = {}
    """
    Benchmark timing data in the form {key: list_of_timings}
    """
    completed_count: int = 0
    """
    How many times this job was marked as completed
    """
    instance_information: InstanceInformation = InstanceInformation()
    """
    Details about the active instance
    """
    last_intermediate_data_cleared_at: Optional[datetime.datetime] = None
    """
    """
    last_intermediate_data_cleared_amount: int = 0
    """
    """
    intermediate_results_size_bytes: int = 0
    """
    """
    intermediate_results_size_last_updated: Optional[datetime.datetime] = None
    """
    """
    is_final_result: bool = False
    """
    Job and its anscestors is the final result and should not be deleted
    during data cleanup steps.
    """
    is_ancestor_of_final_result: bool = False
    """
    Similar to is_final_result
    """
    no_check_inputs_ready: bool = False
    """
    If True, a queued job will be launched even if parent connected jobs have
    not finished running yet.
    """
    progress: List[Dict[str, Any]] = []
    """
    Progress log
    """
    last_exported_at: Optional[datetime.datetime] = None
    """
    Time of last export of outputs
    """
    last_exported_location: Optional[str] = None
    """
    Time of last exported outputs location
    """
    last_exported_version: Optional[str] = None
    """
    Version of CryoSPARC this job was last exported from
    """
    tags: List[str] = []
    """
    Tags associated with this job
    """
    workflow: Optional[JobWorkflowInfo] = None
    """
    Information about workflow that job ran from, if any
    """
    imported_at: Optional[datetime.datetime] = None
    """
    """
    deleted_at: Optional[datetime.datetime] = None
    """
    """
    import_status: Optional[Literal["importing", "complete", "failed"]] = None
    """
    """
    attach_status: Optional[Literal["attaching", "complete", "failed"]] = None
    """
    """
    starred_by: List[str] = []
    """
    List of user IDs who have starred this job.
    """
    requeue_windows_ends_at: Optional[datetime.datetime] = None
    """
    If set, job can be cleared and requeued until this time without losing its place in the queue
    """
    uid_num: int
    """
    Numeric part of the job UID.
    """
    project_uid_num: int
    """
    Numeric part of the project UID.
    """
    status_num: int
    """
    Numeric representation of job status.
    """
    build_errors: List[JobBuildError]
    """
    List of build errors for this job. Jobs cannot be queued or launched if there are build errors.
    """
