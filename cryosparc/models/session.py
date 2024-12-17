# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from .job import JobStatus, RunError
from .session_params import LiveAbinitParams, LiveClass2DParams, LivePreprocessingParams, LiveRefineParams
from .session_spec import SessionStatus
from .signature import ImportSignature
from .workspace import WorkspaceStats


class AbInitioVolumeInfo(BaseModel):
    vol_gname: str
    fileid: Optional[str] = None
    selected: bool = False


class AthenaVolumeUploadParams(BaseModel):
    type: str = "refinement"
    name: str
    path_rel: str
    symmetry: str = "C1"
    psize_A: float = 1.0
    res_A: Optional[float] = None


class DataManagementStat(BaseModel):
    status: Literal["active", "archiving", "archived", "deleted", "deleting", "missing", "calculating"] = "active"
    prev_status: Optional[
        Literal["active", "archiving", "archived", "deleted", "deleting", "missing", "calculating"]
    ] = None
    size: int = 0


class DataManagementStats(BaseModel):
    raw: DataManagementStat = DataManagementStat()
    micrographs: DataManagementStat = DataManagementStat()
    thumbnails: DataManagementStat = DataManagementStat()
    particles: DataManagementStat = DataManagementStat()
    metadata: DataManagementStat = DataManagementStat()


class ECLSessionProperties(BaseModel):
    do_athena_volume_upload: bool = False
    athena_volume_upload_params: Optional[AthenaVolumeUploadParams] = None


class ExposureGroup(BaseModel):
    """
    Full exposure group defintion, not all properties are externally editable
    """

    ignore_exposures: bool = False
    gainref_path: Optional[str] = None
    defect_path: Optional[str] = None
    file_engine_recursive: bool = False
    file_engine_watch_path_abs: str = "/"
    file_engine_filter: str = "*"
    file_engine_interval: int = 10
    file_engine_min_file_size: int = 0
    file_engine_min_modified_time_delta: int = 0
    exp_group_id: int = 1
    num_exposures_found: int = 0
    num_exposures_ready: int = 0
    file_engine_strategy: Literal["entity", "timestamp", "eclathena"] = "entity"
    file_engine_enable: bool = False
    final: bool = False
    is_any_eer: bool = False


class ExposureGroupUpdate(BaseModel):
    """
    Public editable properties for exposure group
    """

    ignore_exposures: bool = False
    gainref_path: Optional[str] = None
    defect_path: Optional[str] = None
    file_engine_recursive: bool = False
    file_engine_watch_path_abs: str = "/"
    file_engine_filter: str = "*"
    file_engine_interval: int = 10
    file_engine_min_file_size: int = 0
    file_engine_min_modified_time_delta: int = 0


class LiveComputeResources(BaseModel):
    phase_one_lane: Optional[str] = None
    phase_one_gpus: int = 1
    phase_two_lane: Optional[str] = None
    phase_two_gpus: int = 1
    phase_two_ssd: bool = True
    auxiliary_lane: Optional[str] = None
    auxiliary_gpus: int = 1
    auxiliary_ssd: bool = True
    priority: int = 0


class Phase2ParticleOutputInfo(BaseModel):
    path: Optional[str] = None
    count: int = 0
    fields: List[str] = []


class Threshold(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    value: Optional[float] = None


class RangeThreshold(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None


class PickingThresholds(BaseModel):
    manual_ncc_score: Threshold = Threshold()
    manual_power: RangeThreshold = RangeThreshold()
    blob_ncc_score: Threshold = Threshold()
    blob_power: RangeThreshold = RangeThreshold()
    template_ncc_score: Threshold = Threshold()
    template_power: RangeThreshold = RangeThreshold()
    deep_ncc_score: Threshold = Threshold()
    deep_power: RangeThreshold = RangeThreshold()


class RTPChild(BaseModel):
    uid: str
    status: JobStatus
    rtp_handle_func: Literal[
        "handle_template_creation_class2D", "phase2_class2D_handle", "phase2_abinit_handle", "phase2_refine_handle"
    ]


class RtpWorkerState(BaseModel):
    status: JobStatus
    errors: List[RunError] = []


class SessionLastAccessed(BaseModel):
    name: str = ""
    accessed_at: datetime.datetime = datetime.datetime(1, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)


class SessionAttribute(BaseModel):
    name: str
    title: str
    min: Optional[float] = None
    max: Optional[float] = None
    round: int = 0


class TemplateClassInfo(BaseModel):
    class_idx: int
    fileid: str
    res_A: float
    selected: bool = False
    num_particles_selected: int = 0
    num_particles_total: int = 0
    mean_prob: float = 0.0
    class_ess: float = 0.0


class SessionStats(BaseModel):
    total_exposures: int = 0
    total_queued: int = 0
    total_seen: int = 0
    total_in_progress: int = 0
    total_thumbs: int = 0
    total_test: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    total_failed: int = 0
    total_ready: int = 0
    average_manual_picks: int = 0
    total_blob_picks: int = 0
    total_template_picks: int = 0
    total_deep_picks: int = 0
    total_manual_picks: int = 0
    total_extracted_particles_blob: int = 0
    total_extracted_particles_template: int = 0
    total_extracted_particles_manual: int = 0
    total_extracted_particles_deep: int = 0
    total_extracted_particles: int = 0
    total_manual_picked_exposures: int = 0
    gsfsc: float = 0.0
    frames: int = 0
    nx: int = 0
    ny: int = 0
    manual_rejected: int = 0
    avg_movies_found_per_hour: int = 0
    avg_movies_ready_per_hour: int = 0
    avg_movies_accepted_per_hour: int = 0
    avg_particles_extracted_per_mic: int = 0
    avg_particles_extracted_per_hour: int = 0


class Session(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    dumped_at: Optional[datetime.datetime] = None
    last_dumped_version: Optional[str] = None
    autodump: bool = True
    uid: str
    project_uid: str
    created_by_user_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    created_by_job_uid: Optional[str] = None
    tags: List[str] = []
    starred_by: List[str] = []
    deleted: bool = False
    last_accessed: Optional[SessionLastAccessed] = None
    workspace_stats: WorkspaceStats = WorkspaceStats()
    notes: str = ""
    notes_lock: Optional[str] = None
    imported: bool = False
    workspace_type: str = "live"
    session_uid: str
    session_dir: str
    status: SessionStatus = "paused"
    failed_at: List[datetime.datetime] = []
    running_at: List[datetime.datetime] = []
    paused_at: List[datetime.datetime] = []
    completed_at: Optional[datetime.datetime] = None
    cleared_at: Optional[datetime.datetime] = None
    elapsed_time: float = 0.0
    parameter_version: int = 0
    params: LivePreprocessingParams = LivePreprocessingParams()
    attributes: List[SessionAttribute] = [
        SessionAttribute(name="found_at", title="Timestamp", min=None, max=None, round=0),
        SessionAttribute(name="check_at", title="Check Stage Completed At", min=None, max=None, round=0),
        SessionAttribute(name="motion_at", title="Motion Stage Completed At", min=None, max=None, round=0),
        SessionAttribute(name="thumbs_at", title="Thumbs Stage Completed At", min=None, max=None, round=0),
        SessionAttribute(name="ctf_at", title="CTF Stage Completed At", min=None, max=None, round=0),
        SessionAttribute(name="pick_at", title="Pick Stage Completed At", min=None, max=None, round=0),
        SessionAttribute(name="extract_at", title="Extract Stage Completed At", min=None, max=None, round=0),
        SessionAttribute(
            name="manual_extract_at", title="Manual Extract Stage Completed At", min=None, max=None, round=0
        ),
        SessionAttribute(name="ready_at", title="Exposure Ready At", min=None, max=None, round=0),
        SessionAttribute(name="total_motion_dist", title="Total Motion (pix)", min=None, max=None, round=2),
        SessionAttribute(name="max_intra_frame_motion", title="Max In-Frame Motion", min=None, max=None, round=3),
        SessionAttribute(name="average_defocus", title="Defocus Avg. (Å)", min=None, max=None, round=0),
        SessionAttribute(name="defocus_range", title="Defocus Range (Å)", min=None, max=None, round=0),
        SessionAttribute(name="astigmatism_angle", title="Astigmatism Angle (deg)", min=None, max=None, round=1),
        SessionAttribute(name="astigmatism", title="Astigmatism", min=None, max=None, round=2),
        SessionAttribute(name="phase_shift", title="Phase Shift (deg)", min=None, max=None, round=1),
        SessionAttribute(name="ctf_fit_to_A", title="CTF Fit (Å)", min=None, max=None, round=3),
        SessionAttribute(name="ice_thickness_rel", title="Relative Ice Thickness", min=None, max=None, round=3),
        SessionAttribute(name="df_tilt_angle", title="Sample Tilt (deg)", min=None, max=None, round=1),
        SessionAttribute(name="total_manual_picks", title="Total Manual Picks", min=None, max=None, round=0),
        SessionAttribute(name="total_blob_picks", title="Total Blob Picks", min=None, max=None, round=0),
        SessionAttribute(name="blob_pick_score_median", title="Median Blob Pick Score", min=None, max=None, round=3),
        SessionAttribute(name="total_template_picks", title="Total Template Picks", min=None, max=None, round=0),
        SessionAttribute(
            name="template_pick_score_median", title="Median Template Pick Score", min=None, max=None, round=3
        ),
        SessionAttribute(
            name="total_extracted_particles",
            title="Total Manual Picker Particles Extracted",
            min=None,
            max=None,
            round=0,
        ),
        SessionAttribute(
            name="total_extracted_particles_manual",
            title="Total Blob Picker Particles Extracted",
            min=None,
            max=None,
            round=0,
        ),
        SessionAttribute(
            name="total_extracted_particles_blob",
            title="Total Template Picker Particles Extracted",
            min=None,
            max=None,
            round=0,
        ),
        SessionAttribute(
            name="total_extracted_particles_template", title="Total Particles Extracted", min=None, max=None, round=0
        ),
    ]
    picking_thresholds: PickingThresholds = PickingThresholds()
    compute_resources: LiveComputeResources = LiveComputeResources()
    phase_one_workers: Dict[str, RtpWorkerState] = {}
    phase_one_workers_soft_kill: List[Any] = []
    live_session_job_uid: Optional[str] = None
    file_engine_status: Literal["inactive", "running"] = "inactive"
    file_engine_last_run: Optional[datetime.datetime] = None
    max_timestamps: List[Any] = []
    known_files: List[Any] = []
    rtp_childs: List[RTPChild] = []
    avg_usage: List[Any] = []
    template_creation_job: Optional[str] = None
    template_creation_project: Optional[str] = None
    template_creation_num_particles_in: int = 0
    template_creation_ready: bool = False
    template_creation_info: List[TemplateClassInfo] = []
    exposure_groups: List[ExposureGroup] = []
    stats: SessionStats = SessionStats()
    data_management: DataManagementStats = DataManagementStats()
    import_signatures: ImportSignature = ImportSignature()
    exposure_summary: dict = {}
    particle_summary: dict = {}
    exposure_processing_priority: Literal["normal", "oldest", "latest", "alternate"] = "normal"
    cleared_extractions_at: Optional[datetime.datetime] = None
    cleared_extractions_size: float = 0.0
    last_compacted_amount: int = 0
    last_compacted_at: Optional[datetime.datetime] = None
    last_compacted_version: Optional[str] = None
    last_restored_amount: int = 0
    last_restored_at: Optional[datetime.datetime] = None
    compacted_exposures_count: int = 0
    restoration_notification_id: Optional[str] = None
    restoration_user_id: Optional[str] = None
    pre_restoration_size: int = 0
    phase2_class2D_restart: bool = False
    phase2_class2D_params_spec: Optional[LiveClass2DParams] = None
    phase2_class2D_params_spec_used: Optional[LiveClass2DParams] = None
    phase2_class2D_job: Optional[str] = None
    phase2_class2D_ready: bool = False
    phase2_class2D_ready_partial: bool = False
    phase2_class2D_info: List[TemplateClassInfo] = []
    phase2_class2D_num_particles_in: int = 0
    phase2_class2D_particles_out: Optional[Phase2ParticleOutputInfo] = None
    phase2_class2D_num_particles_seen: int = 0
    phase2_class2D_num_particles_accepted: int = 0
    phase2_class2D_num_particles_rejected: int = 0
    phase2_class2D_last_updated: Optional[datetime.datetime] = None
    phase2_select2D_job: Optional[str] = None
    phase2_abinit_restart: bool = False
    phase2_abinit_params_spec: LiveAbinitParams = LiveAbinitParams()
    phase2_abinit_job: Optional[str] = None
    phase2_abinit_ready: bool = False
    phase2_abinit_info: List[AbInitioVolumeInfo] = []
    phase2_abinit_num_particles_in: int = 0
    phase2_refine_restart: bool = False
    phase2_refine_params_spec: LiveRefineParams = LiveRefineParams()
    phase2_refine_params_spec_used: Optional[LiveRefineParams] = None
    phase2_refine_job: Optional[str] = None
    phase2_refine_ready: bool = False
    phase2_refine_ready_partial: bool = False
    phase2_refine_num_particles_in: int = 0
    phase2_refine_last_updated: Optional[datetime.datetime] = None
    athena_epu_run_id: Optional[str] = None
    is_multigrid_epu_run: bool = False
    is_gracefully_pausing: bool = False
    computed_stats_last_run_time: Optional[datetime.datetime] = None
    last_processed_exposure_priority: Literal["normal", "oldest", "latest", "alternate"] = "oldest"
    ecl: ECLSessionProperties = ECLSessionProperties()
