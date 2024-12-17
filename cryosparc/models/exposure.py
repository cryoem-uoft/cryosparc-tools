# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .session_params import LivePreprocessingParams


class CTF(BaseModel):
    accel_kv: float = 0
    amp_contrast: float = 0
    cross_corr_ctffind4: float = 0
    cs_mm: float = 0
    ctf_fit_to_A: float = 0
    df1_A: float = 0
    df2_A: float = 0
    df_angle_rad: float = 0
    exp_group_id: int = 0
    fig_of_merit_gctf: float = 0
    path: str = "."
    phase_shift_rad: float = 0
    type: str = ""


class CtfStats(BaseModel):
    cross_corr: int = 0
    ctf_fit_to_A: float = 0
    df_range: List[Any] = [0, 0]
    df_tilt_normal: List[Any] = [0, 0]
    diag_image_path: str = "."
    fit_data_path: str = "."
    ice_thickness_rel: float = 0
    spectrum_dim: int = 0
    type: str = ""


class ECLExposureProperties(BaseModel):
    do_athena_results_upload: bool = False


class StatBlob(BaseModel):
    binfactor: int = 0
    idx: int = 0
    path: str = "."
    psize_A: float = 0
    shape: List[int] = [0, 0]


class GainRefBlob(BaseModel):
    flip_x: int = 0
    flip_y: int = 0
    idx: int = 0
    path: str = "."
    rotate_num: int = 0
    shape: List[int] = []


class MicrographBlob(BaseModel):
    format: str = ""
    idx: int = 0
    is_background_subtracted: bool = False
    path: str = "."
    psize_A: float = 0
    shape: List[int] = [0, 0]


class MovieBlob(BaseModel):
    format: str = ""
    has_defect_file: bool = False
    import_sig: int = 0
    is_gain_corrected: bool = False
    path: str = "."
    psize_A: float = 0
    shape: List[int] = []


class MScopeParams(BaseModel):
    accel_kv: float = 0
    beam_shift: List[int] = [0, 0]
    beam_shift_known: int = 0
    cs_mm: float = 0
    defect_path: Optional[str] = None
    exp_group_id: int = 0
    neg_stain: int = 0
    phase_plate: int = 0
    total_dose_e_per_A2: float = 0


class MotionData(BaseModel):
    frame_end: int = 0
    frame_start: int = 0
    idx: int = 0
    path: str = "."
    psize_A: float = 0
    type: str = ""
    zero_shift_frame: int = 0


class ExposureElement(BaseModel):
    background_blob: Optional[StatBlob] = None
    ctf: Optional[CTF] = None
    ctf_stats: Optional[CtfStats] = None
    gain_ref_blob: Optional[GainRefBlob] = None
    micrograph_blob: Optional[MicrographBlob] = None
    micrograph_blob_non_dw: Optional[MicrographBlob] = None
    micrograph_blob_thumb: Optional[MicrographBlob] = None
    movie_blob: Optional[MovieBlob] = None
    mscope_params: Optional[MScopeParams] = None
    rigid_motion: Optional[MotionData] = None
    spline_motion: Optional[MotionData] = None
    uid: int = 0


class PickerLocations(BaseModel):
    center_x_frac: List[float] = []
    center_y_frac: List[float] = []


class ParticleManual(BaseModel):
    count: int = 0
    fields: List[str] = []
    path: str = "."
    location: PickerLocations = PickerLocations()


class ParticleInfo(BaseModel):
    count: int = 0
    fields: List[str] = []
    path: str = "."


class PickerInfoElement(BaseModel):
    count: int = 0
    fields: List[str] = []
    path: str = "."
    output_shape: Optional[int] = None
    picker_type: Optional[Literal["blob", "template", "manual"]] = None


class ExposureGroups(BaseModel):
    """
    Metadata about outputs produced by a specific exposure
    """

    exposure: ExposureElement = ExposureElement()
    particle_manual: ParticleManual = ParticleManual()
    particle_blob: ParticleInfo = ParticleInfo()
    particle_template: ParticleInfo = ParticleInfo()
    particle_deep: dict = {}
    particle_extracted: Union[List[PickerInfoElement], ParticleInfo] = ParticleInfo()
    particle_manual_extracted: PickerInfoElement = PickerInfoElement()


class ExposureAttributes(BaseModel):
    """
    Exposure processing metadata. The "round" param is used for display in the
    UI (defaults to 0 if not specified).
    """

    found_at: float = 0
    check_at: float = 0
    motion_at: float = 0
    thumbs_at: float = 0
    ctf_at: float = 0
    pick_at: float = 0
    extract_at: float = 0
    manual_extract_at: float = 0
    ready_at: float = 0
    total_motion_dist: float = 0
    max_intra_frame_motion: float = 0
    average_defocus: float = 0
    defocus_range: float = 0
    astigmatism_angle: float = 0
    astigmatism: float = 0
    phase_shift: float = 0
    ctf_fit_to_A: float = 0
    ice_thickness_rel: float = 0
    df_tilt_angle: float = 0
    total_manual_picks: int = 0
    total_blob_picks: int = 0
    blob_pick_score_median: float = 0
    total_template_picks: int = 0
    template_pick_score_median: float = 0
    total_extracted_particles: int = 0
    total_extracted_particles_manual: int = 0
    total_extracted_particles_blob: int = 0
    total_extracted_particles_template: int = 0


class Exposure(BaseModel):
    id: str = Field("000000000000000000000000", alias="_id")
    updated_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    uid: int
    project_uid: str
    session_uid: str
    exp_group_id: int
    abs_file_path: str
    size: int
    discovered_at: datetime.datetime
    picker_type: Optional[Literal["blob", "template", "manual"]] = None
    deleted: bool = False
    parameter_version: Optional[int] = None
    stage: Literal[
        "go_to_found",
        "found",
        "check",
        "motion",
        "ctf",
        "thumbs",
        "pick",
        "extract",
        "extract_manual",
        "ready",
        "restoring",
        "restoring_motion",
        "restoring_thumbs",
        "restoring_ctf",
        "restoring_extract",
        "restoring_extract_manual",
        "compacted",
    ] = "found"
    fail_count: int = 0
    failed: bool = False
    fail_reason: str = ""
    in_progress: bool = False
    manual_reject: bool = False
    threshold_reject: bool = False
    test: bool = False
    worker_juid: Optional[str] = None
    priority: int = 0
    groups: ExposureGroups = ExposureGroups()
    attributes: ExposureAttributes = ExposureAttributes()
    test_parameters: Optional[LivePreprocessingParams] = None
    preview_img_1x: List[str] = []
    preview_img_2x: List[str] = []
    thumb_shape: List[int] = []
    micrograph_shape: List[int] = []
    micrograph_psize: Optional[float] = None
    ecl: ECLExposureProperties = ECLExposureProperties()
