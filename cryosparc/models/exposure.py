# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from .session_params import LivePreprocessingParams


class CTF(BaseModel):
    accel_kv: float
    amp_contrast: float
    cross_corr_ctffind4: float
    cs_mm: float
    ctf_fit_to_A: float
    df1_A: float
    df2_A: float
    df_angle_rad: float
    exp_group_id: int
    fig_of_merit_gctf: float
    path: str
    phase_shift_rad: float
    type: str


class CtfStats(BaseModel):
    cross_corr: int
    ctf_fit_to_A: float
    df_range: List[Any]
    df_tilt_normal: List[Any]
    diag_image_path: str
    fit_data_path: str
    ice_thickness_rel: float
    spectrum_dim: int
    type: str


class ECLExposureProperties(BaseModel):
    do_athena_results_upload: bool = False


class StatBlob(BaseModel):
    binfactor: int
    idx: int
    path: str
    psize_A: float
    shape: List[int]


class GainRefBlob(BaseModel):
    flip_x: int
    flip_y: int
    idx: int
    path: str
    rotate_num: int
    shape: List[int]


class MicrographBlob(BaseModel):
    format: str
    idx: int
    is_background_subtracted: bool
    path: str
    psize_A: float
    shape: List[int]


class MovieBlob(BaseModel):
    format: str
    has_defect_file: bool = False
    import_sig: str = "0"
    """
    Unique 64 bit import signature for this exposure, created by computing the
    sha1 of the original absolute path and taking the first 8 bytes.

    Encoded as a string when saved to mongo or JSON to avoid 32 bit overflow
    errors. When using instances of the MovieBlob model from Python,
    ``import_sig`` should be treated as a regular int.
    """
    is_gain_corrected: bool
    path: str
    psize_A: float
    shape: List[int]


class MScopeParams(BaseModel):
    accel_kv: float
    beam_shift: List[int] = [0, 0]
    beam_shift_known: int = 0
    cs_mm: float
    defect_path: Optional[str] = None
    exp_group_id: int
    neg_stain: int = 0
    phase_plate: int
    total_dose_e_per_A2: float = 0


class MotionData(BaseModel):
    frame_end: int
    frame_start: int
    idx: int
    path: str
    psize_A: float
    type: str
    zero_shift_frame: int


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
    uid: str = "0"


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
    """
    Exposures have an list of length 1 values for each field in each prefix
    below.
    """
    particle_manual: ParticleManual = ParticleManual()
    """
    Manual picks have a list of values for each field in each prefix
    """
    particle_blob: ParticleInfo = ParticleInfo()
    """
    Blob picks have a list of fields, a count, and a path to a cs file
    """
    particle_template: ParticleInfo = ParticleInfo()
    """
    template picks have a list of fields, a count, and a path to a cs file
    """
    particle_deep: Dict[str, Any] = {}
    """
    Unused for now
    """
    particle_extracted: Union[List[PickerInfoElement], ParticleInfo] = ParticleInfo()
    """
    Starts as empty dict, but gets populated as a list of dictionaries. Used for 2-slot extraction.
    """
    particle_manual_extracted: PickerInfoElement = PickerInfoElement()
    """
    Starts as empty dict, but gets populated as a list of dictionaries
    """


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
    """
    When this object was last modified.
    """
    created_at: datetime.datetime = datetime.datetime(1970, 1, 1, 0, 0, tzinfo=datetime.timezone.utc)
    """
    When this object was first created. Imported objects such as projects
    and jobs will retain the created time from their original CryoSPARC instance.
    """
    uid: int
    project_uid: str
    session_uid: str
    exp_group_id: int
    abs_file_path: str
    """
    only used when first reading in this file, otherwise use os.path.join(proj_dir_abs, groups.exposures.movie_blob.path)
    """
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
