# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class LivePreprocessingParams(BaseModel):
    gainref_flip_x: bool = False
    gainref_flip_y: bool = False
    gainref_rotate_num: int = 0
    psize_A: float = 0.0
    accel_kv: float = 0.0
    cs_mm: float = 0.0
    total_dose_e_per_A2: float = 0.0
    phase_plate: bool = False
    neg_stain: bool = False
    eer_upsampfactor: int = 2
    eer_numfractions: int = 40
    motion_res_max_align: float = 5
    bfactor: float = 500
    frame_start: int = 0
    frame_end: Optional[int] = None
    output_fcrop_factor: float = 1
    override_total_exp: Optional[float] = None
    variable_dose: bool = False
    smooth_lambda_cal: float = 0.5
    motion_override_K_Z: Optional[int] = None
    motion_override_K_Y: Optional[int] = None
    motion_override_K_X: Optional[int] = None
    optimize_for_gpu_memory: bool = False
    output_f16: bool = False
    amp_contrast: float = 0.1
    ctf_res_min_align: float = 25
    ctf_res_max_align: float = 4
    df_search_min: float = 1000
    df_search_max: float = 40000
    do_phase_shift_search_refine: bool = False
    phase_shift_min: float = 0
    phase_shift_max: float = 3.141592653589793
    do_phase_shift_refine_only: bool = False
    ctf_override_K_Y: Optional[int] = None
    ctf_override_K_X: Optional[int] = None
    classic_mode: bool = False
    current_picker: Literal["blob", "template", "deep"] = "blob"
    blob_diameter_min: float = 0.0
    blob_diameter_max: float = 0.0
    use_circle: bool = True
    use_ellipse: bool = False
    use_ring: bool = False
    blob_lowpass_res_template: float = 20
    blob_lowpass_res: float = 20
    blob_angular_spacing_deg: float = 5
    blob_use_ctf: bool = False
    blob_min_distance: float = 1.0
    blob_num_process: Optional[int] = None
    blob_num_plot: int = 10
    blob_max_num_hits: int = 4000
    template_diameter: Optional[float] = None
    template_lowpass_res_template: float = 20
    template_lowpass_res: float = 20
    template_angular_spacing_deg: float = 5
    template_use_ctf: bool = True
    template_min_distance: float = 0.5
    template_num_process: Optional[int] = None
    template_num_plot: int = 10
    template_max_num_hits: int = 4000
    templates_from_job: Optional[str] = None
    templates_selected: Optional[str] = None
    thresh_score_min: Optional[float] = None
    thresh_power_min: Optional[float] = None
    thresh_power_max: Optional[float] = None
    box_size_pix: int = 0
    bin_size_pix: Optional[int] = None
    extract_f16: bool = False
    do_plotting: bool = False

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...


class LiveAbinitParams(BaseModel):
    abinit_symmetry: str = "C1"
    abinit_K: int = 1
    abinit_num_particles: Optional[int] = None

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...


class LiveClass2DParams(BaseModel):
    class2D_K: int = 50
    class2D_max_res: int = 6
    class2D_window_inner_A: Optional[float] = None
    compute_use_ssd: bool = True
    psize_mic: Optional[float] = None

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...


class LiveRefineParams(BaseModel):
    refine_symmetry: str = "C1"
    psize_mic: Optional[float] = None

    model_config = ConfigDict(extra="allow")
    if TYPE_CHECKING:

        def __init__(self, **kwargs: Any) -> None: ...
        def __getattr__(self, key: str) -> Any: ...
