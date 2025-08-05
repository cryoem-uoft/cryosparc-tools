# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, ConfigDict


class LivePreprocessingParams(BaseModel):
    gainref_flip_x: bool = False
    """
    Flip gain ref left-to-right (in X axis)
    """
    gainref_flip_y: bool = False
    """
    Flip gain ref top-to-bottom (in Y axis)
    """
    gainref_rotate_num: int = 0
    """
    Rotate gain ref counter-clockwise by 90 degrees this many times
    """
    psize_A: float = 0.0
    """
    Pixel size of the raw movie data in Angstroms
    """
    accel_kv: float = 0.0
    cs_mm: float = 0.0
    total_dose_e_per_A2: float = 0.0
    phase_plate: bool = False
    """
    Were the images collected using a phase plate?
    """
    neg_stain: bool = False
    """
    Are the samples negative stain (True) or cryo (False)?
    """
    eer_upsampfactor: int = 2
    """
    EER upsampling factor (applies to .eer/.ecc format data only.
    """
    eer_numfractions: int = 40
    """
    EER number of fractions (applies to .eer/.ecc format data only.
    """
    motion_res_max_align: float = 5
    """
    Maximum resolution (in Å) to consider when aligning frames. Generally, betwen 5Å and 3Å is best.
    """
    bfactor: float = 500
    """
    B-factor that blurs frames before aligning. Generally 500 to 100 is best.
    """
    frame_start: int = 0
    """
    Which frame number, starting at zero, to begin motion correction from. This value controls how many early frames are dropped from the motion corrected result. This value will also be used in local motion correction.
    """
    frame_end: Optional[int] = None
    """
    Which frame number, starting at zero, to not include in motion correction, also excluding all frames after this one. Generally this does not improve results, as later frames are downweighted during dose weighting in local motion correction.
    """
    output_fcrop_factor: float = 1
    """
    Output Fourier cropping factor. 1.0 means no cropping, 0.5 means crop to 1/2 the resolution, etc. Only 1, 0.75, 0.5, 0.25 are allowed values
    """
    override_total_exp: Optional[float] = None
    """
    Override the dose (in total e/Å^2 over the exposure) that was given at import time but can be overridden here.
    """
    variable_dose: bool = False
    """
    Enable correct processing when frames have variable dose fractionation
    """
    smooth_lambda_cal: float = 0.5
    """
    Calibrated smoothing constant applied to trajectories (None to autotune)
    """
    motion_override_K_Z: Optional[int] = None
    """
    Override automatically selected spline order for Z dimension (time)
    """
    motion_override_K_Y: Optional[int] = None
    """
    Override automatically selected spline order for Y dimension (vertical)
    """
    motion_override_K_X: Optional[int] = None
    """
    Override automatically selected spline order for X dimension (horizontal)
    """
    optimize_for_gpu_memory: bool = False
    """
    If running out of GPU memory, this option can be used to prioritize memory use at the expense of speed (BETA). The results are unchanged.
    """
    output_f16: bool = False
    """
    Reduces the output precision from 32 bits to 16 bits, saving hard drive space.
    """
    amp_contrast: float = 0.1
    """
    Amplitude constrast to use. Typically 0.07 or 0.1 for cryo-EM data.
    """
    ctf_res_min_align: float = 25
    """
    Minimum resolution (in Å) to consider when estimating CTF.
    """
    ctf_res_max_align: float = 4
    """
    Maximum resolution (in Å) to consider when estimating CTF.
    """
    df_search_min: float = 1000
    """
    Defocus range for gridsearch.
    """
    df_search_max: float = 40000
    """
    Defocus range for gridsearch.
    """
    do_phase_shift_search_refine: bool = False
    """
    Whether to carry out search and refinement over phase shift.
    """
    phase_shift_min: float = 0
    """
    Phase-shift range for gridsearch.
    """
    phase_shift_max: float = 3.141592653589793
    """
    Phase-shift range for gridsearch.
    """
    do_phase_shift_refine_only: bool = False
    """
    Whether to carry out refinement over phase shift only
    """
    ctf_override_K_Y: Optional[int] = None
    """
    Override automatically selected spline order for Y dimension (vertical)
    """
    ctf_override_K_X: Optional[int] = None
    """
    Override automatically selected spline order for X dimension (horizontal)
    """
    classic_mode: bool = False
    """
    Uses the old Patch CTF algorithm (cryoSPARC v.2.15 and earlier) intead of the new one.
    """
    current_picker: Literal["blob", "template", "deep"] = "blob"
    """
    Which picker type to use on future exposures.
    """
    blob_diameter_min: float = 0.0
    """
    Min Particle diameter (Å)
    """
    blob_diameter_max: float = 0.0
    """
    Max Particle diameter (Å)
    """
    use_circle: bool = True
    use_ellipse: bool = False
    use_ring: bool = False
    blob_lowpass_res_template: float = 20
    """
    Lowpass filter to apply to templates, (Å)s
    """
    blob_lowpass_res: float = 20
    """
    Lowpass filter to apply, (Å)s
    """
    blob_angular_spacing_deg: float = 5
    """
    Angular sampling of templates in degrees. Lower value will mean finer rotations.
    """
    blob_use_ctf: bool = False
    """
    Whether to use micrograph CTF to filter the templates
    """
    blob_min_distance: float = 1.0
    """
    Minimum distance between particles in units of particle diameter (min diameter for blob picker). The lower this value, the more and closer particles it picks.
    """
    blob_num_process: Optional[int] = None
    """
    Number of micrographs to process. None means all.
    """
    blob_num_plot: int = 10
    """
    Number of micrographs to plot.
    """
    blob_max_num_hits: int = 4000
    """
    Maximum number of local maxima (peaks) considered.
    """
    template_diameter: Optional[float] = None
    """
    Particle diameter (Å)
    """
    template_lowpass_res_template: float = 20
    """
    Lowpass filter to apply to templates, (Å)s
    """
    template_lowpass_res: float = 20
    """
    Lowpass filter to apply, (Å)s
    """
    template_angular_spacing_deg: float = 5
    """
    Angular sampling of templates in degrees. Lower value will mean finer rotations.
    """
    template_use_ctf: bool = True
    """
    Whether to use micrograph CTF to filter the templates
    """
    template_min_distance: float = 0.5
    """
    Minimum distance between particles in units of particle diameter. The lower this value, the more and closer particles it picks.
    """
    template_num_process: Optional[int] = None
    """
    Number of micrographs to process. None means all.
    """
    template_num_plot: int = 10
    """
    Number of micrographs to plot.
    """
    template_max_num_hits: int = 4000
    """
    Maximum number of local maxima (peaks) considered.
    """
    templates_from_job: Optional[str] = None
    templates_selected: Optional[str] = None
    thresh_score_min: Optional[float] = None
    """
    Minimum picking score threshold
    """
    thresh_power_min: Optional[float] = None
    """
    Minimum picking power threshold
    """
    thresh_power_max: Optional[float] = None
    """
    Maximum picking power threshold
    """
    box_size_pix: int = 0
    """
    Size of box to be extracted from micrograph.
    """
    bin_size_pix: Optional[int] = None
    """
    Size of particle boxes after they have been extracted. None means use the same as the extraction box size
    """
    extract_f16: bool = False
    """
    Reduces the output precision from 32 bits to 16 bits, saving hard drive space.
    """
    do_plotting: bool = False
    """
    Enable plotting in the RTP Worker for every 20 movies processed.
    """

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
