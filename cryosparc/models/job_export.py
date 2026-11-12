# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Literal

from pydantic import BaseModel, RootModel

TransferMethod = Literal["symlink", "absolute", "copy", "omit"]


class ExportResultPreview(BaseModel):
    """
    Preview of exporting a single output result.
    """

    name: str
    """
    """
    dtype: str
    """
    """
    num_items: int
    """
    """
    method: TransferMethod
    """
    """
    num_files: int
    """
    """
    path_values: list[str] | None = None
    """
    Path values from this result's files, or None if paths were not requested.
    """
    size_bytes: int | None = None
    """
    """


class ExportOutputPreview(BaseModel):
    """
    Preview of exporting a job output, broken down by result.
    """

    name: str
    """
    """
    type: Literal[
        "exposure",
        "particle",
        "template",
        "volume",
        "volume_multi",
        "mask",
        "live",
        "ml_model",
        "symmetry_candidate",
        "flex_mesh",
        "flex_model",
        "hyperparameter",
        "denoise_model",
        "annotation_model",
    ]
    """
    """
    title: str
    """
    """
    results: list[ExportResultPreview]
    """
    """
    total_size_bytes: int | None = None
    """
    """
    total_num_files: int
    """
    Total files across non-omitted results.
    """


class ExportResultsSelection(RootModel):
    """
    Which results to export and how:
    - dict: exactly these result names are included, regardless of the default
      omit policy, each mapped to a transfer method, or ``True`` to use the
      default method for that result (currently always symlink).
    - list: shorthand for including these names with their default method, e.g.
      ``["map", "map_sharp"]`` is equivalent to ``{"map": True, "map_sharp": True}``.
    """

    root: dict[str, TransferMethod] = {}
    """
    Which results to export and how:
    - dict: exactly these result names are included, regardless of the default
      omit policy, each mapped to a transfer method, or ``True`` to use the
      default method for that result (currently always symlink).
    - list: shorthand for including these names with their default method, e.g.
      ``["map", "map_sharp"]`` is equivalent to ``{"map": True, "map_sharp": True}``.
    """


class ExportOutputsSelection(RootModel):
    """
    Per-output result selection for exporting jobs.
    An output absent from a non-empty mapping is excluded. An empty mapping
    resolves every output using the effective result defaults.
    """

    root: dict[str, ExportResultsSelection] = {}
    """
    Per-output result selection for exporting jobs.
    An output absent from a non-empty mapping is excluded. An empty mapping
    resolves every output using the effective result defaults.
    """


class ExportResultDefaults(BaseModel):
    """
    Instance-wide defaults for which export results get omitted
    """

    omit_dtypes: list[str] | None = None
    """
    """
    omit_result_names: list[str] | None = None
    """
    """
