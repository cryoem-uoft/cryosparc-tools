# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing_extensions import TypedDict


class LicenseInstance(TypedDict, total=False):
    """
    Information about license allowance and usage for a specific instance.
    """

    id: str
    current_in_use: int
    num_queued_jobs: int
    max_licenses_available: int
    default_max_licenses_available: int
    queued_jobs: str
    active_jobs: str
    version: str
    alias: str
    group_id: str
    reserved_licenses: int
    min_reserved_licenses: int
    license_developer: bool
    license_live_enabled: bool
    commercial_instance: bool
    valid: bool
    this_instance: bool


class UpdateTag(TypedDict):
    show_message: bool
    message: str
