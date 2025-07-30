# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class SystemInfo(BaseModel):
    version: str
    master_hostname: str
    port_app: int
    port_mongo: int
    port_api: int
    port_command_vis: int
    port_rtp_webapp: int
    is_db_auth_enabled: bool
