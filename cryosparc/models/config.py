# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class SystemInfo(BaseModel):
    """
    CryoSPARC general system information.
    """

    version: str
    """
    Running version of CryoSPARC, including any applied patches
    """
    master_hostname: str
    """
    Hostname of the master node
    """
    port_app: int
    """
    Port number for the main CryoSPARC web application
    """
    port_mongo: int
    """
    Port number for the MongoDB database
    """
    port_api: int
    """
    Port number for the CryoSPARC API service
    """
    port_command_vis: int
    """
    Port number for visualization service
    """
    port_rtp_webapp: int
    """
    Port number for real-time API application service.
    """
    is_db_auth_enabled: bool
    """
    Whether authentication is enabled for the database.
    """
