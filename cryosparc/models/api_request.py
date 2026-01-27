# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class AppSession(BaseModel):
    """
    Data required to verify a web application session.
    """

    user_id: str
    """
    """
    session_id: str
    """
    """
    signature: str
    """
    """


class SHA256Password(BaseModel):
    """
    SHA-256 hashed password data, in hexadecimal format.
    """

    password: str
    """
    """
