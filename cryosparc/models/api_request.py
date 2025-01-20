# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class AppSession(BaseModel):
    user_id: str
    session_id: str
    signature: str


class SHA256Password(BaseModel):
    password: str
