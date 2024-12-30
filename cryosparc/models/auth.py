# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str
