# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Any, Dict, Union

from pydantic import BaseModel


class When(BaseModel):
    """
    Boolean expression composition utility class that encodes logic for when job
    metadata (e.g., param hidden state) should be True or False based on the
    state of a job.

    Avoid using `When` class directly. Instead, compose `WhenParam` and
    `WhenConnection` instances with `|` ("or"), `&` ("and"), `~` ("not")
    operators or their functional equivalents (defined here).
    """

    expr: Union[str, Dict[str, Any]]
