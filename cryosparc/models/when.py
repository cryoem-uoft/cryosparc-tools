# THIS FILE IS AUTO-GENERATED, DO NOT EDIT DIRECTLY
# SEE dev/api_generate_models.py
from typing import Any, Dict, Union

from pydantic import BaseModel


class When(BaseModel):
    """
    Boolean expression composition utility class that encodes logic for when job
    metadata (e.g., param hidden state) should be True or False based on the
    state of a job.

    Examples:

        Values for ``expr`` for various common cases:

        When param foo is truthy:
            ``"foo"``

        When param foo equals 42:
            ``{"op": "==", "param": "foo", "value": 42}``

        When param bar is in the list [1, 2, 3]:
            ``{"op": "in", "param": "bar", "value": [1, 2, 3]}``

        When input connection "particles" is present:
            ``{"connection": "particles"}``

        When param foo equals 42 AND input connection "particles" is present:
            ``{"op": "and", "left": {"op": "==", "param": "foo", "value": 42}, "right": {"connection": "particles"}}``
    """

    expr: Union[str, Dict[str, Any]]
    """
    Logical expression encoded as a dictionary or a string (param name).
    """
