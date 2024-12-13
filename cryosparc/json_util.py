import base64
from datetime import datetime
from pathlib import PurePath
from typing import Any, Mapping

import numpy as n
from pydantic import BaseModel


def api_default(obj: Any) -> Any:
    """
    json.dump "default" argument for sending objects over a JSON API. Ensures
    that special non-JSON types such as Path are NDArray are encoded correctly.
    """
    if isinstance(obj, n.floating):
        if n.isnan(obj):
            return float(0)
        elif n.isposinf(obj):
            return float("inf")
        elif n.isneginf(obj):
            return float("-inf")
        return float(obj)
    elif isinstance(obj, n.integer):
        return int(obj)
    elif isinstance(obj, n.ndarray):
        return ndarray_to_json(obj)
    elif isinstance(obj, bytes):
        return binary_to_json(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, PurePath):
        return str(obj)
    elif isinstance(obj, BaseModel):
        return obj.model_dump(exclude_none=True)
    else:
        return obj


def api_object_hook(dct: Mapping[str, Any]):
    """
    json.dump "object_hook" argument for receiving JSON from an API request.
    Ensures that special objects that are actually encoded numpy arrays or bytes
    are decoded as such.
    """
    if "$ndarray" in dct:
        return ndarray_from_json(dct)
    elif "$binary" in dct:
        return binary_from_json(dct)
    else:
        return dct  # pydantic will take care of everything else


def binary_to_json(binary: bytes):
    """
    Encode bytes as a JSON-serializeable object
    """
    return {"$binary": {"base64": base64.b64encode(binary).decode()}}


def binary_from_json(dct: Mapping[str, Any]) -> bytes:
    if "base64" not in dct["$binary"] or not isinstance(b64 := dct["$binary"]["base64"], str):
        raise TypeError(f"$binary base64 must be a string: {dct}")
    return base64.b64decode(b64.encode())


def ndarray_to_json(arr: n.ndarray):
    """
    Encode a numpy array a JSON-serializeable object.
    """
    return {
        "$ndarray": {
            "base64": base64.b64encode(arr.data).decode(),
            "dtype": str(arr.dtype),
            "shape": arr.shape,
        }
    }


def ndarray_from_json(dct: Mapping[str, Any]):
    """
    Decode a serialized numpy array.
    """
    data = base64.b64decode(dct["$ndarray"]["base64"])
    return n.frombuffer(data, dct["$ndarray"]["dtype"]).reshape(dct["$ndarray"]["shape"])
