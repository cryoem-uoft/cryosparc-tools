from .dataset import Dataset
from .registry import register_stream_class
from .stream import Stream

register_stream_class(Dataset)
register_stream_class(Stream)
