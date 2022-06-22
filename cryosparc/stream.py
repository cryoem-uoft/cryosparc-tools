"""Stream processing utilities
"""
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterator, BinaryIO, Generator, Iterable, Iterator, Optional, Union
import numpy as n


class BinaryIteratorIO(BinaryIO):
    """Read through a iterator that yields bytes as if it was a file"""

    def __init__(self, iter: Iterator[bytes]):
        self._iter = iter
        self._left = b""

    def readable(self):
        return True

    def seekable(self) -> bool:
        return False

    def _read1(self, n: Optional[int] = None):
        while not self._left:
            try:
                self._left = next(self._iter)
            except StopIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret) :]
        return ret

    def read(self, n: Optional[int] = None):
        l = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                l.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                l.append(m)
        return b"".join(l)


class AsyncBinaryIteratorIO(BinaryIO):
    """Similar to BinaryIteratorIO except the iterator yields bytes asynchronously"""

    def __init__(self, iter: Union[AsyncIterator[bytes], AsyncGenerator[bytes, None]]):
        self._iter = iter
        self._left = b""

    def readable(self):
        return True

    def seekable(self) -> bool:
        return False

    async def _read1(self, n: Optional[int] = None):
        while not self._left:
            try:
                self._left = await self._iter.__anext__()
            except StopAsyncIteration:
                break
        ret = self._left[:n]
        self._left = self._left[len(ret) :]
        return ret

    async def read(self, n: Optional[int] = None):
        l = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                l.append(m)
        else:
            while n > 0:
                m = await self._read1(n)
                if not m:
                    break
                n -= len(m)
                l.append(m)
        return b"".join(l)


class Streamable(ABC):
    @classmethod
    def mime_type(cls) -> str:
        """Return the binary mime type to use in HTTP requests when streaming
        this data e.g., "application/x-cryosparc-dataset"
        """
        return f"application/x-cryosparc-{cls.__name__.lower()}"

    @classmethod
    def api_schema(cls):
        """Schema to use when a FastAPI endpoint returns or requests this
        streamable instance in the request or response body
        """
        mime_type = cls.mime_type()
        return {
            "description": f"A binary stream representing a cryoSPARC {cls.__name__}",
            "content": {mime_type: {"schema": {"title": cls.__name__, "type": "string", "format": "binary"}}},
        }

    @classmethod
    @abstractmethod
    def from_stream(cls, source: BinaryIO, fields: Optional[Iterable[str]] = None) -> Any:
        """The given stream param must at least implement an async read method"""
        ...

    @classmethod
    @abstractmethod
    async def from_async_stream(cls, stream, fields: Optional[Iterable[str]] = None) -> "Streamable":
        """The given stream param must at least implement an async read method"""
        ...

    @classmethod
    def from_binary_iterator(cls, source: Iterator[bytes]):
        return cls.from_stream(BinaryIteratorIO(source))

    @classmethod
    async def from_async_binary_iterator(cls, iterator: Union[AsyncIterator[bytes], AsyncGenerator[bytes, None]]):
        return await cls.from_async_stream(AsyncBinaryIteratorIO(iterator))

    @abstractmethod
    def to_stream(self, fields: Optional[Iterable[str]] = None) -> Generator[bytes, None, None]:
        ...

    async def to_async_stream(self, fields=None):
        for chunk in self.to_stream(fields):
            yield chunk


def read_len(source: BinaryIO):
    """Read 4 bytes for the length of the upcoming data"""
    return int(n.frombuffer(source.read(4), dtype=n.uint32)[0])


async def aread_len(stream):
    """Asynchronously reaad 4 bytes for the length of the upcoming data"""
    return int(n.frombuffer(await stream.read(4), dtype=n.uint32)[0])
