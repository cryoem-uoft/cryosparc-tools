"""Stream processing utilities"""

from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    AsyncIterator,
    BinaryIO,
    Generator,
    Iterator,
    Optional,
    Union,
)

from typing_extensions import Protocol

if TYPE_CHECKING:
    from typing_extensions import Self  # not present in typing-extensions=3.7


class AsyncBinaryIO(Protocol):
    async def read(self, n: Optional[int] = None) -> bytes: ...


class BinaryIteratorIO(BinaryIO):
    """Read through a iterator that yields bytes as if it was a file"""

    def __init__(self, iter: Union[Iterator[bytes], Generator[bytes, Any, Any]]):
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
        out = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                out.append(m)
        else:
            while n > 0:
                m = self._read1(n)
                if not m:
                    break
                n -= len(m)
                out.append(m)
        return b"".join(out)


class AsyncBinaryIteratorIO(AsyncBinaryIO):
    """Similar to BinaryIteratorIO except the iterator yields bytes asynchronously"""

    def __init__(self, iter: Union[AsyncIterator[bytes], AsyncGenerator[bytes, Any]]):
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
        out = []
        if n is None or n < 0:
            while True:
                m = self._read1()
                if not m:
                    break
                out.append(m)
        else:
            while n > 0:
                m = await self._read1(n)
                if not m:
                    break
                n -= len(m)
                out.append(m)
        return b"".join(out)


class Streamable(ABC):
    @classmethod
    def mime_type(cls) -> str:
        """
        Return the binary mime type to use in HTTP requests when streaming this
        data e.g., "application/x-cryosparc-dataset"
        """
        return f"application/x-cryosparc-{cls.__name__.lower()}"

    @classmethod
    def api_schema(cls):
        """
        Schema to use when an API endpoint returns or requests this streamable
        instance in the request or response body.
        """
        return {
            "description": f"A binary stream representing a CryoSPARC {cls.__name__}",
            "content": {cls.mime_type(): {"schema": {"title": cls.__name__, "type": "string", "format": "binary"}}},
        }

    @classmethod
    @abstractmethod
    def load(cls, file: Union[str, PurePath, IO[bytes]]) -> "Self":
        """
        The given stream param must at least implement an async read method
        """
        ...

    @classmethod
    def from_iterator(cls, source: Iterator[bytes]):
        return cls.load(BinaryIteratorIO(source))

    @classmethod
    @abstractmethod
    async def from_async_stream(cls, stream: AsyncBinaryIO) -> "Self":
        """
        Asynchronously load from the given binary stream. The given stream
        parameter must at least have ``async read(n: int | None) -> bytes`` method.
        """
        ...

    @classmethod
    async def from_async_iterator(cls, iterator: Union[AsyncIterator[bytes], AsyncGenerator[bytes, None]]):
        return await cls.from_async_stream(AsyncBinaryIteratorIO(iterator))

    @abstractmethod
    def stream(self) -> Generator[bytes, None, None]: ...

    async def astream(self):
        for chunk in self.stream():
            yield chunk
