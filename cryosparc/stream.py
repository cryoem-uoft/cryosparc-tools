"""Stream processing utilities"""

from abc import ABC, abstractmethod
from pathlib import PurePath
from typing import (
    IO,
    AsyncIterator,
    Awaitable,
    BinaryIO,
    Iterator,
    Optional,
    Protocol,
    Union,
    overload,
)

from typing_extensions import Self

from .constants import EIGHT_MIB
from .util import bopen


class AsyncBinaryIO(Protocol):
    def read(self, size: int = ..., /) -> Awaitable[bytes]: ...


class AsyncBinaryIterator(Protocol):
    def __aiter__(self) -> AsyncIterator[bytes]: ...
    def __anext__(self) -> Awaitable[bytes]: ...


class BinaryIteratorIO(BinaryIO, Iterator[bytes]):
    """Read through a iterator that yields bytes as if it was a file"""

    def __init__(self, iter: Iterator[bytes]):
        self._iter = iter
        self._left = b""

    def __iter__(self):
        assert not self._left, "Cannot iterate over a stream that has already been read"
        return iter(self._iter)

    def __next__(self):
        assert not self._left, "Cannot iterate over a stream that has already been read"
        return next(self._iter)

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


class AsyncBinaryIteratorIO(AsyncBinaryIO, AsyncBinaryIterator, AsyncIterator[bytes]):
    """Similar to BinaryIteratorIO except the iterator yields bytes asynchronously"""

    def __init__(self, iter: AsyncBinaryIterator):
        self._iter = iter
        self._left = b""

    def __aiter__(self):
        assert not self._left, "Cannot iterate over a stream that has already been read"
        return self._iter.__aiter__()

    def __anext__(self):
        assert not self._left, "Cannot iterate over a stream that has already been read"
        return self._iter.__anext__()

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
                m = await self._read1()
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
    media_type = "application/octet-stream"
    """
    May override in subclasses to derive correct stream type, e.g.,
    "application/x-cryosparc-dataset"
    """

    @classmethod
    def api_schema(cls):
        """
        Schema to use when an API endpoint returns or requests this streamable
        instance in the request or response body.
        """
        return {
            "description": f"A binary stream representing a {cls.__name__} class instance",
            "content": {cls.media_type: {"schema": {"title": cls.__name__, "type": "string", "format": "binary"}}},
        }

    @classmethod
    @abstractmethod
    def load(cls, file: Union[str, PurePath, IO[bytes]], *, media_type: Optional[str] = None) -> "Self":
        """
        Load stream from a file path or readable byte stream. The stream must
        at least implement the `read(size)` function.
        """
        ...

    @classmethod
    def from_iterator(cls, source: Iterator[bytes], *, media_type: Optional[str] = None):
        return cls.load(BinaryIteratorIO(source), media_type=media_type)

    @classmethod
    @abstractmethod
    async def from_async_stream(cls, stream: AsyncBinaryIO, *, media_type: Optional[str] = None) -> "Self":
        """
        Asynchronously load from the given binary stream. The given stream
        parameter must at least have ``async read(n: int | None) -> bytes`` method.
        """
        ...

    @classmethod
    async def from_async_iterator(cls, iterator: AsyncBinaryIterator, *, media_type: Optional[str] = None):
        return await cls.from_async_stream(AsyncBinaryIteratorIO(iterator), media_type=media_type)

    @abstractmethod
    def stream(self) -> Iterator[bytes]: ...

    async def astream(self) -> AsyncIterator[bytes]:
        for chunk in self.stream():
            yield chunk

    def save(self, file: Union[str, PurePath, IO[bytes]]):
        with bopen(file, "wb") as f:
            self.dump(f)

    def dump(self, file: IO[bytes]):
        for chunk in self.stream():
            file.write(chunk)

    def dumps(self) -> bytes:
        return b"".join(self.stream())


class Stream(Streamable):
    """
    Generic stream that that leaves handling of the stream data to the caller.
    May accept stream data in any streamable format, though async formats
    must be consumed with async functions.
    """

    @overload
    def __init__(self, *, stream: IO[bytes] = ..., media_type: Optional[str] = ...): ...
    @overload
    def __init__(self, *, iterator: Iterator[bytes] = ..., media_type: Optional[str] = ...): ...
    @overload
    def __init__(self, *, astream: AsyncBinaryIO = ..., media_type: Optional[str] = ...): ...
    @overload
    def __init__(self, *, aiterator: AsyncBinaryIterator = ..., media_type: Optional[str] = ...): ...
    def __init__(
        self,
        *,
        stream: Optional[IO[bytes]] = None,
        iterator: Optional[Iterator[bytes]] = None,
        astream: Optional[AsyncBinaryIO] = None,
        aiterator: Optional[AsyncBinaryIterator] = None,
        media_type: Optional[str] = None,
    ):
        if (stream is not None) + (iterator is not None) + (astream is not None) + (aiterator is not None) != 1:
            raise TypeError("Exactly one of stream, iterator, astream or aiterator must be provided")
        self._stream = stream
        self._iterator = iterator
        self._astream = astream
        self._aiterator = aiterator
        self.media_type = media_type or self.media_type

    @property
    def asynchronous(self):
        return (self._astream is not None) or (self._aiterator is not None)

    @classmethod
    def load(cls, file: Union[str, PurePath, IO[bytes]], *, media_type: Optional[str] = None):
        stream = open(file, "rb") if isinstance(file, (str, PurePath)) else file
        return cls(stream=stream, media_type=media_type)

    @classmethod
    def from_iterator(cls, source: Iterator[bytes], *, media_type: Optional[str] = None):
        return cls(iterator=source, media_type=media_type)

    @classmethod
    async def from_async_stream(cls, stream: AsyncBinaryIO, *, media_type: Optional[str] = None):
        return cls(astream=stream, media_type=media_type)

    @classmethod
    async def from_async_iterator(cls, iterator: AsyncBinaryIterator, *, media_type: Optional[str] = None):
        return cls(aiterator=iterator, media_type=media_type)

    def stream(self) -> Iterator[bytes]:
        if self._stream:
            while chunk := self._stream.read(EIGHT_MIB):
                yield chunk
        elif self._iterator:
            for chunk in self._iterator:
                yield chunk
        else:
            raise TypeError("This is an asynchronous stream, must use astream() instead")

    async def astream(self) -> AsyncIterator[bytes]:
        if self._stream or self._iterator:
            for chunk in self.stream():
                yield chunk
        elif self._astream:
            while chunk := await self._astream.read(EIGHT_MIB):
                yield chunk
        elif self._aiterator:
            async for chunk in self._aiterator:
                yield chunk
