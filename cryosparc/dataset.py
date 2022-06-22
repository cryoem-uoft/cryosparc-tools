from typing import Any, List, MutableMapping, Sequence, Union
import numpy as n


class Column(Sequence):
    """
    Dataset column entry
    """

    def __init__(self, data, field: str, start: int = 0, stop: int = -1):
        # dlen = len(data[field])  # FIXME: Get the data from the given field somehow
        dlen = data
        assert -dlen <= start and start < dlen, f"Invalid start index {start}"
        assert -dlen <= stop and stop <= dlen, f"Invalid stop index {stop}"
        self.data = data
        self.field = field
        self.start = start + dlen if start < 0 else start
        self.stop = stop + dlen + 1 if stop < 0 else stop
        assert self.start <= self.stop, "Cannot initialize a column in reverse order"

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        for i in range(self.start, self.stop):
            yield i

    def __getitem__(self, key: Union[int, slice]):
        clen = len(self)
        if isinstance(key, slice):
            assert key.step is None, "Column does not support slices with step"
            start = key.start or 0
            stop = key.stop or clen
            assert -clen <= start and start < clen, f"Invalid start index {start}"
            assert -clen <= stop and stop <= clen, f"Invalid stop index {stop}"
            start = start + clen if start < 0 else start
            stop = stop + clen + 1 if stop < 0 else stop
            return self.__class__(self.data, self.field, start=self.start + start, stop=self.start + stop)
        else:
            assert -clen <= key and key < clen, f"Invalid col index {key} for column with length {clen}"
            return self.start + key % clen

    def __setitem__(self, key: int, value: Any):
        print(key, value)
        # assert isinstance(key, int) and key >= 0 and key < len(self), f"Invalid column index {key}, must be in range [0, {len(self)}))"

    def as_ndarray(self, copy=False):
        """
        Return the underlying data as a numpy ndarray. Generally faster than
        converting via iteration because it uses the underlying bytes directly
        (except for cases where strings must be converted to objects)

        Underlying data is read-only by default, unless copy is set to `True`.
        """
        pass


class Dataset(MutableMapping):
    """
    Accessor class for working with cryoSPARC .cs files.

    Example usage

    ```
    dset = Dataset.load('/path/to/particles.cs')
    for particle in dset:
        print(f"Particle located in file {particle['blob/path']} at index {particle['blob/idx']}")
    ```

    """

    def __len__(self):
        return 0

    def __iter__(self):
        """
        Iterate over available fields in the dataset
        """
        pass

    def __getitem__(self, key: str) -> Column:
        """
        Get a specific field
        """
        pass

    def __setitem__(self, key: str, val: Union[list, n.ndarray, Column]):
        """
        Set or add a field to the dataset. If the field already exists, enforces
        that the data type is the same.
        """
        pass

    def __delitem__(self, key: str):
        """
        Removes field from the dataset
        """
        pass

    def fields(self) -> List[str]:
        pass

    def add_fields(self):
        pass
