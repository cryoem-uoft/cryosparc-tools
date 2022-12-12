from typing import Any, Dict, Generic, Iterable, List, Mapping, Optional, Tuple, TypeVar

import numpy as n

from .column import Column
from .util import default_rng, random_integers


class Row(Mapping):
    """
    Provides row-by-row access to a dataset. Do not initialize directly. See
    dataset module.
    """

    __slots__ = ("idx", "cols", "__dict__")  # Specifying this speeds up allocation of many rows
    idx: int
    cols: Dict[str, Column]

    def __init__(self, cols: Dict[str, Column], idx: int):
        self.idx = idx
        self.cols = cols
        # note - don't keep around a ref to cols because then when col._data
        # changes (e.g., a field is added to the dataset) the already existing
        # items will be referring to the old dataset._data!

    def __array__(self, dtype=None):
        # Prevent numpy from attempting to turn row to array of strings
        a = n.ndarray(shape=(), dtype=n.dtype(object))
        a[()] = self
        return a

    def __len__(self):
        return len(self.cols)

    def __getitem__(self, key: str):
        return self.cols[key][self.idx]

    def __setitem__(self, key: str, value):
        self.cols[key][self.idx] = value

    def __contains__(self, key: object):
        return key in self.cols

    def __iter__(self):
        return iter(self.cols)

    def item(self, key, default=None):
        return self.cols[key].item(self.idx) if key in self else default

    def to_list(self, exclude_uid=False):
        """
        Convert into a list of native python types in the same order as the
        declared fields.
        """
        return [self.cols[key].item(self.idx) for key in self.cols if not exclude_uid or key != "uid"]

    def to_dict(self):
        return {key: self.cols[key].item(self.idx) for key in self.cols}

    def from_dict(self, d):
        for k in self.cols:
            self[k] = d[k]

    def __repr__(self) -> str:
        result = f"{type(self).__name__}(["
        for k in self:
            val = self[k]
            result += f"\n    ('{k}', {val if isinstance(val, n.ndarray) else repr(val)}),"
        return result + "\n])"

    def _ipython_key_completions_(self):
        return list(self.cols)


R = TypeVar("R", bound=Row)
"""
Type variable for a ``Row`` subclass.
"""


class Spool(List[R], Generic[R]):
    """
    List-like dataset row accessor class with support for splitting and
    randomizing based on row fields

    Args:
        items (Iterable[R]): List of rows
        rng (Generator, optional): Numpy random number generator. Uses
            ``numpy.random.default_rng()`` if not specified. Defaults to None.
    """

    DEFAULT_RNG = default_rng()

    @classmethod
    def set_default_random(cls, rng: "n.random.Generator"):
        """
        Reset the default random number generator for all Spools

        Args:
            rng (Generator): Numpy random generator.
        """
        cls.DEFAULT_RNG = rng

    def __init__(self, items: Iterable[R], rng: "Optional[n.random.Generator]" = None):
        super().__init__(items)
        self.indices = None
        self.random = rng or self.DEFAULT_RNG

    def set_random(self, rng: "n.random.Generator"):
        """
        Reset the random number generator for this Spool.

        Args:
            rng (Generator): Numpy random generator.
        """
        self.random = rng

    # -------------------------------------------------- Spooling and Splitting
    def split(self, num: int, random: bool = True, prefix: Optional[str] = None):
        """
        Return two Spools with the elements of this spool. The first Spool
        contains ``num`` elements. The second contains ``len(self) - num``
        elements.

        Args:
            num (int): Number of elements in first split.
            random (bool, optional): If True, add elements to each split in
                random order. Defaults to True.
            prefix (str, optional): If specified, set each ``{prefix}/split``
                field to ``0`` in the first split and ``1`` in the second split.
                Defaults to None.

        Returns:
            tuple[Spool, Spool]: the two split lists
        """
        if random:
            idxs = self.random.permutation(len(self))
        else:
            idxs = n.arange(len(self))
        d1 = Spool(items=(self[i] for i in idxs[:num]), rng=self.random)
        d2 = Spool(items=(self[i] for i in idxs[num:]), rng=self.random)
        if prefix is not None:
            field = prefix + "/split"
            for img in d1:
                img[field] = 0
            for img in d2:
                img[field] = 1
        return d1, d2

    def split_half_in_order(self, prefix: str, random: bool = True):
        """
        Split into two spools of approximately equal size. Elements are added
        to each split in stable order.

        Args:
            prefix (str): Set each element's ``{prefix}/split`` field to ``0``
                or ``1`` depending on which split it's added to.
            random (bool, optional): If True, randomly assign each element to a
                split. Defaults to True.

        Returns:
            tuple[Spool, Spool]: the two split lists
        """
        if random:
            splitvals = random_integers(self.random, 2, size=len(self))
        else:
            splitvals = n.arange(len(self)) % 2
        for idx, p in enumerate(self):
            p[prefix + "/split"] = splitvals[idx]
        d1 = Spool(items=(p for p in self if p[prefix + "/split"] == 0), rng=self.random)
        d2 = Spool(items=(p for p in self if p[prefix + "/split"] == 1), rng=self.random)
        return d1, d2

    def split_into_quarter(self, num: int):
        """
        Randomly assign the elements of this Spool to two new Spools. The first
        Spool contains ``num`` elements. The second contains ``len(self) - num``
        elements.

        Args:
            num (int): Number of elements in the first split.

        Returns:
            tuple[Spool, Spool]: the two split lists
        """
        return self.split(num, random=True)

    def split_by_splits(self, prefix: str = "alignments"):
        """
        Return two Spools divided by the value of the ``{prefix}/split`` field.

        Args:
            prefix (str, optional): Field to prefix to use to determine which
                element goes into which split list. Defaults to "alignments".

        Returns:
            tuple[Spool, Spool]: the two split lists
        """
        idxs = n.arange(len(self))
        field = "split"
        if prefix is not None:
            field = prefix + "/split"
        d1 = Spool(items=(self[i] for i in idxs if self[i][field] == 0), rng=self.random)
        d2 = Spool(items=(self[i] for i in idxs if self[i][field] == 1), rng=self.random)
        return d1, d2

    def split_from_field(self, field: str, vals: Tuple[Any, Any] = (0, 1)):
        """
        Split into two lists based on the given possible vals of the given
        field.

        Args:
            field (str): Split field to test for split value.
            vals (tuple[any, any], optional): Possible split field values.
                Defaults to (0, 1).

        Returns:
            tuple[Spool, Spool]: the two split lists
        """
        d1 = Spool((img for img in self if img[field] == vals[0]), rng=self.random)
        d2 = Spool((img for img in self if img[field] == vals[1]), rng=self.random)
        return d1, d2

    def split_by(self, field: str) -> Dict[Any, List[R]]:
        """
        Split into a dictionary of lists, where each key is a possible value
        for the given field and each value is a list if elements that have
        that value.

        Args:
            field (str): dataset field to split on

        Returns:
            dict[any, list[R]]: dict of split element lists
        """
        items: Dict[Any, List[R]] = {}
        for item in self:
            val = item[field]
            curr = items.get(val, [])
            curr.append(item)
            items[val] = curr
        return items

    def get_random_subset(self, num: int):
        """
        Randomly selected subset of the given size, without replacement.

        Args:
            num (int): number of elements to select from the Spool

        Returns:
            list[R]: selected elements
        """
        assert num <= len(self), "Not Enough Images!"
        idxs = self.random.choice(len(self), size=num, replace=False)
        return [self[i] for i in idxs]

    def setup_spooling(self, random=True):
        """
        Determine the iteration order for the ``spool()`` method.

        Args:
            random (bool, optional): Randomize spooling order. Defaults to True.
        """
        if random:
            self.indices = self.random.permutation(len(self))
        else:
            self.indices = n.arange(len(self))
        self.spool_index = 0

    def spool(self, num: int, peek: bool = False):
        """
        Get a list consisting of num randomly-selected elements. Advance the
        spool. If peek is true, don't advance the spool, just return the first
        num elements.

        Args:
            num (int): Number of elements to get from the spool.
            peek (bool, optional): If True, does not advance internal spool
                index. Will return the same elements on the next ``spool()``
                call. Defaults to False.

        Returns:
            list[R]: list of selected spool elements
        """
        if self.indices is None:
            self.setup_spooling()
        assert self.indices is not None
        if num >= len(self):  # asking for too many
            return [self[i] for i in range(len(self))]  # just return self, no random order.
        current_indices = n.arange(self.spool_index, self.spool_index + num) % len(self.indices)
        if not peek:
            self.spool_index = (self.spool_index + num) % len(self.indices)
        return [self[self.indices[i]] for i in current_indices]

    def make_batches(self, num: int = 200):
        """
        Get a list of lists, each one a consecutive list of ``num`` images.

        Args:
            num (int, optional): Size of each batch. Defaults to 200.

        Returns:
            list[list[R]]: Split batches
        """
        return [self[idx : idx + num] for idx in n.arange(0, len(self), num)]

    def __str__(self):
        s = f"{type(self).__name__} object with {len(self)} items."
        return s
