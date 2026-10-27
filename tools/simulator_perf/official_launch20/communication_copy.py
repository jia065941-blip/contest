"""Copy-on-read communication dictionaries for the frozen evaluation pipeline.

Each owner has a distinct mutable dict. Unread subtrees use private immutable
snapshots. Retrieving a subtree materializes a local mutable child, so nested
writes and later deepcopies have ordinary dict isolation. No group membership,
geometry, refresh timing, fusion, event ordering or actor logic is replaced.
"""
from collections.abc import ItemsView, ValuesView
import copy


class _Snapshot(dict):
    """Private immutable tree; never returned through the public dict API."""
    __slots__ = ()

    @classmethod
    def freeze(cls, source):
        if type(source) is not dict:
            raise TypeError('Communication snapshot requires exact dictionaries')
        result = dict.__new__(cls)
        for key, value in source.items():
            if type(key) not in (int, str):
                raise TypeError('Unsupported communication key')
            if type(value) is dict:
                value = cls.freeze(value)
            elif type(value) not in (int, float, str, bool, type(None)):
                raise TypeError('Unsupported communication value')
            dict.__setitem__(result, key, value)
        return result

    def _immutable(self, *args, **kwargs):
        raise TypeError('Private communication snapshot is immutable')

    __init__ = __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable


class CommunicationDict(dict):
    __slots__ = ()

    @classmethod
    def from_plain(cls, source):
        snapshot = _Snapshot.freeze(source)
        return cls._from_snapshot(snapshot)

    @classmethod
    def _from_snapshot(cls, snapshot):
        result = dict.__new__(cls)
        # Values are immutable snapshot subtrees or immutable scalar leaves.
        dict.update(result, snapshot)
        return result

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if type(value) is _Snapshot:
            value = type(self)._from_snapshot(value)
            dict.__setitem__(self, key, value)
        return value

    def __iter__(self):
        # Force mapping-copy consumers such as dict(self) through __getitem__.
        return dict.__iter__(self)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        return ItemsView(self)

    def values(self):
        return ValuesView(self)

    def copy(self):
        return {key: self[key] for key in self}

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo):
        cached = memo.get(id(self))
        if cached is not None:
            return cached
        result = dict.__new__(type(self))
        memo[id(self)] = result
        for key, value in dict.items(self):
            # Private snapshots cannot be mutated through any public accessor.
            # Materialized/user-supplied values retain full Python copy semantics.
            copied = value if type(value) is _Snapshot else copy.deepcopy(value, memo)
            dict.__setitem__(result, copy.deepcopy(key, memo), copied)
        return result

    def setdefault(self, key, default=None):
        if key not in self:
            dict.__setitem__(self, key, default)
        return self[key]

    def pop(self, key, *default):
        if len(default) > 1:
            raise TypeError('pop expected at most 2 arguments')
        if key in self:
            result = self[key]
            dict.__delitem__(self, key)
            return result
        if default:
            return default[0]
        raise KeyError(key)

    def popitem(self):
        if not self:
            raise KeyError('popitem(): dictionary is empty')
        key = next(reversed(self))
        return key, self.pop(key)

    def __or__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        return self.copy() | dict(other)

    def __ror__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        return dict(other) | self.copy()


def install():
    from envengine.engine.engine import Engine
    original = Engine._get_common_range_info
    if getattr(original, '_communication_snapshot', False):
        raise RuntimeError('Communication snapshots already installed')

    def common_range(self, group):
        # Execute the native builder in its original order, then take ownership
        # of its scalar-only tree. Every native deepcopy still returns a distinct
        # mutable owner; only immutable unread subtrees share physical storage.
        return CommunicationDict.from_plain(original(self, group))

    common_range._communication_snapshot = True
    Engine._get_common_range_info = common_range
