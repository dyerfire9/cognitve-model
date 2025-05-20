from typing import Tuple, List, Sequence
import re

from ..base import feature, chunk
from .. import numdicts as nd
from .. import dev as cld


__all__ = ["Flags", "Slots"]


class Flags(cld.Process):

    initial = nd.NumDict()
    set_prefix = "set"

    def __init__(self, fs: Sequence[str], vs: Sequence[int] = (-1, 0, 1)) -> None:
        for f in fs:
            if not cld.ispath(f):
                raise ValueError(f"Flag name '{f}' is not a valid path.")
            if f.startswith(f"{self.set_prefix}"):
                raise ValueError("Flag name starts with reserved prefix "
                    f"'{self.set_prefix}'")

        self.fs = tuple(fs)
        self.vs = (None, *vs) # type: ignore
        self.store = nd.NumDict(c=0)

    def call(self, c: nd.NumDict[feature]) -> nd.NumDict[feature]:
        self.update(c)
        return self.store

    def update(self, c: nd.NumDict) -> None:
        self.store = (self.store
            .mul(c
                .keep(sf=lambda f: f.v is None)
                .transform_keys(kf=self.cmd2flag)
                .mask())
            .squeeze()
            .merge(c
                .keep(sf=lambda f: f.v == 1)
                .transform_keys(kf=self.cmd2flag)
                .mask())
            .merge(c
                .keep(sf=lambda f: f.v == -1)
                .transform_keys(kf=self.cmd2flag)
                .mask()
                .mul(-1)))

    def cmd2flag(self, f_cmd):
        l, sep, r = f_cmd.d.partition(cld.FSEP)
        d_cmd = l if not sep else r
        flag = re.sub("^set-", "", d_cmd)
        f = feature(cld.prefix(flag, self.prefix))
        assert f in self.flags, f"regexp sub likely failed: '{f}'"
        return f

    @property
    def flags(self) -> Tuple[feature, ...]:
        return tuple(feature(dim) for dim in cld.prefix(self.fs, self.prefix))

    @property
    def cmds(self):
        dims = ["-".join([self.set_prefix, f]) for f in self.fs]
        dims = cld.prefix(dims, self.prefix)
        return tuple(feature(dim, v) for dim in dims for v in self.vs)

    @property
    def nops(self):
        dims = ["-".join([self.set_prefix, f]) for f in self.fs]
        dims = cld.prefix(dims, self.prefix)
        return tuple(feature(dim, None) for dim in dims)


class Slots(cld.Process):

    initial = (nd.NumDict(), nd.NumDict())
    store: nd.NumDict[Tuple[int, chunk]]

    def __init__(self, slots: int) -> None:
        self.slots = slots
        self.store = nd.NumDict() # (slot, chunk): 1.0, c=0

    def call(
        self, 
        c: nd.NumDict[feature], 
        s: nd.NumDict[chunk], 
        m: nd.NumDict[chunk]
    ) -> Tuple[nd.NumDict[chunk], nd.NumDict[feature]]:
        """
        c: commands
        s: selected chunk
        m: match strengths
        """

        self.update(c, s)

        rd = (c # rd indicates for each slot if its contents should be read
            .keep(sf=lambda k: k.d not in self._write_dims() and k.v == 1)
            .transform_keys(kf=self._cmd2slot))
        chunks = (self.store
            .put(rd, kf=cld.first)
            .squeeze()
            .max_by(kf=cld.second))

        full = (self.store
            .abs()
            .sum_by(kf=cld.first)
            .greater(0)
            .mul(2)
            .sub(1)
            .with_keys(ks=range(1, self.slots + 1))
            .set_c(0)
            .transform_keys(kf=self._full_flag))
        match = (self.store
            .put(m, kf=cld.second)
            .cam_by(kf=cld.first)
            .squeeze()
            .transform_keys(kf=self._match_flag))
        flags = full + match

        return chunks, flags

    def update(self, c: nd.NumDict, s: nd.NumDict) -> None:
        ud = (c
            .keep(sf=lambda k: k.d in self._write_dims() and k.v != 0)
            .transform_keys(kf=self._cmd2slot))
        wrt = (c
            .keep(sf=lambda k: k.d in self._write_dims() and k.v == 1)
            .transform_keys(kf=self._cmd2slot))
        self.store = (wrt
            .outer(s)
            .merge(self.store
                .put(1 - ud, kf=lambda k: k[0])
                .squeeze()))

    def _write_dims(self) -> List[str]:
        return cld.prefix([f"write-{i + 1}" for i in range(self.slots)], 
            self.prefix)
    
    def _full_flag(self, i: int) -> feature:
        return feature(cld.prefix(f"full-{i}", self.prefix))
    
    def _match_flag(self, i: int) -> feature:
        return feature(cld.prefix(f"match-{i}", self.prefix))

    def _cmd2slot(self, cmd: feature) -> int:
        dim = cmd.d
        if self.prefix:
            dim = cld.split(dim).fragment
        return int(dim.split("-")[-1])
        
    @property
    def flags(self) -> Tuple[feature, ...]:
        tup = cld.prefix(("full", "match"), self.prefix) 
        return tuple(feature(f"{k}-{i + 1}") 
            for k in tup for i in range(self.slots))

    @property
    def cmds(self) -> Tuple[feature, ...]:
        d: dict[str, Tuple[int, ...]] = {}
        for i in range(self.slots):
            d[f"read-{i + 1}"] = (0, 1)
            d[f"write-{i + 1}"] = (-1, 0, +1)
        return tuple(feature(k, v) 
            for k, vs in cld.prefix(d, self.prefix).items() for v in vs)

    @property
    def nops(self) -> Tuple[feature, ...]:
        d: dict[str, int] = {}
        for i in range(self.slots):
            d[f"read-{i + 1}"] = 0
            d[f"write-{i + 1}"] = 0
        return tuple(feature(k, v) 
            for k, v in cld.prefix(d, self.prefix).items())
