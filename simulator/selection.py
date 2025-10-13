from dataclasses import dataclass
from typing import NewType

from simulator.persistence import Cattail, Chain


@dataclass(frozen=True)
class CattailId:
    id: int


@dataclass(frozen=True)
class ChainId:
    id: int


def id_for_item(item: Cattail[int] | Chain[int]) -> CattailId | ChainId:
    match item:
        case Cattail(id=id):
            return CattailId(id)
        case Chain(id=id):
            return ChainId(id)
        case _:
            raise ValueError(f"Unknown item type: {type(item)}")


Selection = NewType("Selection", set[CattailId | ChainId])
