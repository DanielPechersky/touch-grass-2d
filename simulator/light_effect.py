from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from simulator.persistence import Chain
from simulator.typing import ChainIdx


@dataclass
class CattailContext:
    centers: np.ndarray[tuple[int, Literal[2]]]
    accelerations: np.ndarray[tuple[int, Literal[2]]]


class LightEffect(Protocol):
    def calculate_chain_brightness(
        self,
        delta_time: float,
        chains: list[Chain],
        cattail_context: CattailContext,
    ) -> np.ndarray[tuple[ChainIdx]]:
        raise NotImplementedError


class TestLightEffect(LightEffect):
    def __init__(self):
        self.accumulated_time = 0.0

    def calculate_chain_brightness(
        self,
        delta_time: float,
        chains: list[Chain],
        cattail_context: CattailContext,
    ) -> np.ndarray[tuple[ChainIdx]]:
        self.accumulated_time += delta_time

        total_acceleration = np.linalg.vector_norm(
            cattail_context.accelerations, axis=1
        ).sum()

        num_chains = len(chains)

        time_brightness = (
            np.sin(
                np.linspace(0.0, num_chains, num_chains, dtype=np.float32)
                + self.accumulated_time * 0.6
            )
            * 0.5
            + 0.5
        )

        acceleration_brightness = min(total_acceleration * 30, 1)

        return (time_brightness + acceleration_brightness) / 2
