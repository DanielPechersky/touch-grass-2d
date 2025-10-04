from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import ndarray_to_scatter_many
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

    def debug_gui(self):
        pass


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


class ProjectileLightEffect(LightEffect):
    def __init__(self, upper_threshold=0.1, lower_threshold=0.03):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.accumulated_time = 0.0

        self.exhausted: np.ndarray[tuple[int], np.dtypes.BoolDType] = np.zeros(
            (0,), dtype=bool
        )
        self.projectile_start_times: np.ndarray[tuple[int], np.dtypes.Float32DType] = (
            np.zeros((0,), dtype=np.float32)
        )
        self.projectile_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtypes.Float32DType
        ] = np.zeros((0, 2), dtype=np.float32)
        self.projectile_velocities: np.ndarray[
            tuple[int, Literal[2]], np.dtypes.Float32DType
        ] = np.zeros((0, 2), dtype=np.float32)

    def calculate_chain_brightness(
        self,
        delta_time: float,
        chains: list[Chain],
        cattail_context: CattailContext,
    ) -> np.ndarray[tuple[ChainIdx]]:
        self.accumulated_time += delta_time

        if len(self.exhausted) != len(cattail_context.centers):
            self.exhausted = np.zeros((len(cattail_context.centers),), dtype=bool)

        acceleration_magnitudes = np.linalg.vector_norm(
            cattail_context.accelerations, axis=1
        )

        below_lower_threshold = acceleration_magnitudes < self.lower_threshold
        above_upper_threshold = acceleration_magnitudes > self.upper_threshold

        should_shoot = above_upper_threshold & ~self.exhausted

        self.exhausted[above_upper_threshold] = True
        self.exhausted[below_lower_threshold] = False

        relevant_accelerations = cattail_context.accelerations[should_shoot]
        projectile_velocities = (
            relevant_accelerations / np.linalg.vector_norm(relevant_accelerations) * 1.0
        )
        projectile_centers = cattail_context.centers[should_shoot]

        self.expire_projectiles()
        self.move_projectiles(delta_time)
        self.add_new_projectiles(projectile_centers, projectile_velocities)

        return self.chain_brightness_from_projectiles(chains)

    def add_new_projectiles(
        self,
        new_positions: np.ndarray[tuple[int, Literal[2]]],
        new_velocities: np.ndarray[tuple[int, Literal[2]]],
    ):
        self.projectile_start_times = np.concatenate(
            (
                self.projectile_start_times,
                np.full((new_positions.shape[0],), self.accumulated_time),
            )
        )
        self.projectile_positions = np.concatenate(
            (self.projectile_positions, new_positions)
        )
        self.projectile_velocities = np.concatenate(
            (self.projectile_velocities, new_velocities)
        )

    def move_projectiles(self, delta_time: float):
        self.projectile_positions += self.projectile_velocities * delta_time

    def expire_projectiles(self):
        alive_mask = (self.accumulated_time - self.projectile_start_times) < 5.0
        self.projectile_start_times = self.projectile_start_times[alive_mask]
        self.projectile_positions = self.projectile_positions[alive_mask]
        self.projectile_velocities = self.projectile_velocities[alive_mask]

    def chain_brightness_from_projectiles(
        self,
        chains: list[Chain],
    ) -> np.ndarray[tuple[ChainIdx]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        for projectile_pos in self.projectile_positions:
            for i, chain in enumerate(chains):
                distance = np.linalg.vector_norm(
                    chain.points - projectile_pos, axis=1
                ).min()
                brightness[i] += max(0.0, 1.0 - distance * 1.0)

        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    def debug_gui(self):
        implot.set_next_marker_style(size=5, fill=imgui.ImVec4(0, 0, 255, 1))
        implot.plot_scatter(
            "projectiles", *ndarray_to_scatter_many(self.projectile_positions)
        )
