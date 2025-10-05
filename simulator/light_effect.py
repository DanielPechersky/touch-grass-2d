from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import implot_draw_circle, ndarray_to_scatter_many
from simulator.persistence import Chain


@dataclass
class CattailContext:
    centers: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]
    accelerations: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]


class LightEffect(Protocol):
    def calculate_chain_brightness(
        self,
        delta_time: float,
        chains: list[Chain[Any]],
        cattail_context: CattailContext,
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """
        Returns an ndarray of shape (len(chains),) with values in [0, 1].
        """
        raise NotImplementedError

    def debug_gui(self):
        """
        Optional GUI for debugging purposes.
        Implement this method if you want to visualize the internal state of a light effect.
        """
        pass


class TestLightEffect(LightEffect):
    def __init__(self):
        self.accumulated_time = 0.0

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
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

        self.exhausted: np.ndarray[tuple[int], np.dtype[np.bool]] = np.zeros(
            (0,), dtype=bool
        )
        self.projectile_start_times: np.ndarray[tuple[int], np.dtype[np.floating]] = (
            np.zeros((0,), dtype=np.float32)
        )
        self.projectile_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros((0, 2), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
        self.projectile_velocities: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros((0, 2), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
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
        new_positions: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
        new_velocities: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
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
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        chain_points: np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.floating]] = (
            np.stack([chain.points for chain in chains], dtype=np.float32)
        )

        for projectile_pos in self.projectile_positions:
            distances = np.linalg.vector_norm(
                chain_points - projectile_pos, axis=2
            ).min(axis=1)
            brightness += np.maximum(0.0, 1.0 - distances * 1.0)

        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    def debug_gui(self):
        implot.set_next_marker_style(size=5, fill=imgui.ImVec4(0, 0, 255, 1))
        implot.plot_scatter(
            "projectiles", *ndarray_to_scatter_many(self.projectile_positions)
        )


class PulseLightEffect(LightEffect):
    def __init__(self, upper_threshold=0.1, lower_threshold=0.03):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.accumulated_time = 0.0

        self.exhausted: np.ndarray[tuple[int], np.dtype[np.bool]] = np.zeros(
            (0,), dtype=bool
        )
        self.projectile_start_times: np.ndarray[tuple[int], np.dtype[np.floating]] = (
            np.zeros((0,), dtype=np.float32)
        )
        self.projectile_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros((0, 2), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
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

        projectile_centers = cattail_context.centers[should_shoot]

        self.expire_projectiles()
        self.add_new_projectiles(projectile_centers)

        return self.chain_brightness_from_projectiles(chains)

    def add_new_projectiles(
        self,
        new_positions: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
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

    def expire_projectiles(self):
        alive_mask = (self.accumulated_time - self.projectile_start_times) < 5.0
        self.projectile_start_times = self.projectile_start_times[alive_mask]
        self.projectile_positions = self.projectile_positions[alive_mask]

    def chain_brightness_from_projectiles(
        self,
        chains: list[Chain],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        chain_points = np.stack([chain.points for chain in chains], dtype=np.float32)
        chain_centers = np.mean(chain_points, axis=1)

        distances = distances_from_circles(
            chain_centers,
            self.projectile_positions,
            self.pulse_sizes,
        )
        brightness += np.maximum(
            1.0 - distances * 1.0,
            0,
        ).sum(axis=1)
        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    @property
    def pulse_sizes(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return (self.accumulated_time - self.projectile_start_times) * 3.0

    def debug_gui(self):
        for pulse_position, pulse_size in zip(
            self.projectile_positions, self.pulse_sizes, strict=True
        ):
            implot_draw_circle(
                center=pulse_position, radius=pulse_size.item(), col=0x33FF0000
            )


class PulseLightEffect2(LightEffect):
    def __init__(self, upper_threshold=0.1, lower_threshold=0.03):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.accumulated_time = 0.0

        self.exhausted: np.ndarray[tuple[int], np.dtype[np.bool]] = np.zeros(
            (0,), dtype=bool
        )
        self.projectile_start_times: np.ndarray[tuple[int], np.dtype[np.floating]] = (
            np.zeros((0,), dtype=np.float32)
        )
        self.projectile_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros((0, 2), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
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

        projectile_centers = cattail_context.centers[should_shoot]

        self.expire_projectiles()
        self.add_new_projectiles(projectile_centers)

        return self.chain_brightness_from_projectiles(chains)

    def add_new_projectiles(
        self,
        new_positions: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
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

    def expire_projectiles(self):
        alive_mask = (self.accumulated_time - self.projectile_start_times) < 5.0
        self.projectile_start_times = self.projectile_start_times[alive_mask]
        self.projectile_positions = self.projectile_positions[alive_mask]

    def chain_brightness_from_projectiles(
        self,
        chains: list[Chain],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        chain_points = np.stack([chain.points for chain in chains], dtype=np.float32)
        chain_centers = np.mean(chain_points, axis=1)

        distances = distances_from_circles(
            chain_centers,
            self.projectile_positions,
            self.pulse_sizes,
        )
        projectile_travel_times = self.accumulated_time - self.projectile_start_times
        brightness += np.maximum(
            1.0
            - distances * 2.0
            - np.maximum((projectile_travel_times - 2.0) * 0.1, 0.0),
            0,
        ).sum(axis=1)
        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    @property
    def pulse_sizes(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return (self.accumulated_time - self.projectile_start_times) * 0.6 + 0.7

    def debug_gui(self):
        for pulse_position, pulse_size in zip(
            self.projectile_positions, self.pulse_sizes, strict=True
        ):
            implot_draw_circle(
                center=pulse_position, radius=pulse_size.item(), col=0x33FF0000
            )


def distances_from_circle(
    points: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray | np.floating,
) -> np.ndarray:
    distance_point_from_center = np.linalg.vector_norm(points - center, axis=-1)
    return np.abs(distance_point_from_center - radius)


def distances_from_circles(
    points: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    return distances_from_circle(
        points[:, np.newaxis, :], centers[np.newaxis, :, :], radii
    )
