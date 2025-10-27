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


class PulsePhysics:
    def __init__(self):
        self.accumulated_time = 0.0
        self.start_times: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )
        self.centers: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]] = (
            np.zeros((0, 2), dtype=np.float32)
        )  # type: ignore
        self.sizes: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )
        self.speeds: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )

    def update(self, delta_time: float, expire_older_than: float):
        self.accumulated_time += delta_time
        self.sizes += self.speeds * delta_time
        self.expire_pulses(expire_older_than)

    def add_pulses(
        self,
        new_centers: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
        new_sizes: np.ndarray[tuple[int], np.dtype[np.floating]],
        new_speeds: np.ndarray[tuple[int], np.dtype[np.floating]],
    ):
        self.start_times = np.concatenate(
            (
                self.start_times,
                np.full((new_centers.shape[0],), self.accumulated_time),
            )
        )
        self.centers = np.concatenate((self.centers, new_centers))
        self.sizes = np.concatenate((self.sizes, new_sizes))
        self.speeds = np.concatenate((self.speeds, new_speeds))

    def expire_pulses(self, older_than: float = 5.0):
        alive_mask = self.ages < older_than
        self.start_times = self.start_times[alive_mask]
        self.centers = self.centers[alive_mask]
        self.sizes = self.sizes[alive_mask]
        self.speeds = self.speeds[alive_mask]

    @property
    def ages(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self.accumulated_time - self.start_times

    def debug_gui(self):
        for position, size in zip(self.centers, self.sizes, strict=True):
            implot_draw_circle(center=position, radius=size.item(), col=0x33FF0000)


class ProjectilePhysics:
    def __init__(self):
        self.accumulated_time = 0.0
        self.start_times: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )
        self.positions: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]] = (
            np.zeros((0, 2), dtype=np.float32)
        )  # type: ignore
        self.velocities: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]] = (
            np.zeros((0, 2), dtype=np.float32)
        )  # type: ignore

    def update(self, delta_time: float, expire_older_than: float):
        self.accumulated_time += delta_time
        self.move_projectiles(delta_time)
        self.expire_projectiles(expire_older_than)

    def add_projectiles(
        self,
        new_positions: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
        new_velocities: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
    ):
        self.start_times = np.concatenate(
            (
                self.start_times,
                np.full((new_positions.shape[0],), self.accumulated_time),
            )
        )
        self.positions = np.concatenate((self.positions, new_positions))
        self.velocities = np.concatenate((self.velocities, new_velocities))

    def move_projectiles(self, delta_time: float):
        self.positions += self.velocities * delta_time

    def expire_projectiles(self, older_than: float):
        alive_mask = self.ages < older_than
        self.start_times = self.start_times[alive_mask]
        self.positions = self.positions[alive_mask]
        self.velocities = self.velocities[alive_mask]

    @property
    def ages(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return self.accumulated_time - self.start_times

    def debug_gui(self):
        implot.set_next_marker_style(size=5, fill=imgui.ImVec4(0, 0, 255, 1))
        implot.plot_scatter("projectiles", *ndarray_to_scatter_many(self.positions))


class PathPhysics:
    """
    Physics system where projectiles follow a single predefined path at fixed speeds.
    Projectiles can bounce a specified number of times before expiring.
    """

    def __init__(self, path: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]):
        self.path = path
        self.path_length = self._calculate_path_length()

        # Per-projectile data
        self.distances: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )
        self.speeds: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )
        self.directions: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            (0,), dtype=np.float32
        )  # 1.0 for forward, -1.0 for backward
        self.num_bounces: np.ndarray[tuple[int], np.dtype[np.int32]] = np.zeros(
            (0,), dtype=np.int32
        )  # Maximum number of bounces allowed
        self.bounce_count: np.ndarray[tuple[int], np.dtype[np.int32]] = np.zeros(
            (0,), dtype=np.int32
        )  # Current bounce count for each projectile

        # Computed positions (cached for efficiency)
        self.positions: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]] = (
            np.zeros((0, 2), dtype=np.float32)
        )  # type: ignore

    def add_projectiles(
        self,
        starting_distances: np.ndarray[tuple[int], np.dtype[np.floating]],
        speeds: np.ndarray[tuple[int], np.dtype[np.floating]],
        num_bounces: np.ndarray[tuple[int], np.dtype[np.int32]],
    ):
        """Add new projectiles on the path.

        Args:
            starting_distances: Starting distance along path for each projectile
            speeds: Speed for each projectile
            num_bounces: Maximum number of bounces for each projectile (0 = no bounces)
        """
        num_new = len(starting_distances)
        self.distances = np.concatenate((self.distances, starting_distances))
        self.speeds = np.concatenate((self.speeds, speeds))
        self.directions = np.concatenate(
            (self.directions, np.ones((num_new,), dtype=np.float32))
        )
        self.num_bounces = np.concatenate((self.num_bounces, num_bounces))
        self.bounce_count = np.concatenate(
            (self.bounce_count, np.zeros((num_new,), dtype=np.int32))
        )

    def update(self, delta_time: float):
        self.move_projectiles(delta_time)
        self.update_positions()
        self.expire_projectiles()

    def move_projectiles(self, delta_time: float):
        """Move projectiles along the path."""
        self.distances += self.speeds * self.directions * delta_time

        # Handle bouncing for each projectile
        for i in range(len(self.distances)):
            # Bounce logic: reverse direction when hitting either end
            while self.distances[i] < 0 or self.distances[i] > self.path_length:
                if self.distances[i] < 0:
                    self.distances[i] = -self.distances[i]
                    self.directions[i] *= -1
                    self.bounce_count[i] += 1
                elif self.distances[i] > self.path_length:
                    self.distances[i] = 2 * self.path_length - self.distances[i]
                    self.directions[i] *= -1
                    self.bounce_count[i] += 1

    def update_positions(self):
        """Calculate 2D positions from distances along the path."""
        if len(self.distances) == 0:
            self.positions = np.zeros((0, 2), dtype=np.float32)  # type: ignore
            return

        positions = []
        for i in range(len(self.distances)):
            position = self._interpolate_position_on_path(self.distances[i])
            positions.append(position)

        self.positions = np.array(positions, dtype=np.float32)

    def _calculate_path_length(self) -> float:
        """Calculate the total length of the path."""
        if len(self.path) < 2:
            return 0.0
        differences = np.diff(self.path, axis=0)
        segment_lengths = np.linalg.norm(differences, axis=1)
        return segment_lengths.sum()

    def _interpolate_position_on_path(self, distance: float) -> np.ndarray:
        """Get the 2D position at a given distance along the path."""
        if len(self.path) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        if distance <= 0:
            return self.path[0].astype(np.float32)

        # Calculate cumulative distances
        differences = np.diff(self.path, axis=0)
        segment_lengths = np.linalg.norm(differences, axis=1)
        cumulative_distances = np.concatenate(([0], np.cumsum(segment_lengths)))

        if distance >= self.path_length:
            return self.path[-1].astype(np.float32)

        # Find which segment we're on
        segment_idx = np.searchsorted(cumulative_distances, distance, side="right") - 1

        # Interpolate within the segment
        segment_start_dist = cumulative_distances[segment_idx]
        segment_length = segment_lengths[segment_idx]
        if segment_length == 0:
            return self.path[segment_idx].astype(np.float32)

        t = (distance - segment_start_dist) / segment_length

        return (
            self.path[segment_idx]
            + t * (self.path[segment_idx + 1] - self.path[segment_idx])
        ).astype(np.float32)

    def expire_projectiles(self):
        """Remove projectiles that have exhausted their bounces."""
        if len(self.distances) == 0:
            return

        # Start with all projectiles alive
        alive_mask = np.ones(len(self.distances), dtype=bool)

        # Expire projectiles that have exhausted bounces
        for i in range(len(self.distances)):
            if self.bounce_count[i] > self.num_bounces[i]:
                alive_mask[i] = False

        self.distances = self.distances[alive_mask]
        self.speeds = self.speeds[alive_mask]
        self.directions = self.directions[alive_mask]
        self.num_bounces = self.num_bounces[alive_mask]
        self.bounce_count = self.bounce_count[alive_mask]
        self.positions = self.positions[alive_mask]

    def debug_gui(self):
        # Draw path
        if len(self.path) > 0:
            implot.plot_line("path", *ndarray_to_scatter_many(self.path))

        # Draw projectiles
        if len(self.positions) > 0:
            implot.set_next_marker_style(size=7, fill=imgui.ImVec4(1, 0, 1, 1))
            implot.plot_scatter(
                "path_projectiles", *ndarray_to_scatter_many(self.positions)
            )


class AccelerationThresholdTrigger:
    def __init__(self, upper_threshold=0.6, lower_threshold=0.4):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        self.exhausted: np.ndarray[tuple[int], np.dtype[np.bool]] = np.zeros(
            (0,), dtype=bool
        )

    def check_triggers(
        self,
        cattail_context: CattailContext,
    ) -> np.ndarray[tuple[int], np.dtype[np.bool]]:
        if len(self.exhausted) != len(cattail_context.centers):
            self.exhausted = np.zeros((len(cattail_context.centers),), dtype=bool)

        acceleration_magnitudes = np.linalg.vector_norm(
            cattail_context.accelerations, axis=1
        )

        below_lower_threshold = acceleration_magnitudes < self.lower_threshold
        above_upper_threshold = acceleration_magnitudes > self.upper_threshold

        should_trigger = above_upper_threshold & ~self.exhausted

        self.exhausted[above_upper_threshold] = True
        self.exhausted[below_lower_threshold] = False

        return should_trigger


class ProjectileLightEffect(LightEffect):
    def __init__(self):
        self.trigger = AccelerationThresholdTrigger()
        self.projectile_physics = ProjectilePhysics()

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
        triggered = self.trigger.check_triggers(cattail_context)

        relevant_accelerations = cattail_context.accelerations[triggered]
        projectile_velocities = (
            -relevant_accelerations
            / np.linalg.vector_norm(relevant_accelerations)
            * 1.0
        )
        projectile_centers = cattail_context.centers[triggered]

        self.projectile_physics.update(delta_time, expire_older_than=5.0)
        projectile_centers += projectile_velocities * 1.0
        self.projectile_physics.add_projectiles(
            projectile_centers, projectile_velocities
        )
        return self.chain_brightness_from_projectiles(chains)

    def chain_brightness_from_projectiles(
        self,
        chains: list[Chain],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        chain_points: np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.floating]] = (
            np.stack([chain.points for chain in chains], dtype=np.float32)
        )

        distances = np.linalg.vector_norm(
            chain_points[:, :, np.newaxis, :]
            - self.projectile_physics.positions[np.newaxis, np.newaxis, :, :],
            axis=-1,
        )
        brightness = np.maximum(0.0, 1.0 - distances.min(axis=1) * 1.0).sum(axis=1)
        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    def debug_gui(self):
        self.projectile_physics.debug_gui()


class PulseLightEffect(LightEffect):
    @dataclass
    class Parameters:
        expansion_speed: float
        starting_size: float

        brightness_falloff_from_edge: float

        age_falloff_start: float
        age_falloff_rate: float

    DEFAULT_PARAMETERS = Parameters(
        expansion_speed=0.6,
        starting_size=0.7,
        brightness_falloff_from_edge=2.0,
        age_falloff_start=2.0,
        age_falloff_rate=0.1,
    )

    def __init__(self, params: Parameters = DEFAULT_PARAMETERS):
        self.trigger = AccelerationThresholdTrigger()
        self.pulse_physics = PulsePhysics()
        self.params = params

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
        triggered = self.trigger.check_triggers(cattail_context)

        projectile_centers = cattail_context.centers[triggered]

        self.pulse_physics.update(delta_time, expire_older_than=5.0)
        self.pulse_physics.add_pulses(
            projectile_centers,
            np.full((projectile_centers.shape[0],), self.params.starting_size),
            np.full((projectile_centers.shape[0],), self.params.expansion_speed),
        )

        return self.chain_brightness_from_projectiles(chains)

    def chain_brightness_from_projectiles(
        self,
        chains: list[Chain],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        chain_points = np.stack([chain.points for chain in chains], dtype=np.float32)
        chain_centers = np.mean(chain_points, axis=1)

        distances = distances_from_circles(
            chain_centers,
            self.pulse_physics.centers,
            self.pulse_physics.sizes,
        )
        brightness += np.maximum(
            1.0
            - distances * self.params.brightness_falloff_from_edge
            - np.maximum(
                (self.pulse_physics.ages - self.params.age_falloff_start)
                * self.params.age_falloff_rate,
                0.0,
            ),
            0.0,
        ).sum(axis=1)
        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    def debug_gui(self):
        self.pulse_physics.debug_gui()


class PathLightEffect(LightEffect):
    """
    Light effect where projectiles follow a single predefined path.
    When cattails trigger, projectiles are spawned on the path.
    """

    @dataclass
    class Parameters:
        path: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]
        projectile_speed: float = 1.0
        num_bounces: int = 0
        brightness_falloff: float = 1.0

    def __init__(self, params: Parameters):
        self.trigger = AccelerationThresholdTrigger()
        self.path_physics = PathPhysics(params.path)
        self.params = params

    def calculate_chain_brightness(
        self,
        delta_time,
        chains,
        cattail_context,
    ):
        triggered = self.trigger.check_triggers(cattail_context)

        # Spawn projectiles on the path for each trigger
        num_triggered = triggered.sum()
        if num_triggered > 0:
            self.path_physics.add_projectiles(
                starting_distances=np.zeros((num_triggered,), dtype=np.float32),
                speeds=np.full(
                    (num_triggered,), self.params.projectile_speed, dtype=np.float32
                ),
                num_bounces=np.full(
                    (num_triggered,), self.params.num_bounces, dtype=np.int32
                ),
            )

        self.path_physics.update(delta_time)
        return self.chain_brightness_from_projectiles(chains)

    def chain_brightness_from_projectiles(
        self,
        chains: list[Chain],
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        brightness = np.zeros((len(chains),), dtype=np.float32)

        if len(self.path_physics.positions) == 0:
            return brightness

        chain_points: np.ndarray[tuple[int, int, Literal[2]], np.dtype[np.floating]] = (
            np.stack([chain.points for chain in chains], dtype=np.float32)
        )

        # Calculate distances from all chain points to all projectiles
        distances = np.linalg.vector_norm(
            chain_points[:, :, np.newaxis, :]
            - self.path_physics.positions[np.newaxis, np.newaxis, :, :],
            axis=-1,
        )

        # Brightness based on minimum distance to any projectile
        brightness = np.maximum(
            0.0, 1.0 - distances.min(axis=1) * self.params.brightness_falloff
        ).sum(axis=1)
        brightness = np.clip(brightness, 0.0, 1.0)

        return brightness

    def debug_gui(self):
        self.path_physics.debug_gui()


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
