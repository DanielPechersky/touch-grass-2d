import traceback
from typing import Literal

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import (
    ndarray_to_scatter_many,
    point_to_ndarray,
)
from simulator.light_effect import CattailContext, LightEffect
from simulator.persistence import Cattail, Chain


class Simulator:
    def __init__(
        self,
        chains: list[Chain],
        cattails: list[Cattail],
        light_effect: LightEffect | None = None,
    ):
        self.chains = chains

        self.cattail_centers: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]
        if cattails:
            self.cattail_centers = np.stack([cattail.pos for cattail in cattails])
        else:
            self.cattail_centers = np.empty((0, 2), dtype=np.float32)  # type: ignore
        self.cattail_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = self.cattail_centers.copy()
        self.cattail_velocities: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros(self.cattail_centers.shape, dtype=np.float32)  # type: ignore

        self.light_effect = light_effect

        self.light_effect_debug_gui = False

    def accelerations_from_cursor(self):
        accelerations = np.zeros(self.cattail_positions.shape, dtype=np.float32)

        pos = point_to_ndarray(implot.get_plot_mouse_pos())
        displacements_from_cursor = self.cattail_positions - pos
        distances_from_cursor = np.linalg.vector_norm(displacements_from_cursor, axis=1)
        valid = distances_from_cursor > 1e-6
        directions_from_cursor = (
            displacements_from_cursor[valid]
            / distances_from_cursor[valid][:, np.newaxis]
        )

        accelerations[valid] += (
            directions_from_cursor
            * 6.0
            * np.maximum(0.5 - distances_from_cursor[valid], 0)[:, np.newaxis]
        )
        return accelerations

    def accelerations_from_spring(self):
        distance_from_centers = self.cattail_positions - self.cattail_centers
        force_inwards = distance_from_centers * 4.0 + distance_from_centers**2 * 1.0
        return -force_inwards

    def decay(self, delta_time: float):
        """
        Decays the velocity of the cattail, as if through friction or air resistance.
        Decays radial velocity more than angular velocity.
        """
        displacement_from_center = self.cattail_positions - self.cattail_centers
        displacement_norm = np.linalg.vector_norm(displacement_from_center, axis=1)
        # Avoid division by zero
        displacement_norm = np.where(displacement_norm == 0, 1, displacement_norm)
        radial_dir = displacement_from_center / displacement_norm[:, np.newaxis]

        # Project velocities onto radial and tangential directions
        radial_velocity = (
            np.sum(self.cattail_velocities * radial_dir, axis=1, keepdims=True)
            * radial_dir
        )
        tangential_velocity = self.cattail_velocities - radial_velocity

        radial_velocity *= np.exp(-0.1 * delta_time)
        tangential_velocity *= np.exp(-0.5 * delta_time)

        self.cattail_velocities = radial_velocity + tangential_velocity

    def update_cattails(
        self, delta_time: float
    ) -> np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]:
        accelerations = (
            self.accelerations_from_cursor() + self.accelerations_from_spring()
        )

        self.cattail_velocities += accelerations * delta_time
        self.cattail_positions += self.cattail_velocities * delta_time

        self.decay(delta_time)

        return accelerations  # type: ignore

    def gui(self, delta_time: float):
        accelerations = self.update_cattails(delta_time)

        self.plot_cattails()

        if self.light_effect is not None:
            try:
                if self.light_effect_debug_gui:
                    self.light_effect.debug_gui()
                chain_brightness = self.light_effect.calculate_chain_brightness(
                    delta_time=delta_time,
                    chains=self.chains,
                    cattail_context=CattailContext(
                        centers=self.cattail_centers,
                        accelerations=accelerations,
                    ),
                )

                assert (0 <= chain_brightness).all() and (chain_brightness <= 1).all()

                for chain, brightness in zip(
                    self.chains, chain_brightness, strict=True
                ):
                    brightness = brightness.item()
                    implot.set_next_marker_style(
                        size=brightness * 5 + 3,
                        fill=imgui.ImVec4(brightness, brightness, 0.1, 1),
                    )
                    implot.plot_scatter(
                        "chain_brightness",
                        *ndarray_to_scatter_many(chain.points),
                    )
            except Exception:
                print("Error in light effect")
                traceback.print_exc()

    def plot_cattails(self):
        for center, position in zip(
            self.cattail_centers, self.cattail_positions, strict=True
        ):
            implot.set_next_line_style(col=imgui.ImVec4(1.0, 0.6, 0.1, 1), weight=3)
            points = np.ascontiguousarray(np.stack([position, center]).T)
            implot.plot_line("cattail_lines", points[0], points[1])

        implot.set_next_marker_style(size=3, fill=imgui.ImVec4(1.0, 0.6, 0.1, 1))
        implot.plot_scatter(
            "cattail_positions",
            *ndarray_to_scatter_many(self.cattail_positions),
        )

    def tool_gui(self):
        _, self.light_effect_debug_gui = imgui.checkbox(
            "Light Effect Debug", self.light_effect_debug_gui
        )
