import traceback
from typing import Literal

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import ndarray_to_scatter_many, point_to_ndarray
from simulator.light_effect import CattailContext, LightEffect
from simulator.persistence import Cattail, Chain


class Simulator:
    def __init__(
        self,
        chains: list[Chain],
        cattails: list[Cattail],
        light_effect: LightEffect | None = None,
        cattail_spring_constant=5.0,
        cattail_damping=0.3,
    ):
        self.cattail_spring_constant = cattail_spring_constant
        self.cattail_damping = cattail_damping

        self.chains = chains

        self.cattail_centers: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]
        if cattails:
            self.cattail_centers = np.stack([cattail.pos for cattail in cattails])
        else:
            self.cattail_centers = np.empty((0, 2), dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]
        self.cattail_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = self.cattail_centers.copy()
        self.cattail_velocities: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros(self.cattail_centers.shape, dtype=np.float32)  # pyright: ignore[reportAttributeAccessIssue]

        self.light_effect = light_effect

        self.light_effect_debug_gui = False

    def update_cattails(
        self, delta_time: float
    ) -> np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]:
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
            * np.maximum(0.5 - distances_from_cursor[valid], 0)[:, np.newaxis]
        )

        distance_from_centers = self.cattail_positions - self.cattail_centers
        accelerations += -distance_from_centers * self.cattail_spring_constant

        self.cattail_velocities += accelerations * delta_time
        self.cattail_positions += self.cattail_velocities * delta_time

        self.cattail_velocities *= self.cattail_damping**delta_time

        return accelerations  # pyright: ignore[reportReturnType]

    def gui(self, delta_time: float):
        accelerations = self.update_cattails(delta_time)

        implot.plot_scatter(
            "cattail_positions",
            *ndarray_to_scatter_many(self.cattail_positions),
        )

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

    def tool_gui(self):
        _, self.light_effect_debug_gui = imgui.checkbox(
            "Light Effect Debug", self.light_effect_debug_gui
        )
