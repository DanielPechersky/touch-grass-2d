from typing import Literal

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import ndarray_to_scatter_many, point_to_ndarray
from simulator.light_effect import CattailContext, LightEffect
from simulator.typing import Chains


class Simulator:
    def __init__(
        self,
        chains: Chains,
        cattail_centers: np.ndarray[tuple[int, Literal[2]]],
        light_effect: LightEffect | None = None,
        cattail_spring_constant=5.0,
        cattail_damping=0.3,
    ) -> None:
        self.cattail_spring_constant = cattail_spring_constant
        self.cattail_damping = cattail_damping

        self.chains = chains

        self.cattail_centers = cattail_centers
        self.cattail_positions = cattail_centers.copy()
        self.cattail_velocities = np.zeros(cattail_centers.shape, dtype=np.float32)

        self.light_effect = light_effect

    def update_cattails(self, delta_time: float):
        accelerations = np.zeros(self.cattail_positions.shape, dtype=np.float32)

        pos = point_to_ndarray(implot.get_plot_mouse_pos())
        displacements_from_cursor = self.cattail_positions - pos
        distances_from_cursor = np.linalg.vector_norm(displacements_from_cursor, axis=1)
        directions_from_cursor = (
            displacements_from_cursor / distances_from_cursor[:, np.newaxis]
        )
        accelerations += (
            directions_from_cursor
            * np.maximum(0.5 - distances_from_cursor, 0)[:, np.newaxis]
        )

        distance_from_centers = self.cattail_positions - self.cattail_centers
        accelerations += -distance_from_centers * self.cattail_spring_constant

        self.cattail_velocities += accelerations * delta_time
        self.cattail_positions += self.cattail_velocities * delta_time

        self.cattail_velocities *= self.cattail_damping**delta_time

        return accelerations

    def gui(self, delta_time: float):
        accelerations = self.update_cattails(delta_time)

        implot.plot_scatter(
            "cattail_positions",
            *ndarray_to_scatter_many(self.cattail_positions),
        )

        if self.light_effect is not None:
            try:
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
                    implot.set_next_marker_style(
                        size=brightness * 5 + 3,
                        fill=imgui.ImVec4(brightness, brightness, 0.1, 1),
                    )
                    implot.plot_scatter(
                        "chain_brightness",
                        *ndarray_to_scatter_many(chain),
                    )
            except Exception as e:
                print(f"Error in light effect: {e}")
