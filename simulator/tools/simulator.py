import traceback
from typing import Literal

import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, implot

from simulator.helpers import (
    ndarray_to_scatter_many,
    point_to_ndarray,
)
from simulator.light_effect_node_editor import Editor, SceneContext
from simulator.persistence import Cattail, Chain, Group
from simulator.tools import Tool


class Simulator(Tool):
    def __init__(
        self,
    ):
        self.set_chains([])
        self.set_cattails([])
        self.set_groups([])

        self.light_effect_debug_gui = False

        self.editor = Editor()

    def set_chains(self, chains: list[Chain[int]]):
        self.chains: list[Chain[int]] = chains

    def set_cattails(self, cattails: list[Cattail[int]]):
        self.cattails: list[Cattail[int]] = cattails
        if cattails:
            cattail_centers = np.stack([cattail.pos for cattail in cattails])
        else:
            cattail_centers = np.empty((0, 2), dtype=np.float32)
        self.cattail_physics = CattailPhysics(cattail_centers)  # type: ignore

    def set_groups(self, groups: list[Group[int]]):
        self.groups: list[Group[int]] = groups

    def main_gui(self):
        hello_imgui_set_idling(False)

        with imgui_ctx.begin("Effect Nodes"):
            self.editor.gui()

        delta_time = imgui.get_io().delta_time
        accelerations = self.cattail_physics.update_cattails(delta_time)

        self.plot_cattails()

        try:
            context = SceneContext(
                chains=self.chains,
                cattails=self.cattails,
                groups=self.groups,
                cattail_accelerations=accelerations,
            )
            self.editor.plot_gui()
            chain_brightness = self.editor.execute(context)
            if chain_brightness is None:
                chain_brightness = np.zeros((len(self.chains),), dtype=np.float32)

            assert (0 <= chain_brightness).all() and (chain_brightness <= 1).all()

            for chain, brightness in zip(self.chains, chain_brightness, strict=True):
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
            self.cattail_physics.cattail_centers,
            self.cattail_physics.cattail_positions,
            strict=True,
        ):
            implot.set_next_line_style(col=imgui.ImVec4(1.0, 0.6, 0.1, 1), weight=3)
            points = np.ascontiguousarray(np.stack([position, center]).T)
            implot.plot_line("cattail_lines", points[0], points[1])

        implot.set_next_marker_style(size=3, fill=imgui.ImVec4(1.0, 0.6, 0.1, 1))
        implot.plot_scatter(
            "cattail_positions",
            *ndarray_to_scatter_many(self.cattail_physics.cattail_positions),
        )

    def switched_away(self):
        self.cattail_physics.reset()
        self.editor = Editor()
        hello_imgui_set_idling(True)


def hello_imgui_set_idling(enable: bool):
    hello_imgui.get_runner_params().fps_idling.enable_idling = enable


class CattailPhysics:
    def __init__(
        self,
        cattail_centers: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
    ):
        self.cattail_centers = cattail_centers
        self.reset()

    def reset(self):
        self.cattail_positions: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = self.cattail_centers.copy()
        self.cattail_velocities: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.zeros(self.cattail_centers.shape, dtype=np.float32)  # type: ignore

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
