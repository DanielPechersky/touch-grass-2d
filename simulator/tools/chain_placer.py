from typing import Literal

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import ndarray_to_scatter, plot_chain, point_to_ndarray
from simulator.persistence import Chain, Persistence
from simulator.tools import Tool


class ChainPlacer(Tool):
    def __init__(
        self,
        persistence: Persistence,
        location_id: int,
        chain_size: int,
        spacing: float,
    ):
        self.persistence = persistence
        self.location_id = location_id
        self.chain_size = chain_size
        self.spacing = spacing

        self.current_chain: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = np.empty((0, 2), dtype=np.float32)  # type: ignore

    @property
    def current_chain_length(self):
        return self.current_chain.shape[0]

    @property
    def last_placed_point(self):
        try:
            return self.current_chain[-1, :]
        except IndexError:
            return None

    @property
    def previewed_point(self):
        if not implot.is_plot_hovered():
            return None

        pos = point_to_ndarray(implot.get_plot_mouse_pos())
        if self.last_placed_point is None:
            return pos
        vec = pos - self.last_placed_point
        norm = np.linalg.vector_norm(vec)
        if norm == 0.0:
            return None
        vec /= np.linalg.vector_norm(vec)
        vec *= self.spacing
        return self.last_placed_point + vec

    def main_gui(self):
        plot_chain(
            "new_chain", self.current_chain, color=imgui.ImVec4(0.0, 0.8, 0.0, 1.0)
        )

        if self.previewed_point is None:
            return

        implot.set_next_marker_style(
            marker=implot.Marker_.diamond, size=5, fill=imgui.ImVec4(0.0, 1.0, 0.0, 0.5)
        )
        implot.plot_scatter("preview", *ndarray_to_scatter(self.previewed_point))
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            self.current_chain = np.concatenate(
                (self.current_chain, self.previewed_point[np.newaxis, :])
            )

        if self.current_chain_length == self.chain_size:
            result = self.current_chain
            self.current_chain = np.empty((0, 2), dtype=np.float32)  # type: ignore
            self.persistence.append_chain(
                self.location_id,
                Chain(id=None, points=result),
            )

    def switched_away(self):
        self.current_chain = np.empty((0, 2), dtype=np.float32)  # type: ignore
