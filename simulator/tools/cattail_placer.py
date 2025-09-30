from typing import Literal

import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import ndarray_to_scatter, point_to_ndarray


class CattailPlacer:
    def __init__(self):
        pass

    @property
    def previewed_point(self):
        if not implot.is_plot_hovered():
            return None

        return point_to_ndarray(implot.get_plot_mouse_pos())

    def gui(self) -> np.ndarray[tuple[Literal[2]]] | None:
        if self.previewed_point is None:
            return None

        implot.set_next_marker_style(size=5, fill=imgui.ImVec4(1.0, 0.0, 0.0, 0.5))
        implot.plot_scatter("preview", *ndarray_to_scatter(self.previewed_point))
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            return self.previewed_point
