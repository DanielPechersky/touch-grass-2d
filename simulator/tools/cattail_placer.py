from imgui_bundle import imgui, implot

from simulator.helpers import (
    display_cattails,
    display_chains,
    ndarray_to_scatter,
    point_to_ndarray,
)
from simulator.persistence import Cattail, Persistence
from simulator.tools import Tool


class CattailPlacer(Tool):
    def __init__(self, persistence: Persistence):
        self.persistence = persistence

    @property
    def previewed_point(self):
        if not implot.is_plot_hovered():
            return None

        return point_to_ndarray(implot.get_plot_mouse_pos())

    def main_gui(self):
        display_chains(self.persistence.get_chains())
        display_cattails(self.persistence.get_cattails())

        if self.previewed_point is None:
            return None

        implot.set_next_marker_style(size=5, fill=imgui.ImVec4(1.0, 0.0, 0.0, 0.5))
        implot.plot_scatter("preview", *ndarray_to_scatter(self.previewed_point))
        if imgui.is_mouse_clicked(imgui.MouseButton_.left):
            self.persistence.append_cattail(Cattail(id=None, pos=self.previewed_point))
