import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot

from simulator.helpers import (
    ndarray_to_scatter,
    ndarray_to_scatter_many,
    point_to_ndarray,
)


class Measurer:
    def __init__(self) -> None:
        self.first_point = None
        self.second_point = None
        self.user_set_distance = 0.0

        self.close_popup()

    @property
    def pixel_distance(self):
        if self.first_point is not None and self.second_point is not None:
            return np.linalg.vector_norm(self.second_point - self.first_point)

    @property
    def previewed_point(self):
        if not implot.is_plot_hovered():
            return None

        return point_to_ndarray(implot.get_plot_mouse_pos())

    def close_popup(self):
        if imgui.is_popup_open("set_distance"):
            with imgui_ctx.begin_popup("set_distance"):
                imgui.close_current_popup()

    # returns pixels_per_metre if the user is done measuring
    def gui(self) -> float | None:
        if self.previewed_point is not None and imgui.is_mouse_clicked(
            imgui.MouseButton_.left
        ):
            pos = self.previewed_point
            if self.first_point is None:
                self.first_point = pos
            elif self.second_point is None:
                self.second_point = pos
                imgui.open_popup("set_distance")
            else:
                self.first_point = None
                self.second_point = None

        if self.first_point is not None:
            implot.set_next_marker_style(size=5, fill=imgui.ImVec4(1.0, 0.0, 0.0, 1.0))
            implot.plot_scatter("measurement", *ndarray_to_scatter(self.first_point))

            if self.second_point is not None:
                implot.set_next_marker_style(
                    size=5, fill=imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
                )
                implot.plot_scatter(
                    "measurement", *ndarray_to_scatter(self.second_point)
                )

                line = np.array([self.first_point, self.second_point])
                implot.set_next_line_style(
                    col=imgui.ImVec4(1.0, 0.0, 0.0, 1.0), weight=2
                )
                implot.plot_line("measurement", *ndarray_to_scatter_many(line))

        if imgui.is_popup_open("set_distance"):
            with imgui_ctx.begin_popup("set_distance"):
                imgui.text("Set the distance")
                set, value = imgui.input_float("metres", self.user_set_distance)
                if set:
                    self.user_set_distance = value
                imgui.same_line()
                if imgui.button("Submit"):
                    if self.user_set_distance != 0.0:
                        imgui.close_current_popup()
                        assert self.pixel_distance is not None
                        return (self.pixel_distance / self.user_set_distance).item()

        else:
            self.user_set_distance = 0.0
            if self.pixel_distance is not None:
                self.first_point = None
                self.second_point = None
