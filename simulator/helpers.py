from typing import Literal

import numpy as np
from imgui_bundle import imgui, implot


def point_to_ndarray(
    point: implot.Point,
) -> np.ndarray[tuple[Literal[2]], np.dtype[np.floating]]:
    return np.array([point.x, point.y], dtype=np.float32)


def ndarray_to_scatter(array: np.ndarray[tuple[Literal[2]], np.dtype[np.floating]]):
    return array[0, np.newaxis], array[1, np.newaxis]


def ndarray_to_scatter_many(
    array: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
):
    return np.ascontiguousarray(array[:, 0]), np.ascontiguousarray(array[:, 1])


def plot_chain(
    label_id: str,
    chain: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]],
    color: imgui.ImVec4,
):
    implot.set_next_marker_style(fill=color)
    implot.plot_scatter(
        label_id,
        *ndarray_to_scatter_many(chain),
    )

    for i in range(len(chain) - 1):
        implot.set_next_line_style(col=color)
        implot.plot_line(
            label_id,
            np.ascontiguousarray(chain[i : i + 2, 0]),
            np.ascontiguousarray(chain[i : i + 2, 1]),
        )


def implot_draw_circle(
    center: np.ndarray[tuple[Literal[2]], np.dtype[np.floating]],
    radius: float,
    **kwargs,
):
    center_pixels = implot.plot_to_pixels(*center.tolist())  # type: ignore
    radius_pixels = plot_line_length_to_pixel_line_length(radius).x
    draw_list = implot.get_plot_draw_list()
    draw_list.add_circle_filled(
        center=center_pixels,
        radius=radius_pixels,
        # col=0x33FF0000,
        **kwargs,
    )


def plot_line_length_to_pixel_line_length(line_length: float) -> imgui.ImVec2:
    p1 = implot.plot_to_pixels(0, 0)
    p2 = implot.plot_to_pixels(line_length, line_length)
    return p2 - p1
