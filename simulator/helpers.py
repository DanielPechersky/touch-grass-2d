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
