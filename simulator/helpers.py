from typing import Generator, Literal

import numpy as np
from imgui_bundle import imgui, implot

from simulator.persistence import Cattail, Chain, Group


def display_chains(chains: list[Chain]):
    for chain in chains:
        plot_chain("chains", chain.points, color=imgui.ImVec4(1.0, 1.0, 0.0, 1.0))


def display_cattails(cattails: list[Cattail]):
    implot.set_next_marker_style(
        marker=implot.Marker_.square, size=10, fill=(1.0, 0.0, 0.0, 1.0)
    )
    if not cattails:
        return
    cattail_positions = np.stack([cattail.pos for cattail in cattails])
    implot.plot_scatter(
        "cattails",
        *ndarray_to_scatter_many(cattail_positions),
    )


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
        **kwargs,
    )


def implot_draw_rectangle(
    bottom_left: np.ndarray[tuple[Literal[2]], np.dtype[np.floating]],
    top_right: np.ndarray[tuple[Literal[2]], np.dtype[np.floating]],
    **kwargs,
):
    bottom_left_pixels = implot.plot_to_pixels(*bottom_left.tolist())  # type: ignore
    top_right_pixels = implot.plot_to_pixels(*top_right.tolist())  # type: ignore
    draw_list = implot.get_plot_draw_list()
    draw_list.add_rect_filled(
        p_min=bottom_left_pixels, p_max=top_right_pixels, **kwargs
    )


def plot_line_length_to_pixel_line_length(line_length: float) -> imgui.ImVec2:
    p1 = implot.plot_to_pixels(0, 0)
    p2 = implot.plot_to_pixels(line_length, line_length)
    return p2 - p1


def pixel_line_length_to_plot_line_length(line_length: float) -> implot.Point:
    p1 = implot.pixels_to_plot(0, 0)
    p2 = implot.pixels_to_plot(line_length, line_length)
    return implot.Point(p2.x - p1.x, p2.y - p1.y)


type GroupChild = Group[int] | Chain[int] | Cattail[int]
type ChildrenTree = dict[int | None, list[GroupChild]]


def group_children(
    groups: list[Group[int]],
    chains: list[Chain[int]],
    cattails: list[Cattail[int]],
    is_root=False,
) -> ChildrenTree:
    children: ChildrenTree = {}
    if is_root:
        children[None] = []

    for group in groups:
        children.setdefault(group.parent_group_id, []).append(group)
        children.setdefault(group.id, [])

    for cattail in cattails:
        children.setdefault(cattail.group_id, []).append(cattail)

    for chain in chains:
        children.setdefault(chain.group_id, []).append(chain)

    return children


def children_under_group(
    tree: ChildrenTree, group_id: int | None
) -> Generator[GroupChild]:
    for child in tree[group_id]:
        match child:
            case Group() as group:
                yield group
                yield from children_under_group(tree, group.id)
            case Chain() | Cattail() as child:
                yield child
