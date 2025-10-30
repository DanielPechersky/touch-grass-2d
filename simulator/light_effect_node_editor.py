import contextlib
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, Self

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot
from imgui_bundle import imgui_node_editor as ed
from imgui_bundle import imgui_node_editor_ctx as ed_ctx
from imgui_bundle.immapp import icons_fontawesome_6 as fa6
from pydantic import BaseModel, ValidationError

from simulator.helpers import (
    ChildrenTree,
    children_under_group,
    group_children,
    ndarray_to_scatter_many,
    point_to_ndarray,
)
from simulator.light_effect import (
    CattailContext,
    PathLightEffect,
    ProjectileLightEffect,
    PulseLightEffect,
)
from simulator.persistence import (
    Cattail,
    Chain,
    EffectNode,
    EffectNodeLink,
    Group,
    Persistence,
)


class SceneContext:
    def __init__(
        self,
        chains: list[Chain[int]],
        cattails: list[Cattail[int]],
        groups: list[Group[int]],
        cattail_accelerations: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ],
    ):
        self.chains: list[Chain[int]] = chains
        self.cattails: list[Cattail[int]] = cattails
        self.groups: list[Group[int]] = groups
        self.cattail_accelerations: np.ndarray[
            tuple[int, Literal[2]], np.dtype[np.floating]
        ] = cattail_accelerations

        self.child_tree: ChildrenTree = group_children(groups, chains, cattails)

    def scoped_to_group(
        self,
        group_id: int,
        filter_chains: bool,
        filter_cattails: bool,
        filter_groups: bool = False,
    ) -> Self:
        groups = []
        chains = []
        cattails = []
        for child in children_under_group(self.child_tree, group_id):
            match child:
                case Group() as g:
                    groups.append(g)
                case Chain() as c:
                    chains.append(c)
                case Cattail() as c:
                    cattails.append(c)

        cattail_ids = np.sort([c.id for c in self.cattails])
        new_cattail_ids = np.sort([c.id for c in cattails])

        accelerations = self.cattail_accelerations[
            np.searchsorted(cattail_ids, new_cattail_ids)
        ]

        if not filter_chains:
            chains = self.chains
        if not filter_cattails:
            cattails = self.cattails
            accelerations = self.cattail_accelerations
        if not filter_groups:
            groups = self.groups

        return SceneContext(chains, cattails, groups, accelerations)  # type: ignore

    def cattail_centers(
        self,
    ) -> np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]:
        if len(self.cattails) == 0:
            return np.zeros((0, 2), dtype=np.float32)  # type: ignore
        return np.stack([cattail.pos for cattail in self.cattails])

    def chain_ids(self):
        return np.sort(np.array([chain.id for chain in self.chains]))


@dataclass
class Brightness:
    values: np.ndarray[tuple[int], np.dtype[np.floating]]
    chain_ids: np.ndarray[tuple[int], np.dtype[np.integer]]


@dataclass(frozen=True)
class PinId:
    node_id: int
    name: str


@dataclass(frozen=True)
class PinType:
    type: Literal["input", "output"]
    data: Literal["scene_context", "brightness"]


@dataclass(frozen=True)
class Link:
    start: PinId
    end: PinId


def draw_pin_text(type: PinType) -> None:
    match type.data:
        case "brightness":
            symbol = fa6.ICON_FA_LIGHTBULB
            color = imgui.ImVec4(1.0, 1.0, 0.0, 1.0)
            hover_text = "Brightness Values"
        case "scene_context":
            symbol = fa6.ICON_FA_SQUARE
            color = imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
            hover_text = "Scene Context (Cattails, Chains, and Groups)"
    match type.type:
        case "input":
            text = f"{fa6.ICON_FA_ARROW_RIGHT_LONG} {symbol}"
        case "output":
            text = f"{symbol} {fa6.ICON_FA_ARROW_RIGHT_LONG}"
    imgui.text_colored(color, text)
    if imgui.is_item_hovered():
        imgui.set_tooltip(hover_text)


def draw_pin(
    pin_ids: Callable[[PinId], ed.PinId], pin_id: PinId, type: PinType
) -> None:
    with ed_ctx.begin_pin(
        pin_ids(pin_id), ed.PinKind.input if type.type == "input" else ed.PinKind.output
    ):
        draw_pin_text(type)


def draw_pins(pin_ids: Callable[[PinId], ed.PinId], pins: dict[PinId, PinType]) -> None:
    input_pins = [
        pin_id for pin_id, pin_type in pins.items() if pin_type.type == "input"
    ]
    output_pins = [
        pin_id for pin_id, pin_type in pins.items() if pin_type.type == "output"
    ]

    for pin_id in input_pins:
        draw_pin(pin_ids, pin_id, pins[pin_id])

    imgui.same_line(0, 30)

    for pin_id in output_pins:
        draw_pin(pin_ids, pin_id, pins[pin_id])


INVALID_INPUT_COLOR = imgui.ImVec4(1.0, 0.5, 0.0, 1.0)
INVALID_INPUT_SYMBOL = fa6.ICON_FA_TRIANGLE_EXCLAMATION


class EmptyParams(BaseModel):
    pass


class Node[P: BaseModel](Protocol):
    id: int
    params: P
    Params: type[P]

    def __init__(self, id: int):
        raise NotImplementedError

    @property
    def deletable(self) -> bool:
        return True

    def pins(self) -> dict[PinId, PinType]:
        raise NotImplementedError

    def gui(self, pin_ids: Callable[[PinId], ed.PinId]) -> None:
        raise NotImplementedError

    def plot_gui(self) -> None:
        pass

    def run(
        self, inputs: dict[PinId, Any]
    ) -> Mapping[PinId, Brightness | SceneContext]:
        raise NotImplementedError


class SourceNode(Node[EmptyParams]):
    Params = EmptyParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()
        self.context: SceneContext | None = None

        self.output_pin = PinId(self.id, "output")

    def set_context(self, context: SceneContext):
        self.context = context

    def pins(self):
        return {self.output_pin: PinType("output", "scene_context")}

    def gui(self, pin_ids):
        imgui.text("Source")
        imgui.new_line()
        draw_pins(pin_ids, self.pins())

    def run(self, inputs) -> dict[PinId, Any]:
        assert self.context is not None
        _ = inputs
        return {self.output_pin: self.context}


def combine_brightnesses(*brightnesses: Brightness) -> Brightness:
    if not brightnesses:
        return Brightness(
            values=np.zeros((0,), dtype=np.float32),
            chain_ids=np.zeros((0,), dtype=np.int32),
        )

    all_chain_ids = np.concatenate([b.chain_ids for b in brightnesses])
    all_chain_ids = np.unique(all_chain_ids)

    result_values = np.zeros((len(all_chain_ids),), dtype=np.float32)

    for brightness in brightnesses:
        result_indices = np.searchsorted(all_chain_ids, brightness.chain_ids)
        np.add.at(result_values, result_indices, brightness.values)

    result_values = np.clip(result_values, 0.0, 1.0)

    return Brightness(values=result_values, chain_ids=all_chain_ids)


class DestinationNode(Node):
    Params = EmptyParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()
        self.brightness: Brightness | None = None

        self.input_pin = PinId(self.id, "input")

    def pins(self):
        return {self.input_pin: PinType("input", "brightness")}

    def gui(self, pin_ids):
        imgui.text("Destination")
        draw_pins(pin_ids, self.pins())

    def run(self, inputs):
        self.brightness = inputs[self.input_pin]
        return {}


class AlwaysBrightNode(Node):
    Params = EmptyParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids):
        imgui.text("Always Bright")
        draw_pins(pin_ids, self.pins())

    def run(self, inputs):
        context: SceneContext = inputs[self.input_pin]
        brightness = Brightness(
            values=np.ones((len(context.chains),), dtype=np.float32),
            chain_ids=context.chain_ids(),
        )
        return {self.output_pin: brightness}


class InvertNode(Node):
    Params = EmptyParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

    def pins(self):
        return {
            self.input_pin: PinType("input", "brightness"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Invert")
        draw_pins(pin_ids, self.pins())

    def run(self, inputs) -> dict[PinId, Any]:
        brightness: Brightness = inputs[self.input_pin]
        brightness.values = 1.0 - brightness.values
        return {self.output_pin: brightness}


class DimNodeParams(BaseModel):
    brightness: float = 0.5


class DimNode(Node):
    Params = DimNodeParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

    def pins(self):
        return {
            self.input_pin: PinType("input", "brightness"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Dim")
        imgui.set_next_item_width(100)
        self.params.brightness = imgui.slider_float(
            f"##{self.id}Brightness", self.params.brightness, 0.0, 1.0
        )[1]

        draw_pins(pin_ids, self.pins())

    def run(self, inputs) -> dict[PinId, Any]:
        brightness: Brightness = inputs[self.input_pin]
        brightness.values *= self.params.brightness
        return {self.output_pin: brightness}


class ProjectileLightEffectNode(Node):
    Params = EmptyParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

        self.light_effect: ProjectileLightEffect = ProjectileLightEffect()
        self.debug_gui_enabled = False

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Projectile Light Effect")
        self.debug_gui_enabled = imgui.checkbox(
            f"Debug GUI##{self.id}Debug Gui",
            self.debug_gui_enabled,
        )[1]
        draw_pins(pin_ids, self.pins())

    def plot_gui(self) -> None:
        if self.debug_gui_enabled:
            self.light_effect.debug_gui()

    def run(self, inputs) -> dict[PinId, Any]:
        context: SceneContext = inputs[self.input_pin]
        brightness = self.light_effect.calculate_chain_brightness(
            imgui.get_io().delta_time,
            context.chains,
            CattailContext(
                centers=context.cattail_centers(),
                accelerations=context.cattail_accelerations,
            ),
        )
        indices = np.array([c.id for c in context.chains])
        return {self.output_pin: Brightness(brightness, indices)}


class PulseLightEffectNodeParams(BaseModel):
    expansion_speed: float = 3.0
    starting_size: float = 0.0

    brightness_falloff_from_edge: float = 1.0

    age_falloff_start: float = 0.0
    age_falloff_rate: float = 0.0


class PulseLightEffectNode(Node[PulseLightEffectNodeParams]):
    Params = PulseLightEffectNodeParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

        self.effect = PulseLightEffect()
        self.debug_gui_enabled = False

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Pulse Light Effect")

        if imgui.collapsing_header(f"Parameters##{self.id}Parameters"):
            SLIDER_WIDTH = 150

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.expansion_speed = imgui.slider_float(
                f"Expansion Speed##{self.id}Expansion Speed",
                self.params.expansion_speed,
                -10.0,
                10.0,
            )[1]

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.starting_size = imgui.slider_float(
                f"Starting Size##{self.id}Starting Size",
                self.params.starting_size,
                0.0,
                10.0,
            )[1]

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.brightness_falloff_from_edge = imgui.slider_float(
                f"Brightness Falloff from Edge##{self.id}Brightness Falloff from Edge",
                self.params.brightness_falloff_from_edge,
                0.0,
                5.0,
            )[1]

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.age_falloff_start = imgui.slider_float(
                f"Age Falloff Start##{self.id}Age Falloff Start",
                self.params.age_falloff_start,
                0.0,
                10.0,
            )[1]

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.age_falloff_rate = imgui.slider_float(
                f"Age Falloff Rate##{self.id}Age Falloff Rate",
                self.params.age_falloff_rate,
                0.0,
                1.0,
            )[1]

        self.debug_gui_enabled = imgui.checkbox(
            f"Debug GUI##{self.id}Debug Gui",
            self.debug_gui_enabled,
        )[1]

        draw_pins(pin_ids, self.pins())

    def plot_gui(self) -> None:
        if self.debug_gui_enabled:
            self.effect.debug_gui()

    def run(self, inputs) -> dict[PinId, Any]:
        self.effect.params = PulseLightEffect.Parameters(**self.params.model_dump())
        context: SceneContext = inputs[self.input_pin]
        brightness = self.effect.calculate_chain_brightness(
            imgui.get_io().delta_time,
            context.chains,
            CattailContext(
                centers=context.cattail_centers(),
                accelerations=context.cattail_accelerations,
            ),
        )
        indices = np.array([c.id for c in context.chains])
        return {self.output_pin: Brightness(brightness, indices)}


@dataclass
class Points:
    array: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]] = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.float32)
    )  # type: ignore

    def __eq__(self, other):
        if not isinstance(other, Points):
            return False
        return np.array_equal(self.array, other.array)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        def _deserialize(v):
            if isinstance(v, np.ndarray):
                return cls(v)
            return cls(np.array(v, dtype=np.float32, ndmin=2).reshape(-1, 2))  # type: ignore

        def _serialize(v):
            return v.array.tolist()

        return core_schema.no_info_plain_validator_function(
            _deserialize,
            serialization=core_schema.plain_serializer_function_ser_schema(
                _serialize,
                return_schema=core_schema.list_schema(
                    core_schema.list_schema(core_schema.float_schema())
                ),
            ),
        )


class PathLightEffectNodeParams(BaseModel):
    path: Points = Points()
    projectile_speed: float = 1.0
    num_bounces: int = 0
    brightness_falloff: float = 1.0


class PathLightEffectNode(Node):
    Params = PathLightEffectNodeParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

        self.effect: PathLightEffect = PathLightEffect(self.effect_params)
        self.debug_gui_enabled = False
        self.editing_path = False

    @property
    def path(self) -> np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]:
        return self.params.path.array

    @path.setter
    def path(self, value: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]):
        self.params.path = Points(value)

    @property
    def effect_params(self):
        return PathLightEffect.Parameters(
            path=self.path,
            projectile_speed=self.params.projectile_speed,
            num_bounces=self.params.num_bounces,
            brightness_falloff=self.params.brightness_falloff,
        )

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Path Light Effect")

        if self.editing_path:
            button_text = f"{fa6.ICON_FA_PENCIL} Editing Path"
            button_style = imgui_ctx.push_style_color(
                imgui.Col_.button, imgui.ImVec4(0.2, 0.7, 0.2, 1.0)
            )
        else:
            button_text = "Edit Path"
            button_style = contextlib.nullcontext()

        with button_style:
            if imgui.button(f"{button_text}##{self.id}EditPath"):
                self.editing_path = not self.editing_path

        # Show status indicator
        num_points = len(self.path)
        if num_points < 2:
            imgui.text_colored(
                INVALID_INPUT_COLOR,
                f"{INVALID_INPUT_SYMBOL} Path needs at least 2 points ({num_points}/2)",
            )
        else:
            imgui.text(f"Path has {num_points} points")

        if imgui.collapsing_header(f"Parameters##{self.id}Parameters"):
            SLIDER_WIDTH = 150

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.projectile_speed = imgui.slider_float(
                f"Projectile Speed##{self.id}ProjectileSpeed",
                self.params.projectile_speed,
                0.0,
                5.0,
            )[1]

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.num_bounces = imgui.slider_int(
                f"Num Bounces##{self.id}NumBounces",
                self.params.num_bounces,
                0,
                10,
            )[1]

            imgui.set_next_item_width(SLIDER_WIDTH)
            self.params.brightness_falloff = imgui.slider_float(
                f"Brightness Falloff##{self.id}BrightnessFalloff",
                self.params.brightness_falloff,
                0.0,
                5.0,
            )[1]

        self.debug_gui_enabled = imgui.checkbox(
            f"Debug GUI##{self.id}DebugGui",
            self.debug_gui_enabled,
        )[1]

        draw_pins(pin_ids, self.pins())

    def plot_gui(self) -> None:
        if self.editing_path:
            self._draw_path_editor()
        elif self.debug_gui_enabled and self.effect is not None:
            self.effect.debug_gui()

    def _draw_path_editor(self) -> None:
        """Interactive path editor within the plot."""

        # Draw existing path as a line
        if len(self.path) >= 2:
            implot.plot_line(f"path_{self.id}", *ndarray_to_scatter_many(self.path))

        # Draw path points as editable markers
        if len(self.path) > 0:
            # Highlight differently when editing
            marker_color = (
                imgui.ImVec4(0.0, 1.0, 0.0, 1.0)
                if self.editing_path
                else imgui.ImVec4(1.0, 0.5, 0.0, 1.0)
            )
            implot.set_next_marker_style(size=10, fill=marker_color)
            implot.plot_scatter(f"path_{self.id}", *ndarray_to_scatter_many(self.path))

        # Only handle user interaction if editing is enabled
        if self.editing_path:
            plot_hovered = implot.is_plot_hovered()

            # Disable editing if clicking outside the plot area
            if imgui.is_mouse_clicked(imgui.MouseButton_.left) and not plot_hovered:
                self.editing_path = False

            if plot_hovered:
                mouse_pos = point_to_ndarray(implot.get_plot_mouse_pos())

                # Left click to add point
                if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    self.path = np.concatenate([self.path, mouse_pos[np.newaxis, :]])

                # Right click to remove nearest point
                elif imgui.is_mouse_clicked(imgui.MouseButton_.right):
                    if len(self.path) > 0:
                        # Find nearest point to mouse
                        distances = np.linalg.norm(self.path - mouse_pos, axis=1)
                        nearest_idx = int(np.argmin(distances))

                        # Remove if within reasonable distance (e.g., 0.5 units)
                        if distances[nearest_idx] < 0.5:
                            self.path = np.delete(self.path, nearest_idx, axis=0)

                # Show instructions as tooltip when editing and hovering
                imgui.begin_tooltip()
                imgui.text("Left Click: Add point")
                imgui.text("Right Click: Remove nearest point")
                imgui.text("Click outside plot to stop editing")
                imgui.end_tooltip()

    def run(self, inputs) -> dict[PinId, Any]:
        self.effect.params = self.effect_params

        context: SceneContext = inputs[self.input_pin]

        if self.effect is None:
            brightness = np.zeros((len(context.chains),), dtype=np.float32)
        else:
            brightness = self.effect.calculate_chain_brightness(
                imgui.get_io().delta_time,
                context.chains,
                CattailContext(
                    centers=context.cattail_centers(),
                    accelerations=context.cattail_accelerations,
                ),
            )

        indices = np.array([c.id for c in context.chains])
        return {self.output_pin: Brightness(brightness, indices)}


class FilterByGroupNode(Node):
    Params = EmptyParams

    def __init__(self, id: int):
        self.id = id
        self.params = self.Params()

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

        self.group_id: str = ""
        self.group_id_not_found = False

        self.filter_chains = True
        self.filter_cattails = True

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "scene_context"),
        }

    def gui(self, pin_ids):
        imgui.text("Filter By Group")
        imgui.set_next_item_width(50)
        set, value = imgui.input_text(f"Group ID##{self.id}Group ID", self.group_id)
        if set:
            self.group_id = value
        if not self.is_group_id_format_valid:
            imgui.text_colored(
                INVALID_INPUT_COLOR,
                f"{INVALID_INPUT_SYMBOL} Invalid group ID (must start with 'g_')",
            )
        elif self.group_id_not_found:
            imgui.text_colored(
                INVALID_INPUT_COLOR, f"{INVALID_INPUT_SYMBOL} Group ID not found"
            )
        self.filter_chains = imgui.checkbox(
            f"Filter Chains##{self.id}Filter Chains", self.filter_chains
        )[1]
        self.filter_cattails = imgui.checkbox(
            f"Filter Cattails##{self.id}Filter Cattails", self.filter_cattails
        )[1]

        draw_pins(pin_ids, self.pins())

    @property
    def is_group_id_format_valid(self):
        return self.parsed_group_id is not None

    @property
    def parsed_group_id(self) -> int | None:
        if not self.group_id.startswith("g_"):
            return None
        try:
            return int(self.group_id[2:])
        except ValueError:
            return None

    def run(self, inputs):
        scene_context = inputs[self.input_pin]
        if (group_id := self.parsed_group_id) is None:
            pass
        elif group_id not in {g.id for g in scene_context.groups}:
            self.group_id_not_found = True
        else:
            self.group_id_not_found = False
            scene_context = scene_context.scoped_to_group(
                group_id,
                filter_chains=self.filter_chains,
                filter_cattails=self.filter_cattails,
            )
        return {self.output_pin: scene_context}


class EditorIdProtocol(Protocol):
    def id(self) -> int:
        raise NotImplementedError


class EditorIdGenerator[K, V: EditorIdProtocol]:
    def __init__(self, start_editor_id: int, wrapper: Callable[[int], V]):
        self.ids: dict[K, int] = {}
        self.ids_reverse: dict[int, K] = {}
        self.current_id = start_editor_id
        self.wrapper = wrapper

    def get(self, k: K) -> V:
        if k not in self.ids:
            self.ids[k] = self.current_id
            self.ids_reverse[self.current_id] = k
            self.current_id += 1
        return self.wrapper(self.ids[k])

    def get_key(self, v: V) -> K:
        return self.ids_reverse[v.id()]


def node_types() -> dict[str, type[Node]]:
    return {
        "Source": SourceNode,
        "Destination": DestinationNode,
        "AlwaysBright": AlwaysBrightNode,
        "Invert": InvertNode,
        "ProjectileLightEffect": ProjectileLightEffectNode,
        "Dim": DimNode,
        "FilterByGroup": FilterByGroupNode,
        "PulseLightEffect": PulseLightEffectNode,
        "PathLightEffect": PathLightEffectNode,
    }


def node_display_names() -> dict[str, str]:
    return {
        "Source": "Source",
        "Destination": "Destination",
        "AlwaysBright": "Always Bright",
        "Invert": "Invert",
        "Dim": "Dim",
        "FilterByGroup": "Filter by Group",
        "PulseLightEffect": "Pulse Light Effect",
        "PathLightEffect": "Path Light Effect",
        "ProjectileLightEffect": "Projectile Light Effect",
    }


class Editor:
    def __init__(self, persistence: Persistence):
        self.persistence = persistence
        self.nodes: dict[int, Node] = {}
        self.node_ids = EditorIdGenerator[int, ed.NodeId](
            start_editor_id=1, wrapper=ed.NodeId
        )
        self.link_ids = EditorIdGenerator[int, ed.LinkId](
            start_editor_id=10_000_000, wrapper=ed.LinkId
        )
        self.pin_ids = EditorIdGenerator[PinId, ed.PinId](
            start_editor_id=20_000_000, wrapper=ed.PinId
        )

        self.frame_count = 0

        self.context_window_opened_at: np.ndarray[
            tuple[Literal[2]], np.dtype[np.floating]
        ] = np.array([0.0, 0.0])

    def _load_nodes(self):
        for effect_node in self.persistence.get_effect_nodes():
            try:
                node_type = node_types()[effect_node.type]
            except KeyError:
                print(f"Invalid node type {effect_node.type}")
                print(f"Deleting node {effect_node.id}")
                self.persistence.delete_effect_node(effect_node.id)
                continue

            try:
                params = node_type.Params.model_validate_json(effect_node.params)  # type: ignore
            except ValidationError as e:
                print(f"Error loading node params for {effect_node.id}: {e}")
                print(f"Resetting params for node {effect_node.id}")
                params = node_type.Params()  # type: ignore
                continue

            if (existing_node := self.nodes.get(effect_node.id)) is not None:
                existing_node.params = params
            else:
                node = node_type(effect_node.id)
                node.params = params
                self.nodes[node.id] = node
                ed.set_node_position(
                    self.node_ids.get(node.id),
                    imgui.ImVec2(*effect_node.position),
                )

    def links(self) -> dict[int, Link]:
        links = {}
        pins = self.pins()
        for effect_link in self.persistence.get_effect_node_links():
            link = Link(
                PinId(effect_link.start_node_id, effect_link.start_pin_name),
                PinId(effect_link.end_node_id, effect_link.end_pin_name),
            )
            if not self._valid_link(link, links, pins):
                print(f"Invalid link {effect_link.id}, deleting")
                self.persistence.delete_effect_node_link(effect_link.id)
                continue

            links[effect_link.id] = link

        return links

    def pins(self) -> dict[PinId, PinType]:
        pins: dict[PinId, PinType] = {}
        for node in self.nodes.values():
            pins.update(node.pins())
        return pins

    def execute(
        self, context: SceneContext
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]] | None:
        self._load_nodes()
        links = self.links()

        source_nodes = [
            node for node in self.nodes.values() if isinstance(node, SourceNode)
        ]
        destination_nodes = [
            node for node in self.nodes.values() if isinstance(node, DestinationNode)
        ]

        for source_node in source_nodes:
            source_node.set_context(context)

        for dest_node in destination_nodes:
            dest_node.brightness = None

        input_values: dict[PinId, Brightness | SceneContext] = {}

        fulfilled_links: set[Link] = set[Link]()
        links_by_start_node: dict[int, list[Link]] = {}
        for link in links.values():
            links_by_start_node.setdefault(link.start.node_id, []).append(link)

        links_by_end_node: dict[int, list[Link]] = {}
        for link in links.values():
            links_by_end_node.setdefault(link.end.node_id, []).append(link)

        ready_nodes: set[Node] = set(source_nodes)
        while ready_nodes:
            node = ready_nodes.pop()

            try:
                node_input_values = {
                    pin_id: input_values[pin_id]
                    for pin_id, pin_type in node.pins().items()
                    if pin_type.type == "input"
                }
            except KeyError:
                continue

            output_values = node.run(node_input_values)

            for link in links_by_start_node.get(node.id, []):
                output_value = output_values[link.start]
                link_data_type = node.pins()[link.start].data
                match link_data_type:
                    case "brightness":
                        assert isinstance(output_value, Brightness)
                        if (b := input_values.get(link.end)) is not None:
                            assert isinstance(b, Brightness)
                            output_value = combine_brightnesses(output_value, b)
                        input_values[link.end] = output_value
                    case "scene_context":
                        if link.end in input_values:
                            print(
                                "Warning, assigning multiple scene contexts to one pin"
                            )
                        else:
                            input_values[link.end] = output_value

                fulfilled_links.add(link)

                target_node = self.nodes[link.end.node_id]
                all_links_to_target_fulfilled = all(
                    link in fulfilled_links
                    for link in links_by_end_node[target_node.id]
                )
                if all_links_to_target_fulfilled:
                    ready_nodes.add(target_node)

        # Combine brightness from all destination nodes
        destination_brightnesses = [
            dest_node.brightness
            for dest_node in destination_nodes
            if dest_node.brightness is not None
        ]

        if not destination_brightnesses:
            return None

        combined_brightness = combine_brightnesses(*destination_brightnesses)
        original_ids = context.chain_ids()
        final_brightness: np.ndarray[tuple[int], np.dtype[np.floating]] = np.zeros(
            original_ids.shape, dtype=np.float32
        )  # type: ignore
        indices = np.searchsorted(original_ids, combined_brightness.chain_ids)
        final_brightness[indices] = combined_brightness.values
        return final_brightness

    def gui(self):
        try:
            self._load_nodes()
            links = self.links()
            pins = self.pins()

            with ed_ctx.begin("Effect Nodes Editor"):
                for node in self.nodes.values():
                    with ed_ctx.begin_node(self.node_ids.get(node.id)):
                        prev_params = node.params.model_copy(deep=True)
                        node.gui(lambda pin_id: self.pin_ids.get(pin_id))
                        self._persist_params_if_changed(node, prev_params)

                if imgui.is_key_pressed(imgui.Key.delete | imgui.Key.backspace):
                    for link in ed.get_selected_links():
                        ed.delete_link(link)
                    for node in ed.get_selected_nodes():
                        ed.delete_node(node)

                ed.suspend()
                if ed.show_background_context_menu():
                    self.context_window_opened_at = np.array(
                        ed.screen_to_canvas(imgui.get_mouse_pos())
                    )
                    imgui.open_popup("Node Editor Context Menu")

                with imgui_ctx.begin_popup("Node Editor Context Menu") as visible:
                    if visible:
                        with imgui_ctx.begin_menu("New Node"):
                            nodes = node_display_names()
                            for node_type, node_name in nodes.items():
                                if imgui.menu_item(node_name, "", False)[0]:
                                    # Save to database
                                    self.persistence.append_effect_node(
                                        EffectNode(
                                            id=None,
                                            type=node_type,
                                            params=r"{}",
                                            position=self.context_window_opened_at,
                                        )
                                    )

                ed.resume()

                with ed_ctx.begin_create() as begin_create:
                    if begin_create:
                        start, end = ed.PinId(), ed.PinId()
                        if ed.query_new_link(start, end):
                            start = self.pin_ids.get_key(start)
                            end = self.pin_ids.get_key(end)
                            if pins[start].type == "input":
                                start, end = end, start
                            link = Link(start, end)

                            if self._valid_link(link, links, pins):
                                if ed.accept_new_item(imgui.ImVec4(0, 1, 0, 1)):
                                    # Save to database
                                    self.persistence.append_effect_node_link(
                                        EffectNodeLink(
                                            id=None,
                                            start_node_id=start.node_id,
                                            start_pin_name=start.name,
                                            end_node_id=end.node_id,
                                            end_pin_name=end.name,
                                        )
                                    )
                            else:
                                ed.reject_new_item(imgui.ImVec4(1, 0, 0, 1))

                with ed_ctx.begin_delete() as begin_delete:
                    if begin_delete:
                        node_id = ed.NodeId()
                        while ed.query_deleted_node(node_id):
                            node = self.nodes[self.node_ids.get_key(node_id)]
                            if node.deletable:
                                if ed.accept_deleted_item():
                                    # Delete node from database
                                    self.persistence.delete_effect_node(node.id)
                                    # Delete from memory
                                    del self.nodes[node.id]
                            else:
                                ed.reject_deleted_item()

                        link = ed.LinkId()
                        while ed.query_deleted_link(link):
                            db_link_id = self.link_ids.get_key(link)
                            if ed.accept_deleted_item():
                                self.persistence.delete_effect_node_link(db_link_id)

                for link_id, link in links.items():
                    ed.link(
                        self.link_ids.get(link_id),
                        self.pin_ids.get(link.start),
                        self.pin_ids.get(link.end),
                    )

                if self.frame_count == 2:
                    ed.navigate_to_content(0.0)
                self.frame_count += 1

            for node in self.nodes.values():
                effect_node = self.persistence.get_effect_node(node.id)
                if effect_node is None:
                    continue
                current_pos = ed.get_node_position(self.node_ids.get(node.id))
                current_pos = np.array([current_pos.x, current_pos.y])
                if not np.array_equal(effect_node.position, current_pos):
                    effect_node.position = current_pos
                    self.persistence.update_effect_node(effect_node)

        except Exception:
            print("Error in node editor")
            traceback.print_exc()

    def plot_gui(self):
        for node in self.nodes.values():
            prev_params = node.params.model_copy(deep=True)
            node.plot_gui()
            self._persist_params_if_changed(node, prev_params)

    def _valid_link(
        self, link: Link, links: dict[int, Link], pins: dict[PinId, PinType]
    ):
        if (start_type := pins.get(link.start)) is None:
            return False
        if (end_type := pins.get(link.end)) is None:
            return False

        if link.start.node_id not in self.nodes:
            return False
        if link.end.node_id not in self.nodes:
            return False

        if not start_type.type == "output":
            return False
        if not end_type.type == "input":
            return False

        if not start_type.data == end_type.data:
            return False

        if link in links:
            return False

        return True

    def _persist_params_if_changed(self, node: Node, prev_params: BaseModel):
        if node.params == prev_params:
            return

        if (effect_node := self.persistence.get_effect_node(node.id)) is None:
            print(f"Failed to update node {node.id}, it's missing")
            return

        effect_node.params = node.params.model_dump_json()
        self.persistence.update_effect_node(effect_node)
