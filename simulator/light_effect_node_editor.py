import traceback
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol

import numpy as np
from imgui_bundle import imgui, imgui_ctx
from imgui_bundle import imgui_node_editor as ed
from imgui_bundle import imgui_node_editor_ctx as ed_ctx

from simulator.light_effect import (
    CattailContext,
    LightEffect,
    ProjectileLightEffect,
    PulseLightEffect,
    PulseLightEffect2,
    TestLightEffect,
)
from simulator.persistence import Chain


@dataclass
class SceneContext:
    chains: list[Chain[Any]]
    cattail_context: CattailContext


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
            symbol = "B"
            color = imgui.ImVec4(1.0, 1.0, 0.0, 1.0)
            hover_text = "Brightness Values"
        case "scene_context":
            symbol = "C"
            color = imgui.ImVec4(1.0, 0.0, 0.0, 1.0)
            hover_text = "Scene Context (Cattails, Chains, and Groups)"
    match type.type:
        case "input":
            text = f"-> {symbol}"
        case "output":
            text = f"{symbol} ->"
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

    imgui.same_line()

    for pin_id in output_pins:
        draw_pin(pin_ids, pin_id, pins[pin_id])


class Node:
    def __init__(self, id: int):
        self.id = id

    @classmethod
    def type(cls) -> str:
        return cls.__name__

    @property
    def deletable(self) -> bool:
        return True

    def pins(self) -> dict[PinId, PinType]:
        raise NotImplementedError

    def gui(self, pin_ids: Callable[[PinId], ed.PinId]) -> None:
        raise NotImplementedError

    def plot_gui(self) -> None:
        pass

    def run(self, inputs: dict[PinId, list[Any]]) -> dict[PinId, Any]:
        raise NotImplementedError


class SourceNode(Node):
    def __init__(self, id: int):
        super().__init__(id)
        self.context: SceneContext | None = None

        self.output_pin = PinId(self.id, "output")

    @property
    def deletable(self) -> bool:
        return False

    def set_context(self, context: SceneContext):
        self.context = context

    def pins(self):
        return {self.output_pin: PinType("output", "scene_context")}

    def gui(self, pin_ids):
        imgui.text("Source")
        draw_pins(pin_ids, self.pins())

    def run(self, inputs) -> dict[PinId, Any]:
        assert self.context is not None
        _ = inputs
        return {self.output_pin: self.context}


class DestinationNode(Node):
    def __init__(self, id: int):
        super().__init__(id)
        self.brightness: np.ndarray | None = None

        self.input_pin = PinId(self.id, "input")

    @property
    def deletable(self) -> bool:
        return False

    def pins(self):
        return {self.input_pin: PinType("input", "brightness")}

    def gui(self, pin_ids):
        imgui.text("Destination")
        draw_pins(pin_ids, self.pins())

    def run(self, inputs):
        self.brightness = np.clip(sum(inputs[self.input_pin]), 0.0, 1.0)
        return {}


class AlwaysBrightNode(Node):
    def __init__(self, id: int):
        super().__init__(id)

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Always Bright")
        draw_pins(pin_ids, self.pins())

    def run(self, inputs) -> dict[PinId, Any]:
        context: SceneContext = inputs[self.input_pin][0]
        return {self.output_pin: np.ones((len(context.chains),), dtype=np.float32)}


class InvertNode(Node):
    def __init__(self, id: int):
        super().__init__(id)

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
        brightness: np.ndarray = inputs[self.input_pin][0]
        return {self.output_pin: 1.0 - brightness}


class DimNode(Node):
    def __init__(self, id: int):
        super().__init__(id)

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")
        self.brightness = 0.5

    def pins(self):
        return {
            self.input_pin: PinType("input", "brightness"),
            self.output_pin: PinType("output", "brightness"),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Dim")
        imgui.set_next_item_width(100)
        set, value = imgui.slider_float(
            f"##{self.id}Brightness", self.brightness, 0.0, 1.0
        )
        if set:
            self.brightness = value

        draw_pins(pin_ids, self.pins())

    def run(self, inputs) -> dict[PinId, Any]:
        brightness: np.ndarray = np.clip(sum(inputs[self.input_pin]), 0.0, 1.0)
        return {self.output_pin: brightness * self.brightness}


class LightEffectNode(Node):
    def __init__(self, id: int):
        super().__init__(id)

        self.input_pin = PinId(self.id, "input")
        self.output_pin = PinId(self.id, "output")

        self.selected_light_effect_name = None
        self.selected_light_effect: LightEffect | None = None
        self.debug_gui_enabled = False

    def pins(self):
        return {
            self.input_pin: PinType("input", "scene_context"),
            self.output_pin: PinType("output", "brightness"),
        }

    @property
    def light_effects(self):
        return {
            "Pulse": PulseLightEffect(),
            "Pulse 2": PulseLightEffect2(),
            "Projectile": ProjectileLightEffect(),
            "Test": TestLightEffect(),
        }

    def gui(self, pin_ids) -> None:
        imgui.text("Light Effect")
        for name in self.light_effects.keys():
            if imgui.radio_button(
                f"{name}##{self.id}{name}", self.selected_light_effect_name == name
            ):
                self.selected_light_effect_name = name
                self.selected_light_effect = self.light_effects[name]
        set, value = imgui.checkbox(
            f"Debug GUI##{self.id}Debug Gui",
            self.debug_gui_enabled,
        )
        if set:
            self.debug_gui_enabled = value
        draw_pins(pin_ids, self.pins())

    def plot_gui(self) -> None:
        if self.debug_gui_enabled and self.selected_light_effect is not None:
            self.selected_light_effect.debug_gui()

    def run(self, inputs) -> dict[PinId, Any]:
        context: SceneContext = inputs[self.input_pin][0]
        if self.selected_light_effect is None:
            brightness = np.zeros((len(context.chains),), dtype=np.float32)
        else:
            brightness = self.selected_light_effect.calculate_chain_brightness(
                imgui.get_io().delta_time, context.chains, context.cattail_context
            )
        return {self.output_pin: brightness}


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


def addable_node_types(id: int) -> dict[str, Node]:
    return {
        "Always Bright": AlwaysBrightNode(id),
        "Invert": InvertNode(id),
        "Light Effect": LightEffectNode(id),
        "Dim": DimNode(id),
    }


class Editor:
    def __init__(self):
        self.source_node = SourceNode(id=1)
        self.destination_node = DestinationNode(id=2)
        nodes = [
            self.source_node,
            self.destination_node,
        ]
        self.nodes: dict[int, Node] = {node.id: node for node in nodes}
        self.links: list[Link] = []
        self.link_ids = EditorIdGenerator[Link, ed.LinkId](
            start_editor_id=10_000, wrapper=ed.LinkId
        )
        self.pin_ids = EditorIdGenerator[PinId, ed.PinId](
            start_editor_id=20_000, wrapper=ed.PinId
        )

    def pins(self) -> dict[PinId, PinType]:
        pins: dict[PinId, PinType] = {}
        for node in self.nodes.values():
            pins.update(node.pins())
        return pins

    def execute(
        self, context: SceneContext
    ) -> np.ndarray[tuple[int], np.dtype[np.floating]] | None:
        self.source_node.set_context(context)
        self.destination_node.brightness = None

        ready_nodes: set[Node] = {self.source_node}

        input_values: dict[PinId, list[Any]] = {}

        fulfilled_links: set[Link] = set[Link]()
        links_by_start_node: dict[int, list[Link]] = {}
        for link in self.links:
            links_by_start_node.setdefault(link.start.node_id, []).append(link)

        links_by_end_node: dict[int, list[Link]] = {}
        for link in self.links:
            links_by_end_node.setdefault(link.end.node_id, []).append(link)

        while ready_nodes:
            node = ready_nodes.pop()

            node_input_values = {
                pin_id: input_values.get(pin_id, []) for pin_id in node.pins()
            }
            output_values = node.run(node_input_values)

            for link in links_by_start_node.get(node.id, []):
                output_value = output_values[link.start]
                input_values.setdefault(link.end, []).append(output_value)
                fulfilled_links.add(link)

                target_node = self.nodes[link.end.node_id]
                all_links_to_target_fulfilled = all(
                    link in fulfilled_links
                    for link in links_by_end_node[target_node.id]
                )
                if all_links_to_target_fulfilled:
                    ready_nodes.add(target_node)

        brightness = self.destination_node.brightness
        return brightness

    def gui(self):
        try:
            pins = self.pins()

            with ed_ctx.begin("Effect Nodes Editor"):
                for node in self.nodes.values():
                    with ed_ctx.begin_node(ed.NodeId(node.id)):
                        node.gui(lambda pin_id: self.pin_ids.get(pin_id))

                if imgui.is_key_pressed(imgui.Key.delete | imgui.Key.backspace):
                    for link in ed.get_selected_links():
                        ed.delete_link(link)
                    for node in ed.get_selected_nodes():
                        ed.delete_node(node)

                ed.suspend()
                if ed.show_background_context_menu():
                    imgui.open_popup("Node Editor Context Menu")

                with imgui_ctx.begin_popup("Node Editor Context Menu") as visible:
                    if visible:
                        with imgui_ctx.begin_menu("New Node"):
                            nodes = addable_node_types(id=max(self.nodes.keys()) + 1)
                            for node_name, node in nodes.items():
                                if imgui.menu_item(node_name, "", False)[0]:
                                    self.nodes[node.id] = node

                ed.resume()

                with ed_ctx.begin_create() as begin_create:
                    if begin_create:
                        start, end = ed.PinId(), ed.PinId()
                        if ed.query_new_link(start, end):
                            start = self.pin_ids.get_key(start)
                            end = self.pin_ids.get_key(end)
                            start_type = pins[start]
                            end_type = pins[end]

                            matches_input_output = {start_type.type, end_type.type} == {
                                "input",
                                "output",
                            }

                            matches_data = start_type.data == end_type.data

                            no_existing_link = all(
                                not (
                                    (link.start == start and link.end == end)
                                    or (link.start == end and link.end == start)
                                )
                                for link in self.links
                            )

                            if (
                                matches_input_output
                                and matches_data
                                and no_existing_link
                            ):
                                if ed.accept_new_item(imgui.ImVec4(0, 1, 0, 1)):
                                    self.links.append(
                                        Link(
                                            start=start,
                                            end=end,
                                        )
                                    )
                            else:
                                ed.reject_new_item(imgui.ImVec4(1, 0, 0, 1))

                with ed_ctx.begin_delete() as begin_delete:
                    if begin_delete:
                        node_id = ed.NodeId()
                        while ed.query_deleted_node(node_id):
                            node = self.nodes[node_id.id()]
                            if node.deletable:
                                if ed.accept_deleted_item():
                                    del self.nodes[node_id.id()]
                                    for link in self.links:
                                        self.links = [
                                            link
                                            for link in self.links
                                            if link.start.node_id != node.id
                                            and link.end.node_id != node.id
                                        ]
                            else:
                                ed.reject_deleted_item()

                        link = ed.LinkId()
                        while ed.query_deleted_link(link):
                            link_id = self.link_ids.get_key(link)
                            if ed.accept_deleted_item():
                                self.links = [
                                    link for link in self.links if link != link_id
                                ]

                for link in self.links:
                    ed.link(
                        self.link_ids.get(link),
                        self.pin_ids.get(link.start),
                        self.pin_ids.get(link.end),
                    )

        except Exception:
            print("Error in node editor")
            traceback.print_exc()

    def plot_gui(self):
        for node in self.nodes.values():
            node.plot_gui()
