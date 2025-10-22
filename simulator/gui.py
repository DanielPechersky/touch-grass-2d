import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from imgui_bundle import hello_imgui, imgui, imgui_ctx, implot, portable_file_dialogs
from imgui_bundle.immapp import icons_fontawesome_6 as fa6
from PIL import Image

from simulator.chain_menu import chain_menu
from simulator.gl_texture import GlTexture
from simulator.helpers import (
    implot_draw_rectangle,
    ndarray_to_scatter,
    plot_chain,
    point_to_ndarray,
)
from simulator.measurer import Measurer
from simulator.persistence import Cattail, Chain, Persistence
from simulator.selection import CattailId, ChainId, Selection
from simulator.tools import Tool
from simulator.tools.cattail_placer import CattailPlacer
from simulator.tools.chain_placer import ChainTool
from simulator.tools.move import MoveTool
from simulator.tools.simulator import Simulator


class AddImageGui:
    def __init__(self, persistence: Persistence):
        self.persistence = persistence
        self.image = None
        self.texture = None
        self.measurer = None

    def gui(self) -> tuple[Image.Image, float] | None:
        if self.image is None:
            imgui.text("Select a map image")
            if imgui.button("Open Image"):
                img_path = portable_file_dialogs.open_file(
                    "Select a map image",
                    default_path=".",
                    filters=["Image files", "*.png *.jpg *.jpeg *.bmp"],
                ).result()
                match img_path:
                    case [img_path]:
                        self.image = Image.open(img_path).convert("RGBA")
                    case _:
                        pass
            return

        if self.texture is None:
            self.texture = GlTexture.load_texture_rgba(self.image)

        size = imgui.ImVec2(800, 600)
        with implot_begin_plot(
            "Set the scale by picking two points and inputting the distance between them",
            size,
            flags=implot.Flags_.no_legend | implot.Flags_.equal,
        ):
            implot.setup_axes("x (pixels)", "y (pixels)")

            implot.plot_image(
                "img",
                imgui.ImTextureRef(self.texture.id),
                implot.Point(0.0, 0.0),
                implot.Point(
                    self.texture.w,
                    self.texture.h,
                ),
            )

            if self.measurer is None or imgui.is_key_pressed(imgui.Key.escape):
                self.measurer = Measurer()
            scale = self.measurer.gui()
            if scale is not None:
                return self.image, scale


class InProjectGui:
    def __init__(self, persistence: Persistence, location_id: int, chain_size=5):
        self.texture = None

        self.persistence = persistence
        self.chain_size = chain_size

        self.location_id: int = location_id
        scale = self.persistence.get_scale(location_id)
        assert scale is not None
        self.scale: float = scale

        self.simulator = Simulator(persistence)
        self.update_simulator()
        self.move_tool = MoveTool(persistence)
        self.chain_tool = ChainTool(persistence, chain_size, 2.0 / chain_size)
        self.cattail_placer = CattailPlacer(persistence)

        self.tool: Tool = self.simulator
        self.selection: Selection = set()

        self.box_selecting = False
        self.box_selection_start: (
            np.ndarray[tuple[Literal[2]], np.dtype[np.floating]] | None
        ) = None

    def update_simulator(self):
        self.simulator.set_chains(self.chains)
        self.simulator.set_cattails(self.cattails)
        self.simulator.set_groups(self.persistence.get_groups())

    @property
    def axes_limits(self):
        if self.texture is None:
            return None
        return (
            0,
            self.texture.w / self.scale,
            self.texture.h / self.scale,
            0,
        )

    @property
    def chains(self) -> list[Chain]:
        chains = self.persistence.get_chains()
        assert chains is not None
        return chains

    @property
    def cattails(self) -> list[Cattail]:
        assert self.location_id is not None
        cattails = self.persistence.get_cattails()
        assert cattails is not None
        return cattails

    def sidebar_gui(self):
        with imgui_ctx.begin_child("Sidebar", size=imgui.ImVec2(200, 0)):
            with imgui_ctx.begin_child(
                "Tools",
                child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
            ):
                imgui.separator_text("Tools")
                if imgui.radio_button(
                    f"{fa6.ICON_FA_EYE} View", isinstance(self.tool, Simulator)
                ):
                    self.tool.switched_away()
                    self.update_simulator()
                    self.tool = self.simulator
                if imgui.radio_button(
                    f"{fa6.ICON_FA_ARROWS_UP_DOWN_LEFT_RIGHT} Move",
                    isinstance(self.tool, MoveTool),
                ):
                    self.tool.switched_away()
                    self.tool = self.move_tool
                if imgui.radio_button(
                    f"{fa6.ICON_FA_WHEAT_AWN} Place chain",
                    isinstance(self.tool, ChainTool),
                ):
                    self.tool.switched_away()
                    self.tool = self.chain_tool
                if imgui.radio_button(
                    f"{fa6.ICON_FA_TOWER_BROADCAST} Place cattail",
                    isinstance(self.tool, CattailPlacer),
                ):
                    self.tool.switched_away()
                    self.tool = self.cattail_placer

            self.tool.sidebar_gui()

            with imgui_ctx.begin_child(
                "Chain Menu",
                child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
            ):
                self.selection = chain_menu(self.persistence, self.selection)

    def show_selection(self):
        for chain in self.chains:
            if ChainId(chain.id) in self.selection:
                plot_chain(
                    "selected_chain",
                    chain.points,
                    color=imgui.ImVec4(0.0, 0.0, 1.0, 1.0),
                )
        for cattail in self.cattails:
            if CattailId(cattail.id) in self.selection:
                implot.set_next_marker_style(
                    size=5, fill=imgui.ImVec4(0.0, 0.0, 1.0, 1.0)
                )
                implot.plot_scatter(
                    "selected_cattail", *ndarray_to_scatter(cattail.pos)
                )

    def gui(self):
        if self.texture is None:
            img = self.persistence.get_image(self.location_id)
            assert img is not None
            self.texture = GlTexture.load_texture_rgba(img)

        self.sidebar_gui()

        imgui.same_line()

        size = imgui.ImVec2(800, 600)
        with implot_begin_plot(
            "Exhibit simulator",
            size,
            flags=implot.Flags_.equal
            | implot.Flags_.no_legend
            | implot.Flags_.no_title,
        ):
            axis_flags = (
                implot.AxisFlags_.no_decorations | implot.AxisFlags_.no_highlight
            )
            implot.setup_axes(
                "x (metres)", "y (metres)", x_flags=axis_flags, y_flags=axis_flags
            )

            axes = self.axes_limits
            if axes is not None:
                implot.setup_axes_limits(*axes)

            implot.plot_image(
                "img",
                imgui.ImTextureRef(self.texture.id),
                implot.Point(0.0, 0.0),
                implot.Point(
                    self.texture.w / self.scale,
                    self.texture.h / self.scale,
                ),
            )

            self.show_selection()

            if imgui.is_key_pressed(imgui.Key.escape):
                self.tool.switched_away()

            if (
                imgui.is_window_focused(imgui.FocusedFlags_.root_and_child_windows)
                and imgui.is_key_pressed(imgui.Key.delete | imgui.Key.backspace)
                and len(self.selection) > 0
            ):
                imgui.open_popup("Delete selection")
            with imgui_ctx.begin_popup_modal("Delete selection") as visible:
                if visible:
                    imgui.text("Delete selected items?")
                    num_chains = 0
                    num_cattails = 0
                    selection = self.selection
                    for x in selection:
                        match x:
                            case ChainId():
                                num_chains += 1
                            case CattailId():
                                num_cattails += 1
                    if num_chains != 0:
                        imgui.text(f"Delete {num_chains} chain(s)?")
                    if num_cattails != 0:
                        imgui.text(f"Delete {num_cattails} cattail(s)?")
                    if imgui.button("Cancel"):
                        imgui.close_current_popup()
                    imgui.same_line()
                    if imgui.button("Delete"):
                        for x in selection:
                            match x:
                                case ChainId(id=id):
                                    self.persistence.delete_chain(id)
                                case CattailId(id=id):
                                    self.persistence.delete_cattail(id)
                        self.selection = set()
                        imgui.close_current_popup()

            box_selection_mouse_button = imgui.MouseButton_.left
            if implot.is_plot_hovered() and imgui.is_mouse_clicked(
                box_selection_mouse_button
            ):
                self.selection = set()
                self.box_selection_start = point_to_ndarray(implot.get_plot_mouse_pos())

            if (
                imgui.is_mouse_released(box_selection_mouse_button)
                and self.box_selecting
            ):
                assert self.box_selection_start is not None
                box_selection_end = point_to_ndarray(implot.get_plot_mouse_pos())
                box_selection = np.stack([self.box_selection_start, box_selection_end])
                box_selection_min = box_selection.min(axis=0)
                box_selection_max = box_selection.max(axis=0)

                box_selected_chains = []
                for chain in self.chains:
                    if np.any(
                        np.all((box_selection_min < chain.points), axis=1)
                        & np.all((chain.points < box_selection_max), axis=1)
                    ):
                        box_selected_chains.append(chain.id)

                box_selected_cattails = []
                for cattail in self.cattails:
                    if np.all(
                        (box_selection_min < cattail.pos)
                        & (cattail.pos < box_selection_max)
                    ):
                        box_selected_cattails.append(cattail.id)

                self.selection = set(map(ChainId, box_selected_chains)) | set(
                    map(CattailId, box_selected_cattails)
                )

            if implot.is_plot_hovered() and imgui.is_mouse_dragging(
                box_selection_mouse_button
            ):
                self.box_selecting = True
            else:
                self.box_selecting = False

            if self.box_selecting and self.box_selection_start is not None:
                box_selection_end = point_to_ndarray(implot.get_plot_mouse_pos())
                implot_draw_rectangle(
                    self.box_selection_start, box_selection_end, col=0x40FF0000
                )
            self.move_tool.update_selection(self.selection)

            self.tool.main_gui()


def configure_implot():
    map = implot.get_input_map()
    map.fit = -1
    map.menu = -1
    map.pan = imgui.MouseButton_.right
    map.select_mod = imgui.Key.mod_ctrl


class ProjectPicker:
    def __init__(self):
        self.selected_folder: Path = Path(".").absolute()
        self.new_project_name = ""

    @property
    def project_path(self) -> Path:
        return self.selected_folder / f"{self.new_project_name}.sgp"

    def gui(self) -> Persistence | None:
        imgui.text(
            "No project is opened. Create a new project or open an existing project."
        )
        with imgui_ctx.begin_child(
            "New Project",
            child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
        ):
            imgui.separator_text("New Project")
            set, name = imgui.input_text_with_hint(
                "##New Project Name", "My Project", self.new_project_name
            )
            if set:
                self.new_project_name = name
            imgui.same_line()
            if imgui.button("Choose Folder"):
                d = portable_file_dialogs.select_folder("Choose folder for new project")
                path = d.result()
                if path:
                    self.selected_folder = Path(path)

            imgui.text(f"{self.project_path}")
            imgui.same_line()
            with imgui_disable(self.new_project_name == ""):
                if imgui.button("Create"):
                    return Persistence(str(self.project_path))

        with imgui_ctx.begin_child(
            "Open Existing Project",
            child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
        ):
            imgui.separator_text("Open Existing Project")
            if imgui.button("Open Project"):
                d = portable_file_dialogs.open_file(
                    "Select a project file",
                    default_path=".",
                    filters=["Switchgrass Projects", "*.sgp"],
                )
                match d.result():
                    case [path]:
                        set_last_opened_project(path)
                        return Persistence(path)


def get_last_opened_project() -> str | None:
    p = hello_imgui.load_user_pref("last_opened_project").strip() or None
    return p


def set_last_opened_project(path: str | None):
    hello_imgui.save_user_pref("last_opened_project", path or "")


@contextmanager
def implot_begin_plot(*args, **kwargs):
    if implot.begin_plot(*args, **kwargs):
        try:
            yield
        finally:
            implot.end_plot()


@contextmanager
def imgui_disable(disabled: bool = True):
    imgui.begin_disabled(disabled)
    try:
        yield
    finally:
        imgui.end_disabled()


class Gui:
    def __init__(self):
        self.persistence: Persistence | None = None

        self.project_picker: ProjectPicker | None = None
        self.add_image_gui: AddImageGui | None = None
        self.in_project_gui: InProjectGui | None = None

        self.configured_implot = False

    def menu_bar(self):
        with imgui_ctx.begin_main_menu_bar():
            with imgui_ctx.begin_menu("File"):
                if imgui.menu_item(
                    "Close", "", False, enabled=self.persistence is not None
                )[0]:
                    set_last_opened_project(None)
                    self.persistence = None

    def gui(self):
        try:
            self.menu_bar()

            if not self.configured_implot:
                configure_implot()
                self.configured_implot = True

            if (
                self.persistence is None
                and (last_opened_project := get_last_opened_project()) is not None
            ):
                print(f"Last opened project: {last_opened_project}")
                try:
                    self.persistence = Persistence(last_opened_project)
                    return
                except Exception as e:
                    print(f"Failed to open last project: {e}")
                    set_last_opened_project(None)

            if self.persistence is None:
                self.in_project_gui = None
                if self.project_picker is None:
                    self.project_picker = ProjectPicker()
                self.persistence = self.project_picker.gui()
                return
            else:
                self.project_picker = None

            location_id = self.persistence.get_location_id()
            if location_id is None:
                if self.add_image_gui is None:
                    self.add_image_gui = AddImageGui(self.persistence)
                result = self.add_image_gui.gui()
                if result is not None:
                    img, scale = result
                    self.location_id = self.persistence.create_location(img, scale)
                return
            else:
                self.add_image_gui = None

            if self.in_project_gui is None:
                self.in_project_gui = InProjectGui(self.persistence, location_id)
            self.in_project_gui.gui()
        except Exception:
            print("Error in GUI")
            traceback.print_exc()
