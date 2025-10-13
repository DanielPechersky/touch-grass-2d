import traceback
from contextlib import contextmanager
from pathlib import Path

from imgui_bundle import hello_imgui, imgui, imgui_ctx, implot, portable_file_dialogs
from PIL import Image

from simulator.chain_menu import chain_menu
from simulator.gl_texture import GlTexture
from simulator.measurer import Measurer
from simulator.persistence import Cattail, Chain, Persistence
from simulator.selection import Selection
from simulator.tools import Tool
from simulator.tools.cattail_placer import CattailPlacer
from simulator.tools.chain_placer import ChainTool
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

        self.simulator = Simulator()
        self.update_simulator()
        self.chain_tool = ChainTool(persistence, chain_size, 2.5 / chain_size)
        self.cattail_placer = CattailPlacer(persistence)

        self.tool: Tool = self.simulator
        self.selection = Selection(set())

    def update_simulator(self):
        self.simulator.set_chains(self.chains)
        self.simulator.set_cattails(self.cattails)

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
                if imgui.radio_button("View", isinstance(self.tool, Simulator)):
                    self.tool.switched_away()
                    self.update_simulator()
                    self.tool = self.simulator
                if imgui.radio_button("Place chain", isinstance(self.tool, ChainTool)):
                    self.tool.switched_away()
                    self.tool = self.chain_tool
                if imgui.radio_button(
                    "Place cattail", isinstance(self.tool, CattailPlacer)
                ):
                    self.tool.switched_away()
                    self.tool = self.cattail_placer

            self.tool.sidebar_gui()

            with imgui_ctx.begin_child(
                "Chain Menu",
                child_flags=imgui.ChildFlags_.borders | imgui.ChildFlags_.auto_resize_y,
            ):
                self.selection = chain_menu(self.persistence, self.selection)

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
            flags=implot.Flags_.no_legend | implot.Flags_.equal,
        ):
            implot.setup_axes("x (metres)", "y (metres)")

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

            if imgui.is_key_pressed(imgui.Key.escape):
                self.tool.switched_away()

            self.tool.main_gui()


def disable_double_click_to_fit():
    map = implot.get_input_map()
    map.fit = -1


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

        self.created_context = False

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

            if not self.created_context:
                implot.create_context()
                disable_double_click_to_fit()
                self.created_context = True

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
