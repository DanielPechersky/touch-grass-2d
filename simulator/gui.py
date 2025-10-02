from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot, portable_file_dialogs
from PIL import Image

from simulator.gl_texture import GlTexture
from simulator.helpers import ndarray_to_scatter_many, plot_chain
from simulator.light_effect import TestLightEffect
from simulator.persistence import Cattail, Chain, Persistence
from simulator.simulator import Simulator
from simulator.tools.cattail_placer import CattailPlacer
from simulator.tools.chain_placer import ChainPlacer
from simulator.tools.measurer import Measurer


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


def light_effect():
    return TestLightEffect()


class InProjectGui:
    def __init__(self, persistence: Persistence, chain_size=5):
        self.texture = None
        self.tool: Literal["view", "measure", "place_chain", "place_cattail"] = "view"

        self.simulator: Simulator | None = None
        self.measurer: Measurer | None = None
        self.point_placer: ChainPlacer | None = None
        self.cattail_placer = CattailPlacer()

        self.persistence = persistence
        self.chain_size = chain_size

        self.location_id: int | None = self.persistence.get_location_id()

        self.change_in_scale = None

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

    def disable_double_click_to_fit(self):
        map = implot.get_input_map()
        map.fit = -1

    @property
    def scale(self):
        scale = self.persistence.get_scale(self.location_id)
        assert scale is not None
        return scale

    @property
    def chains(self) -> list[Chain]:
        chains = self.persistence.get_chains(self.location_id)
        assert chains is not None
        return chains

    @property
    def cattails(self) -> list[Cattail]:
        cattails = self.persistence.get_cattails(self.location_id)
        assert cattails is not None
        return cattails

    def gui(self):
        implot.create_context()
        self.disable_double_click_to_fit()

        if self.location_id is None:
            imgui.text("Select a map image")
            if imgui.button("Open Image"):
                img_path = portable_file_dialogs.open_file(
                    "Select a map image",
                    default_path=".",
                    filters=["Image files", "*.png *.jpg *.jpeg *.bmp"],
                ).result()
                match img_path:
                    case [img_path]:
                        img = Image.open(img_path).convert("RGBA")
                        self.location_id = self.persistence.create_location(
                            img, scale=1.0
                        )
                    case _:
                        pass
            return

        if self.texture is None:
            img = self.persistence.get_image(self.location_id)
            assert img is not None
            self.texture = GlTexture.load_texture_rgba(img)

        size = imgui.ImVec2(800, 600)
        if implot.begin_plot(
            "Image plot", size, flags=implot.Flags_.no_legend | implot.Flags_.equal
        ):
            implot.setup_axes("x (metres)", "y (metres)")

            cond = (
                imgui.Cond_.once if self.change_in_scale is None else imgui.Cond_.always
            )
            axes = self.axes_limits
            if axes is not None:
                implot.setup_axes_limits(*axes, cond=cond)

            implot.plot_image(
                "img",
                imgui.ImTextureRef(self.texture.id),
                implot.Point(0.0, 0.0),
                implot.Point(
                    self.texture.w / self.scale,
                    self.texture.h / self.scale,
                ),
            )

            if self.change_in_scale is not None:
                self.persistence.set_scale_and_rescale(
                    self.location_id, self.change_in_scale
                )

            if self.tool == "view":
                if self.simulator is None:
                    self.simulator = Simulator(
                        chains=self.chains,
                        cattails=self.cattails,
                        light_effect=light_effect(),
                    )
                self.simulator.gui(imgui.get_io().delta_time)
            else:
                self.simulator = None

                display_chains(self.chains)
                display_cattails(self.cattails)

            pressed_escape = imgui.is_key_pressed(imgui.Key.escape)

            if self.tool == "place_chain":
                if self.point_placer is None or pressed_escape:
                    self.point_placer = ChainPlacer(
                        chain_size=self.chain_size, spacing=2.5 / self.chain_size
                    )
                new_chain = self.point_placer.gui()
                if new_chain is not None:
                    new_chain = Chain(id=None, points=new_chain)
                    self.persistence.append_chain(self.location_id, new_chain)
            else:
                self.point_placer = None

            if self.tool == "place_cattail":
                new_cattail = self.cattail_placer.gui()
                if new_cattail is not None:
                    new_cattail = Cattail(id=None, pos=new_cattail)
                    self.persistence.append_cattail(self.location_id, new_cattail)

            self.change_in_scale = None
            if self.tool == "measure":
                if self.measurer is None or pressed_escape:
                    self.measurer = Measurer()
                new_scale = self.measurer.gui()
                if new_scale is not None:
                    self.change_in_scale = new_scale / self.scale
                    self.persistence.set_scale_and_rescale(self.location_id, new_scale)
            else:
                self.measurer = None

            implot.end_plot()

        with imgui_ctx.begin("Tools"):
            if imgui.radio_button("View", self.tool == "view"):
                self.tool = "view"
            if imgui.radio_button("Measure", self.tool == "measure"):
                self.tool = "measure"
            if imgui.radio_button("Place chain", self.tool == "place_chain"):
                self.tool = "place_chain"
            if imgui.radio_button("Place cattail", self.tool == "place_cattail"):
                self.tool = "place_cattail"


class ProjectPicker:
    def __init__(self):
        self.selected_folder: Path = Path(".").absolute()
        self.new_project_name = ""

    @property
    def project_path(self) -> Path:
        return self.selected_folder / f"{self.new_project_name}.sgp"

    def gui(self) -> Persistence | None:
        imgui.separator_text("New Project")
        set, name = imgui.input_text_with_hint(
            "##", "My Project", self.new_project_name
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

        imgui.separator_text("Open Existing Project")
        if imgui.button("Open Project"):
            d = portable_file_dialogs.open_file(
                "Select a project file",
                default_path=".",
                filters=["Switchgrass Projects", "*.sgp"],
            )
            match d.result():
                case [path]:
                    return Persistence(path)


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
        self.in_project_gui: InProjectGui | None = None

    def gui(self):
        if self.persistence is None:
            self.in_project_gui = None
            if self.project_picker is None:
                self.project_picker = ProjectPicker()
            self.persistence = self.project_picker.gui()
        else:
            self.project_picker = None
            if self.in_project_gui is None:
                self.in_project_gui = InProjectGui(self.persistence)
            self.in_project_gui.gui()
