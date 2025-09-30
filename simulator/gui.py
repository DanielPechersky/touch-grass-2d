from typing import Literal

import numpy as np
from imgui_bundle import imgui, imgui_ctx, implot

from simulator.gl_texture import GlTexture
from simulator.helpers import ndarray_to_scatter_many, plot_chain
from simulator.light_effect import TestLightEffect
from simulator.simulator import Simulator
from simulator.tools.cattail_placer import CattailPlacer
from simulator.tools.chain_placer import ChainPlacer
from simulator.tools.measurer import Measurer
from simulator.typing import Chains


def display_chains(chains: Chains):
    for chain in chains:
        plot_chain("chains", chain, color=imgui.ImVec4(1.0, 1.0, 0.0, 1.0))


def display_cattails(cattails: np.ndarray[tuple[int, Literal[2]]]):
    implot.set_next_marker_style(
        marker=implot.Marker_.square, size=10, fill=(1.0, 0.0, 0.0, 1.0)
    )
    implot.plot_scatter(
        "cattails",
        *ndarray_to_scatter_many(cattails),
    )


def light_effect():
    return TestLightEffect()


class Gui:
    def __init__(self, chain_size=5) -> None:
        self.texture = None
        self.tool: Literal["view", "measure", "place_chain", "place_cattail"] = "view"

        self.cattails: np.ndarray[tuple[int, Literal[2]]] = np.zeros(
            (0, 2), dtype=np.float32
        )
        self.chains: Chains = np.empty((0, chain_size, 2), dtype=np.float32)

        self.simulator: Simulator | None = None
        self.measurer: Measurer | None = None
        self.point_placer: ChainPlacer | None = None
        self.cattail_placer = CattailPlacer()

        self.pixels_per_metre = 1.0
        self.last_pixels_per_metre = self.pixels_per_metre

    @property
    def change_in_scale(self):
        if self.pixels_per_metre == self.last_pixels_per_metre:
            return None
        return self.pixels_per_metre / self.last_pixels_per_metre

    @property
    def chain_size(self):
        return self.chains.shape[1]

    @property
    def axes_limits(self):
        if self.texture is None:
            return None
        return (
            0,
            self.texture.w / self.pixels_per_metre,
            self.texture.h / self.pixels_per_metre,
            0,
        )

    def disable_double_click_to_fit(self):
        map = implot.get_input_map()
        map.fit = -1

    def gui(self):
        implot.create_context()
        self.disable_double_click_to_fit()

        if self.texture is None:
            self.texture = GlTexture.load_texture_rgba("map.png")

        size = imgui.ImVec2(800, 600)
        if implot.begin_plot(
            "Image plot", size, flags=implot.Flags_.no_legend | implot.Flags_.equal
        ):
            implot.setup_axes("x (metres)", "y (metres)")

            cond = (
                imgui.Cond_.once if self.change_in_scale is None else imgui.Cond_.always
            )
            implot.setup_axes_limits(*self.axes_limits, cond=cond)

            implot.plot_image(
                "img",
                imgui.ImTextureRef(self.texture.id),
                implot.Point(0.0, 0.0),
                implot.Point(
                    self.texture.w / self.pixels_per_metre,
                    self.texture.h / self.pixels_per_metre,
                ),
            )

            if self.change_in_scale is not None:
                self.chains /= self.change_in_scale
                self.cattails /= self.change_in_scale

            if self.tool == "view":
                if self.simulator is None:
                    self.simulator = Simulator(
                        chains=self.chains,
                        cattail_centers=self.cattails,
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
                    self.chains = np.concatenate(
                        (self.chains, new_chain[np.newaxis, ...])
                    )
            else:
                self.point_placer = None

            if self.tool == "place_cattail":
                new_cattail = self.cattail_placer.gui()
                if new_cattail is not None:
                    self.cattails = np.concatenate(
                        (self.cattails, new_cattail[np.newaxis, :])
                    )

            self.last_pixels_per_metre = self.pixels_per_metre
            if self.tool == "measure":
                if self.measurer is None or pressed_escape:
                    self.measurer = Measurer()
                pixels_per_metre = self.measurer.gui()
                if pixels_per_metre is not None:
                    self.pixels_per_metre = pixels_per_metre
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
