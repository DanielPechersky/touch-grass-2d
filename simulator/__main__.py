from imgui_bundle import hello_imgui

from simulator.gui import Gui

hello_imgui.run(Gui().gui, window_title="Touch Grass 2D", window_size=(1100, 650))
