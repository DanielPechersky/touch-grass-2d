from imgui_bundle import hello_imgui

from simulator.gui import Gui

runner_params = hello_imgui.SimpleRunnerParams()
runner_params.gui_function = Gui().gui
runner_params.window_title = "Touch Grass 2D"
runner_params.window_size = (1100, 650)
runner_params = runner_params.to_runner_params()
runner_params.imgui_window_params.show_menu_bar = True
runner_params.imgui_window_params.show_menu_app = False
runner_params.imgui_window_params.show_menu_view = False

hello_imgui.run(runner_params)
