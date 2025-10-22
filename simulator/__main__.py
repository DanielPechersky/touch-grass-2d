from imgui_bundle import hello_imgui, immapp
from imgui_bundle import imgui_node_editor as ed

from simulator.gui import Gui

runner_params = hello_imgui.SimpleRunnerParams()
runner_params.gui_function = Gui().gui
runner_params.window_title = "Touch Grass 2D"
runner_params.window_size = (1100, 650)
runner_params = runner_params.to_runner_params()
runner_params.callbacks.default_icon_font = hello_imgui.DefaultIconFont.font_awesome6
runner_params.imgui_window_params.show_menu_bar = True
runner_params.imgui_window_params.show_menu_app = False
runner_params.imgui_window_params.show_menu_view = False

node_editor_config = ed.Config()
node_editor_config.force_window_content_width_to_node_width = True
node_editor_config.settings_file = ""

addons_params = immapp.AddOnsParams()
addons_params.with_node_editor_config = node_editor_config
addons_params.with_implot = True

immapp.run(runner_params, addons_params)
