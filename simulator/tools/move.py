import numpy as np
from imgui_bundle import imgui, implot

from simulator.helpers import (
    display_cattails,
    display_chains,
    implot_draw_rectangle,
    pixel_line_length_to_plot_line_length,
)
from simulator.persistence import Cattail, Chain, Persistence
from simulator.selection import CattailId, ChainId, Selection
from simulator.tools import Tool


class MoveTool(Tool):
    def __init__(self, persistence: Persistence):
        self.persistence = persistence
        self.selection: Selection = set()

    def update_selection(self, selection: Selection):
        self.selection = selection

    def main_gui(self):
        # Always display all items
        display_cattails(self.persistence.get_cattails())
        display_chains(self.persistence.get_chains())

        if not self.selection:
            return

        # Get selected items and calculate bounding box
        selected_cattails = []
        selected_chains = []
        all_points = []

        # Get all cattails and chains
        all_cattails = self.persistence.get_cattails()
        all_chains = self.persistence.get_chains()

        # Filter selected items and collect their positions
        for item_id in self.selection:
            if isinstance(item_id, CattailId):
                cattail = next((c for c in all_cattails if c.id == item_id.id), None)
                if cattail:
                    selected_cattails.append(cattail)
                    all_points.append(cattail.pos)
            elif isinstance(item_id, ChainId):
                chain = next((c for c in all_chains if c.id == item_id.id), None)
                if chain:
                    selected_chains.append(chain)
                    all_points.extend(chain.points)

        # Highlight selected items
        if selected_cattails:
            # Draw selected cattails with a different style
            implot.set_next_marker_style(
                marker=implot.Marker_.square,
                size=12,
                fill=imgui.ImVec4(0.0, 1.0, 0.0, 0.7),
                outline=imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                weight=2.0,
            )
            cattail_positions = np.stack([cattail.pos for cattail in selected_cattails])
            from simulator.helpers import ndarray_to_scatter_many

            implot.plot_scatter(
                "selected_cattails",
                *ndarray_to_scatter_many(cattail_positions),
            )

        if selected_chains:
            # Draw selected chains with a different color
            from simulator.helpers import plot_chain

            for chain in selected_chains:
                plot_chain(
                    f"selected_chain_{chain.id}",
                    chain.points,
                    color=imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
                )

        if not all_points:
            return

        # Convert to numpy array and calculate bounding box
        points_array = np.array(all_points)
        bbox_center = points_array.mean(axis=0)

        # Calculate bounding rectangle corners
        min_coords = points_array.min(axis=0)
        max_coords = points_array.max(axis=0)

        extra_rectangle_spacing = pixel_line_length_to_plot_line_length(10).x

        # Draw bounding rectangle around all selected points
        implot_draw_rectangle(
            min_coords - extra_rectangle_spacing,
            max_coords + extra_rectangle_spacing,
            col=0x4000FF00,  # Semi-transparent green
        )

        # Draw draggable point at bounding box center
        coords_changed, new_x, new_y, clicked, hovered, held = implot.drag_point(
            -1,  # Use -1 as a unique ID for the move handle
            *bbox_center.tolist(),
            col=imgui.ImVec4(0.0, 1.0, 0.0, 1.0),
            size=8,
            out_clicked=True,
            out_hovered=True,
            held=True,
        )

        # If the drag point moved, update all selected items
        if coords_changed:
            new_center = np.array([new_x, new_y], dtype=np.float32)
            offset = new_center - bbox_center

            # Update cattails
            for cattail in selected_cattails:
                updated_cattail = Cattail(
                    id=cattail.id,
                    pos=cattail.pos + offset,
                    group_id=cattail.group_id,
                )
                self.persistence.update_cattail(updated_cattail)

            # Update chains
            for chain in selected_chains:
                updated_chain = Chain(
                    id=chain.id,
                    points=chain.points + offset,
                    group_id=chain.group_id,
                )
                self.persistence.update_chain(updated_chain)

    def switched_away(self):
        self.selection = set()
