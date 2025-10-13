from typing import Generator

from imgui_bundle import imgui

from simulator.persistence import Cattail, Chain, Group, Persistence
from simulator.selection import CattailId, ChainId, Selection, id_for_item

type GroupChild = Group[int] | Chain[int] | Cattail[int]
type ChildrenTree = dict[int | None, list[GroupChild]]


def chain_menu(persistence: Persistence, selection: Selection) -> Selection:
    return ChainMenu(persistence, selection).gui()


class ChainMenu:
    def __init__(self, persistence: Persistence, selection: Selection):
        self.persistence = persistence
        self.selection: Selection = selection
        self.chains = persistence.get_chains()
        self.cattails = persistence.get_cattails()
        self.groups = persistence.get_groups()
        self.grouped_children = group_children(self.groups, self.chains, self.cattails)
        self.flattened_children = list(flatten_tree(self.grouped_children))

        self.groups_to_add = []
        self.groups_to_delete = []

    def gui(self) -> Selection:
        imgui.separator_text("Chains & Cattails")

        ms_io = imgui.begin_multi_select(
            imgui.MultiSelectFlags_.none,
            selection_size=len(self.selection),
            items_count=len(self.flattened_children),
        )

        self.handle_selection_requests(ms_io.requests)

        self.render_group(None)

        ms_io = imgui.end_multi_select()
        self.handle_selection_requests(ms_io.requests)

        for group in self.groups_to_add:
            group.id = self.persistence.append_group(group)
            self.groups.append(group)
        for group_id in self.groups_to_delete:
            self.persistence.delete_group(group_id)

        return self.selection

    def render_group(self, group: Group[int] | None, selectable_idx=0) -> int:
        group_id = group.id if group else None
        display = f"{group.external_id}" if group else "Root"

        if group is None:
            imgui.set_next_item_open(True, imgui.Cond_.always)
        expanded = imgui.tree_node(display)
        if imgui.begin_popup_context_item(f"group_context_menu_{display}"):
            if imgui.menu_item("Create New Group", "", False)[0]:
                self.groups_to_add.append(Group(id=None, parent_group_id=group_id))
                imgui.close_current_popup()
            if group is not None:
                if imgui.menu_item("Delete Group", "", False)[0]:
                    self.groups_to_delete.append(group.id)
                    imgui.close_current_popup()
            imgui.end_popup()
        if group is not None and imgui.begin_drag_drop_source():
            imgui.set_drag_drop_payload_py_id("group_id", group.id)
            imgui.text(f"Dragging group {group.external_id}")
            imgui.end_drag_drop_source()
        if imgui.begin_drag_drop_target():
            if (
                selection_id := imgui.accept_drag_drop_payload_py_id("selection_id")
            ) is not None:
                selection_id = selection_id.data_id
                assert selection_id == 0
                for chain in self.chains:
                    if ChainId(chain.id) not in self.selection:
                        continue
                    chain.group_id = group_id
                    self.persistence.update_chain(chain)
                for cattail in self.cattails:
                    if CattailId(cattail.id) not in self.selection:
                        continue
                    cattail.group_id = group_id
                    self.persistence.update_cattail(cattail)
            if (
                dragged_group_id := imgui.accept_drag_drop_payload_py_id("group_id")
            ) is not None:
                dragged_group_id = dragged_group_id.data_id
                for group in self.groups:
                    if group.id == dragged_group_id:
                        group.parent_group_id = group_id
                        self.persistence.update_group(group)
                        break

            imgui.end_drag_drop_target()
        if expanded:
            selectable_idx = self.render_children(
                parent_group=group_id,
                selectable_idx=selectable_idx,
            )
            imgui.tree_pop()
        else:
            selectable_idx += self.count_items_in_group(group_id)

        return selectable_idx

    def render_children(self, parent_group: int | None = None, selectable_idx=0) -> int:
        def set_next_item_selection():
            nonlocal selectable_idx
            imgui.set_next_item_selection_user_data(selectable_idx)
            selectable_idx += 1

        for child in self.grouped_children[parent_group]:
            match child:
                case Group() as group:
                    selectable_idx = self.render_group(group, selectable_idx)

                case Cattail() | Chain() as child:
                    set_next_item_selection()
                    imgui.selectable(
                        f"{child.external_id}", id_for_item(child) in self.selection
                    )
                    if imgui.begin_drag_drop_source():
                        imgui.set_drag_drop_payload_py_id("selection_id", 0)
                        imgui.text(f"Dragging {len(self.selection)} items")
                        imgui.end_drag_drop_source()

        return selectable_idx

    def handle_selection_requests(
        self,
        requests: imgui.ImVector_SelectionRequest,
    ):
        for request in requests:
            if request.type == imgui.SelectionRequestType.set_all:
                if request.selected:
                    self.selection = Selection(
                        {CattailId(cattail.id) for cattail in self.cattails}
                        | {ChainId(chain.id) for chain in self.chains},
                    )
                else:
                    self.selection = Selection(set())
            elif request.type == imgui.SelectionRequestType.set_range:
                matching_items = self.flattened_children[
                    request.range_first_item : request.range_last_item + 1
                ]
                matching_items = [id_for_item(i) for i in matching_items]
                if request.selected:
                    self.selection.update(matching_items)
                else:
                    self.selection.difference_update(matching_items)

    def count_items_in_group(self, group_id: int | None):
        count = 0
        for child in self.grouped_children[group_id]:
            match child:
                case Group(id=id):
                    count += self.count_items_in_group(id)
                case Chain() | Cattail():
                    count += 1
        return count


def group_children(
    groups: list[Group[int]], chains: list[Chain[int]], cattails: list[Cattail[int]]
) -> ChildrenTree:
    children: ChildrenTree = {g.id: [] for g in groups}
    children[None] = []

    for group in groups:
        children[group.parent_group_id].append(group)

    for cattail in cattails:
        children[cattail.group_id].append(cattail)

    for chain in chains:
        children[chain.group_id].append(chain)

    return children


def flatten_tree(
    children: ChildrenTree, parent_group: int | None = None
) -> Generator[Chain[int] | Cattail[int]]:
    for child in children[parent_group]:
        match child:
            case Group(id=id):
                yield from flatten_tree(children, id)
            case Chain() | Cattail() as child:
                yield child
