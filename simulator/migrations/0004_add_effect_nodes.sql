CREATE TABLE effect_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    type TEXT NOT NULL,
    params TEXT NOT NULL,
    position TEXT NOT NULL,
    UNIQUE(project_id, id),
    FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE
) STRICT;

CREATE TABLE effect_node_links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start_node_id INTEGER NOT NULL,
    start_pin_name TEXT NOT NULL,
    end_node_id INTEGER NOT NULL,
    end_pin_name TEXT NOT NULL,
    UNIQUE(start_node_id, start_pin_name, end_node_id, end_pin_name),
    FOREIGN KEY(start_node_id) REFERENCES effect_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(end_node_id) REFERENCES effect_nodes(id) ON DELETE CASCADE
)
