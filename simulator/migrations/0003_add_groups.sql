CREATE TABLE groups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    parent_group_id INTEGER,
    FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE,
    FOREIGN KEY(parent_group_id) REFERENCES groups(id) ON DELETE CASCADE
) STRICT;
CREATE TABLE chains_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    group_id INTEGER,
    points TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE,
    FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
) STRICT;
CREATE TABLE cattails_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    group_id INTEGER,
    x REAL NOT NULL,
    y REAL NOT NULL,
    FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE,
    FOREIGN KEY(group_id) REFERENCES groups(id) ON DELETE CASCADE
) STRICT;

INSERT INTO chains_new (id, project_id, points)
SELECT c.id, c.project_id, c.points
FROM chains c;

INSERT INTO cattails_new (id, project_id, x, y)
SELECT c.id, c.project_id, c.x, c.y
FROM cattails c;

DROP TABLE chains;
DROP TABLE cattails;

ALTER TABLE chains_new RENAME TO chains;
ALTER TABLE cattails_new RENAME TO cattails;
