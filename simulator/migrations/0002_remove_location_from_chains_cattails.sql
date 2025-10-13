CREATE TABLE chains_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    points TEXT NOT NULL,
    FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE
) STRICT;
CREATE TABLE cattails_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE
) STRICT;

INSERT INTO chains_new (id, project_id, points)
SELECT c.id, l.project_id, c.points
FROM chains c
INNER JOIN location l ON c.location_id = l.id;

INSERT INTO cattails_new (id, project_id, x, y)
SELECT c.id, l.project_id, c.x, c.y
FROM cattails c
INNER JOIN location l ON c.location_id = l.id;

DROP TABLE chains;
DROP TABLE cattails;

ALTER TABLE chains_new RENAME TO chains;
ALTER TABLE cattails_new RENAME TO cattails;
