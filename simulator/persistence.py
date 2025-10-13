from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image

from simulator.migrate_database import migrate_database

type Scale = float
"""
Pixels per metre
"""


@dataclass
class Group[Id]:
    id: Id

    parent_group_id: int | None = None

    @property
    def external_id(self):
        return f"g_{self.id}"


@dataclass
class Chain[Id]:
    id: Id
    points: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]

    group_id: int | None = None

    @property
    def external_id(self):
        return f"c_{self.id}"


@dataclass
class Cattail[Id]:
    id: Id
    pos: np.ndarray[tuple[Literal[2]], np.dtype[np.floating]]

    group_id: int | None = None

    @property
    def external_id(self):
        return f"t_{self.id}"


class Persistence:
    def __init__(self, db: str):
        self.db = db
        # Autocommit mode (isolation_level=None) so individual statements persist immediately
        self.conn = sqlite3.connect(self.db, autocommit=True)
        self.conn.execute("PRAGMA foreign_keys = ON")
        migrate_database(self.conn)

        self.project_id = self.get_or_create_project()

    def get_or_create_project(self) -> int:
        if (project := self.get_project()) is not None:
            return project
        return self.create_project()

    def get_project(self) -> int | None:
        row = self.conn.execute("SELECT id FROM project").fetchone()
        if row is not None:
            return row[0]

    def create_project(self) -> int:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO project DEFAULT VALUES")
        assert cur.lastrowid is not None
        return cur.lastrowid

    def create_location(self, image: Image.Image, scale: Scale) -> int:
        cur = self.conn.cursor()
        img_bytes = BytesIO()
        image.save(img_bytes, format="PNG")
        cur.execute(
            "INSERT INTO location(project_id, image, scale) VALUES(?, ?, ?)",
            (self.project_id, img_bytes.getvalue(), scale),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def get_location_id(self) -> int | None:
        row = self.conn.execute("SELECT id FROM location").fetchone()
        if row is not None:
            return row[0]

    def get_image(self, location_id: int) -> Image.Image | None:
        row = self.conn.execute(
            "SELECT image FROM location WHERE id = ?", (location_id,)
        ).fetchone()
        if row is not None:
            return Image.open(
                BytesIO(row[0]),
            )

    def get_scale(self, location_id: int) -> Scale | None:
        row = self.conn.execute(
            "SELECT scale FROM location WHERE id = ?",
            (location_id,),
        ).fetchone()
        if row is not None:
            return row[0]

    def get_groups(self) -> list[Group[int]]:
        rows = self.conn.execute(
            "SELECT id, parent_group_id FROM groups WHERE project_id = ? ORDER BY id",
            (self.project_id,),
        )
        return [Group(id=row[0], parent_group_id=row[1]) for row in rows]

    def append_group(self, group: Group[None]) -> int:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO groups(project_id, parent_group_id) VALUES(?, ?)",
            (self.project_id, group.parent_group_id),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def update_group(self, group: Group[int]):
        self.conn.execute(
            "UPDATE groups SET parent_group_id = ? WHERE id = ?",
            (group.parent_group_id, group.id),
        )

    def delete_group(self, group_id: int):
        self.conn.execute(
            "DELETE FROM groups WHERE id = ?",
            (group_id,),
        )

    def get_chains(self) -> list[Chain[int]]:
        rows = self.conn.execute(
            "SELECT id, points, group_id FROM chains WHERE project_id = ? ORDER BY id",
            (self.project_id,),
        )
        return [
            Chain(
                id=row[0],
                points=np.array(json.loads(row[1]), dtype=np.float32),
                group_id=row[2],
            )
            for row in rows
        ]

    def update_chain(self, chain: Chain[int]):
        self.conn.execute(
            "UPDATE chains SET points = ?, group_id = ? WHERE id = ?",
            (json.dumps(chain.points.tolist()), chain.group_id, chain.id),
        )

    def delete_chain(self, chain_id: int):
        self.conn.execute(
            "DELETE FROM chains WHERE id = ?",
            (chain_id,),
        )

    def append_chain(self, chain: Chain[None]):
        self.conn.execute(
            "INSERT INTO chains(project_id, points) VALUES(?, ?)",
            (self.project_id, json.dumps(chain.points.tolist())),
        )

    def get_cattails(self) -> list[Cattail[int]]:
        rows = self.conn.execute(
            "SELECT id, x, y, group_id FROM cattails WHERE project_id = ? ORDER BY id",
            (self.project_id,),
        )
        return [
            Cattail(
                id=row[0],
                pos=np.array([row[1], row[2]], dtype=np.float32),
                group_id=row[3],
            )
            for row in rows
        ]

    def update_cattail(self, cattail: Cattail[int]):
        self.conn.execute(
            "UPDATE cattails SET x = ?, y = ?, group_id = ? WHERE id = ?",
            (*cattail.pos.tolist(), cattail.group_id, cattail.id),
        )

    def append_cattail(self, cattail: Cattail[None]):
        self.conn.execute(
            "INSERT INTO cattails(project_id, x, y, group_id) VALUES(?, ?, ?, ?)",
            (self.project_id, *cattail.pos.tolist(), cattail.group_id),
        )

    def close(self):
        self.conn.close()
