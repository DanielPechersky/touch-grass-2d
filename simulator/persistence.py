from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from io import BytesIO
from typing import Literal

import numpy as np
from PIL import Image

type Scale = float
"""
Pixels per metre
"""


@dataclass
class Chain[Id]:
    id: Id
    points: np.ndarray[tuple[int, Literal[2]], np.dtype[np.floating]]


@dataclass
class Cattail[Id]:
    id: Id
    pos: np.ndarray[tuple[Literal[2]], np.dtype[np.floating]]


class Persistence:
    def __init__(self, db: str):
        self.db = db
        # Autocommit mode (isolation_level=None) so individual statements persist immediately
        self.conn = sqlite3.connect(self.db, autocommit=True)
        self.init_db()

        self.project_id = self.get_or_create_project()

    def init_db(self):
        self.conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS project (
                id INTEGER PRIMARY KEY AUTOINCREMENT
            ) STRICT;
            CREATE TABLE IF NOT EXISTS location (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                image BLOB NOT NULL,
                -- scale is in terms of pixels per metre
                scale REAL NOT NULL,
                FOREIGN KEY(project_id) REFERENCES project(id) ON DELETE CASCADE
            ) STRICT;
            CREATE TABLE IF NOT EXISTS chains (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id INTEGER NOT NULL,
                points TEXT NOT NULL,
                FOREIGN KEY(location_id) REFERENCES location(id) ON DELETE CASCADE
            ) STRICT;
            CREATE TABLE IF NOT EXISTS cattails (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location_id INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                FOREIGN KEY(location_id) REFERENCES location(id) ON DELETE CASCADE
            ) STRICT;
            """
        )

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

    def get_chains(self, location_id: int) -> list[Chain[int]]:
        rows = self.conn.execute(
            "SELECT id, points FROM chains WHERE location_id = ? ORDER BY id",
            (location_id,),
        )
        return [
            Chain(id=row[0], points=np.array(json.loads(row[1]), dtype=np.float32))
            for row in rows
        ]

    def update_chain(self, chain: Chain[int]):
        self.conn.execute(
            "UPDATE chains SET points = ? WHERE id = ?",
            (json.dumps(chain.points.tolist()), chain.id),
        )

    def delete_chain(self, chain_id: int):
        self.conn.execute(
            "DELETE FROM chains WHERE id = ?",
            (chain_id,),
        )

    def append_chain(self, location_id: int, chain: Chain[None]):
        self.conn.execute(
            "INSERT INTO chains(location_id, points) VALUES(?, ?)",
            (location_id, json.dumps(chain.points.tolist())),
        )

    def get_cattails(self, location_id: int) -> list[Cattail[int]]:
        rows = self.conn.execute(
            "SELECT id, x, y FROM cattails WHERE location_id = ? ORDER BY id",
            (location_id,),
        )
        return [
            Cattail(id=row[0], pos=np.array([row[1], row[2]], dtype=np.float32))
            for row in rows
        ]

    def update_cattail(self, cattail: Cattail[int]):
        self.conn.execute(
            "UPDATE cattails SET x = ?, y = ? WHERE id = ?",
            (*cattail.pos.tolist(), cattail.id),
        )

    def append_cattail(self, location_id: int, cattail: Cattail[None]):
        self.conn.execute(
            "INSERT INTO cattails(location_id, x, y) VALUES(?, ?, ?)",
            (location_id, *cattail.pos.tolist()),
        )

    def close(self):
        self.conn.close()
