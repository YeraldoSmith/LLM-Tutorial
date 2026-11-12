"""Settings and learning progress, stored as JSON in the user's data folder."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def data_dir(app_id: str) -> Path:
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    path = base / app_id
    path.mkdir(parents=True, exist_ok=True)
    return path


class JsonFile:
    def __init__(self, path: Path, default: dict):
        self.path = path
        self.data = dict(default)
        try:
            self.data.update(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


class Progress(JsonFile):
    def __init__(self, folder: Path):
        super().__init__(folder / "progress.json", {"lessons": {}, "exercises": {}, "last": "", "drafts": {}})

    def lesson_done(self, key: str) -> bool:
        return key in self.data["lessons"]

    def set_lesson(self, key: str, done: bool = True):
        if done:
            self.data["lessons"][key] = int(time.time())
        else:
            self.data["lessons"].pop(key, None)
        self.save()

    def exercise_done(self, key: str) -> bool:
        return bool(self.data["exercises"].get(key))

    def set_exercise(self, key: str):
        self.data["exercises"][key] = int(time.time())
        self.save()

    def draft(self, key: str) -> str | None:
        return self.data["drafts"].get(key)

    def set_draft(self, key: str, code: str):
        self.data["drafts"][key] = code
        self.save()
