"""Download the newest course content from GitHub (keeps the app in sync with the repo)."""
from __future__ import annotations

import io
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path


def download_content(repo: str, dest_root: Path, timeout: float = 120) -> Path:
    """Fetch https://github.com/<repo>/archive/HEAD.zip and unpack tutorial.json + courses/.

    Returns the folder that now holds the content. Older downloads are removed.
    """
    url = f"https://github.com/{repo}/archive/HEAD.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "tutorial-app"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    archive = zipfile.ZipFile(io.BytesIO(data))
    names = archive.namelist()
    if not names:
        raise RuntimeError("empty archive")
    top = names[0].split("/")[0] + "/"
    if top + "tutorial.json" not in names:
        raise RuntimeError("tutorial.json not found in archive")
    target = dest_root / f"content-{int(time.time())}"
    target.mkdir(parents=True)
    for name in names:
        if not name.startswith(top):
            continue
        rel = name[len(top):]
        if not rel or not (rel == "tutorial.json" or rel.startswith("courses/")):
            continue
        out = (target / rel).resolve()
        if not str(out).startswith(str(target.resolve())):  # zip-slip guard
            continue
        if name.endswith("/"):
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(archive.read(name))
    json.loads((target / "tutorial.json").read_text(encoding="utf-8"))  # sanity check
    for old in dest_root.glob("content-*"):
        if old != target:
            shutil.rmtree(old, ignore_errors=True)
    return target
