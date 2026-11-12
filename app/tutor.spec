# PyInstaller recipe: builds a standalone app (no Python installation needed).
#   pip install pyinstaller PySide6 pillow
#   pyinstaller app/tutor.spec --noconfirm
# Output: dist/<AppName>/ (Windows/Linux) or dist/<AppName>.app (macOS).
import json
import sys
from pathlib import Path

app_dir = Path(SPECPATH)
root = app_dir.parent
manifest = json.loads((root / "tutorial.json").read_text(encoding="utf-8"))
APP_NAME = {"our-tutorial": "OurTutorial", "llm-tutorial": "LLMTutorial"}.get(manifest["id"], "Tutor")

datas = [
    (str(root / "tutorial.json"), "content"),
    (str(root / "courses"), "content/courses"),
    (str(app_dir / "tutor" / "locales"), "tutor/locales"),
    (str(app_dir / "tutor" / "icon.png"), "tutor"),
]

# The app doubles as a Python interpreter for beginners who haven't installed Python
# (`App --run-python main.py`), so include the standard-library modules lessons use.
stdlib = """
abc argparse array ast asyncio base64 bisect calendar cmath collections collections.abc colorsys
concurrent.futures contextlib copy csv dataclasses datetime decimal difflib doctest email enum
fractions functools getpass glob gzip hashlib heapq hmac html html.parser http http.client
http.server importlib inspect io ipaddress itertools json keyword locale logging math mimetypes
numbers operator os pathlib pickle platform pprint queue random re secrets shelve shlex shutil
signal socket sqlite3 statistics string struct subprocess sys tempfile textwrap threading time
timeit tomllib traceback types typing unicodedata unittest urllib urllib.parse urllib.request uuid
warnings weakref xml.etree.ElementTree zipfile zlib zoneinfo
""".split()

a = Analysis(
    [str(app_dir / "run.py")],
    pathex=[str(app_dir)],
    datas=datas,
    hiddenimports=stdlib,
    excludes=["tkinter", "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
              "PySide6.QtQuick", "PySide6.QtQml", "PySide6.QtMultimedia", "PySide6.QtCharts",
              "PySide6.QtDataVisualization", "PySide6.QtPdf", "PySide6.QtBluetooth", "PySide6.QtSql"],
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = str(app_dir / "icon.ico") if sys.platform.startswith("win") else str(app_dir / "tutor" / "icon.png")
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, console=False,
          icon=icon if sys.platform != "linux" else None)
coll = COLLECT(exe, a.binaries, a.datas, name=APP_NAME)
if sys.platform == "darwin":
    app = BUNDLE(coll, name=f"{APP_NAME}.app", icon=icon,
                 bundle_identifier=f"io.github.yeraldosmith.{manifest['id']}",
                 info_plist={"NSHighResolutionCapable": True, "LSApplicationCategoryType": "public.app-category.education"})
