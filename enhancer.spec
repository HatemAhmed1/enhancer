# PyInstaller build definition.
#
# One folder, not one file. A single-file build of this would be a ~5 GB
# executable that unpacks itself to a temporary directory on every launch,
# taking half a minute before the window appears. A folder starts instantly.
#
# Build with:  .venv\Scripts\pyinstaller.exe enhancer.spec --noconfirm

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("src/enhancer/manifest.json", "enhancer")]
binaries = []
hiddenimports = []

# spandrel picks an architecture at load time by inspecting the weights, so
# none of its architecture modules are reachable by static analysis. Without
# this the packaged app loads no models at all.
for package in ("spandrel", "spandrel_extra_arches"):
    try:
        extra_datas, extra_binaries, extra_hidden = collect_all(package)
        datas += extra_datas
        binaries += extra_binaries
        hiddenimports += extra_hidden
    except Exception:
        pass  # spandrel_extra_arches is optional

# yt-dlp resolves its extractors lazily by name.
hiddenimports += collect_submodules("yt_dlp.extractor")

# The vendored RIFE network is imported by path at runtime.
hiddenimports += ["enhancer.rife", "enhancer.rife.ifnet"]

# Large packages pulled in transitively but never used.
#
# Only whole top-level packages, never a submodule of one that is kept.
# Excluding torchvision.models.detection broke the build outright, because
# torchvision.models.__init__ imports it by name — the exclusion turned a
# working import into a circular-import failure at startup, and only in the
# packaged application.
excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook",
    "pytest", "_pytest", "pytest_qt", "setuptools", "pip",
    "tensorboard", "pandas", "scipy",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtMultimedia",
    "PySide6.QtQuick", "PySide6.QtQml", "PySide6.Qt3DRender",
]

a = Analysis(
    ["src/enhancer/__main__.py"],
    pathex=["src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Enhancer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX corrupts CUDA libraries
    console=False,      # no console window; the app has its own log pane
    disable_windowed_traceback=False,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Enhancer",
)
