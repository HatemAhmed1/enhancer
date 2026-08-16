"""Startup feedback.

A packaged build has to load several gigabytes of graphics libraries before the
window can appear, which takes tens of seconds on a first run while Windows
pulls them off disk. An application that shows nothing for that long reads as
broken, and gets double-clicked again.

Qt itself is cheap to start. So the window frame goes up immediately with a
line of text, and the expensive imports happen afterwards with the message
updating as they go.
"""

from __future__ import annotations



def run_with_splash() -> int:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

    from . import theme
    from .single import InstanceServer, another_instance_running

    app = QApplication([])
    if another_instance_running():
        return 0

    palette = theme.apply(app)

    splash = QWidget()
    splash.setWindowTitle("Enhancer")
    splash.setWindowFlag(Qt.FramelessWindowHint)
    splash.resize(360, 120)
    layout = QVBoxLayout(splash)
    layout.setContentsMargins(28, 24, 28, 24)

    title = QLabel("Enhancer")
    title.setObjectName("headline")
    layout.addWidget(title)

    status = QLabel("Starting…")
    status.setObjectName("status")
    layout.addWidget(status)

    splash.setStyleSheet(
        f"QWidget {{ background: {palette.surface}; "
        f"border: 1px solid {palette.border_strong}; border-radius: 8px; }}"
    )
    splash.show()
    app.processEvents()

    holder: dict[str, object] = {}

    def load() -> None:
        status.setText("Loading graphics libraries…")
        app.processEvents()
        from .window import MainWindow  # noqa: F401  (the expensive part)

        status.setText("Opening…")
        app.processEvents()

        guard = InstanceServer()
        window = MainWindow()
        guard.raise_requested.connect(window.bring_to_front)
        holder["guard"] = guard
        holder["window"] = window
        window.show()
        splash.close()

    QTimer.singleShot(50, load)
    try:
        return app.exec()
    finally:
        guard = holder.get("guard")
        if guard is not None:
            guard.close()  # type: ignore[union-attr]
