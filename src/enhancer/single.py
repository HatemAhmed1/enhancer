"""Single-instance guard.

Two copies would compete for the same graphics memory and, worse, could pick up
the same job directory and write over each other's segments. One window only.

Connecting to the named pipe is not sufficient evidence that a copy is running:
Windows can leave an orphaned pipe behind when a process dies badly, and a
connect against one of those succeeds. So the check is a handshake — the live
copy must answer. A stale pipe cannot, and gets cleared instead of locking the
application out permanently.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket

SERVER_NAME = "enhancer-single-instance"
CONNECT_TIMEOUT_MS = 300
REPLY_TIMEOUT_MS = 600

PING = b"ping"
PONG = b"pong"


def another_instance_running(name: str = SERVER_NAME) -> bool:
    """True when a live copy answers on the named pipe.

    Also nudges that copy to bring its window forward, so using the shortcut a
    second time behaves the way people expect rather than doing nothing.
    """
    socket = QLocalSocket()
    socket.connectToServer(name)
    if not socket.waitForConnected(CONNECT_TIMEOUT_MS):
        return False

    try:
        socket.write(PING)
        socket.flush()
        if not socket.waitForReadyRead(REPLY_TIMEOUT_MS):
            # Connected, but nothing is listening: an orphaned pipe. Clear it
            # so the application is not locked out for good.
            QLocalServer.removeServer(name)
            return False
        return socket.readAll().data().startswith(PONG)
    finally:
        socket.disconnectFromServer()


class InstanceServer(QObject):
    """Listens for later launches and asks the window to come forward."""

    raise_requested = Signal()

    def __init__(self, name: str = SERVER_NAME) -> None:
        super().__init__()
        self.name = name
        self._server = QLocalServer()
        # Safe: we only reach here after failing the handshake above, so any
        # existing pipe of this name is known to be stale.
        QLocalServer.removeServer(name)
        self._server.listen(name)
        self._server.newConnection.connect(self._on_connection)

    @property
    def listening(self) -> bool:
        return self._server.isListening()

    def _on_connection(self) -> None:
        socket = self._server.nextPendingConnection()
        if socket is None:
            return
        socket.waitForReadyRead(REPLY_TIMEOUT_MS)
        socket.readAll()
        socket.write(PONG)
        socket.flush()
        socket.waitForBytesWritten(REPLY_TIMEOUT_MS)
        socket.disconnectFromServer()
        self.raise_requested.emit()

    def close(self) -> None:
        self._server.close()
        QLocalServer.removeServer(self.name)
