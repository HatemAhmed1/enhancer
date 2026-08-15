"""Two copies would fight over the same graphics memory and could write over
each other's segments.

Each test uses its own pipe name so the suite never depends on, or disturbs,
the real application's.
"""

import subprocess
import sys
import uuid

import pytest

pytest.importorskip("PySide6")

from enhancer import single  # noqa: E402


@pytest.fixture
def name():
    return f"enhancer-test-{uuid.uuid4().hex[:12]}"


def test_real_server_name_is_stable():
    assert single.SERVER_NAME == "enhancer-single-instance"


def test_nothing_running_on_an_unused_name(qapp, name):
    assert not single.another_instance_running(name)


def test_a_live_server_is_detected(qapp, name):
    """Driven from a real subprocess.

    Server and client cannot handshake inside one thread: the server needs its
    event loop to answer, and the client is blocking that thread waiting for
    the answer. Two processes is what actually happens, so that is what is
    tested.
    """
    holder = _spawn_holder(name)
    try:
        assert holder.stdout.readline().strip() == b"listening"
        assert single.another_instance_running(name)
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_detection_stops_after_the_server_closes(qapp, name):
    single.InstanceServer(name).close()
    assert not single.another_instance_running(name)


def test_second_launch_asks_the_first_to_come_forward(qapp, name):
    holder = _spawn_holder(name)
    try:
        assert holder.stdout.readline().strip() == b"listening"
        assert single.another_instance_running(name)
        assert holder.stdout.readline().strip() == b"raised", (
            "the running copy was never asked to bring its window forward")
    finally:
        holder.terminate()
        holder.wait(timeout=10)


def test_an_orphaned_pipe_does_not_lock_the_app_out(qapp, name):
    """A pipe left behind by a crash is connectable but answers nothing.

    Treating mere connectability as proof of life would refuse to start the
    application for good.
    """
    orphan = QLocalServerRaw(name)
    try:
        assert not single.another_instance_running(name), (
            "a silent pipe was mistaken for a running copy")
    finally:
        orphan.close()


class QLocalServerRaw:
    """A pipe that accepts connections and then says nothing at all."""

    def __init__(self, name):
        from PySide6.QtNetwork import QLocalServer

        QLocalServer.removeServer(name)
        self._server = QLocalServer()
        self._server.listen(name)

    def close(self):
        from PySide6.QtNetwork import QLocalServer

        self._server.close()
        QLocalServer.removeServer(self._server.serverName())


def test_a_second_server_can_start_after_an_orphan_is_cleared(qapp, name):
    orphan = QLocalServerRaw(name)
    assert not single.another_instance_running(name)
    orphan.close()
    server = single.InstanceServer(name)
    try:
        assert server.listening
    finally:
        server.close()


HOLDER_SOURCE = """
import sys
from PySide6.QtCore import QCoreApplication
from enhancer.single import InstanceServer

app = QCoreApplication([])
server = InstanceServer(sys.argv[1])
server.raise_requested.connect(lambda: (print("raised", flush=True)))
print("listening", flush=True)
app.exec()
"""


def _spawn_holder(name):
    """Run a real InstanceServer in its own process, as happens in practice."""
    return subprocess.Popen(
        [sys.executable, "-c", HOLDER_SOURCE, name],
        stdout=subprocess.PIPE,
    )
