from pathlib import Path

import pytest

from enhancer.queue import RenderQueue, TaskState
from enhancer.requests import RenderRequest


def _req(name="in.mp4"):
    return RenderRequest(model=Path("m.pth"), source=Path(name), output=Path("out.mkv"))


@pytest.fixture
def queue():
    return RenderQueue()


def test_new_queue_is_empty(queue):
    assert len(queue) == 0
    assert queue.running is None


def test_adding_a_task_leaves_it_waiting(queue):
    task = queue.add(_req())
    assert task.state is TaskState.WAITING
    assert len(queue) == 1


def test_tasks_get_distinct_ids(queue):
    assert queue.add(_req()).id != queue.add(_req()).id


def test_task_is_named_after_its_source(queue):
    assert queue.add(_req("song.mkv")).name == "song.mkv"


def test_starting_a_task_marks_it_running(queue):
    task = queue.add(_req())
    assert queue.start(task)
    assert task.state is TaskState.RUNNING
    assert queue.running is task


def test_only_one_task_runs_at_a_time(queue):
    """Two renders would fight over the same graphics memory."""
    first, second = queue.add(_req("a.mkv")), queue.add(_req("b.mkv"))
    assert queue.start(first)
    assert not queue.start(second)
    assert second.state is TaskState.WAITING


def test_next_waiting_returns_them_in_order(queue):
    first, second = queue.add(_req("a.mkv")), queue.add(_req("b.mkv"))
    assert queue.next_waiting() is first
    queue.start(first)
    queue.finish(first)
    assert queue.next_waiting() is second


def test_finishing_a_task_marks_it_done(queue):
    task = queue.add(_req())
    queue.start(task)
    task.total_frames = 100
    queue.finish(task)
    assert task.state is TaskState.DONE
    assert task.percent == 100


def test_stopping_keeps_progress_so_it_can_resume(queue):
    task = queue.add(_req())
    queue.start(task)
    task.total_frames, task.done_frames = 100, 40
    queue.stop(task)
    assert task.state is TaskState.STOPPED
    assert task.done_frames == 40


def test_failing_records_the_reason(queue):
    task = queue.add(_req())
    queue.start(task)
    queue.fail(task, "out of disk space")
    assert task.state is TaskState.FAILED
    assert "disk" in task.message


def test_a_stopped_task_can_be_queued_again(queue):
    task = queue.add(_req())
    queue.start(task)
    queue.stop(task)
    queue.requeue(task)
    assert task.state is TaskState.WAITING
    assert queue.start(task)


def test_percent_is_zero_before_any_progress(queue):
    assert queue.add(_req()).percent == 0


def test_percent_tracks_progress(queue):
    task = queue.add(_req())
    task.total_frames, task.done_frames = 200, 50
    assert task.percent == 25


def test_percent_never_exceeds_one_hundred(queue):
    task = queue.add(_req())
    task.total_frames, task.done_frames = 100, 140
    assert task.percent == 100


def test_a_waiting_task_can_be_removed(queue):
    task = queue.add(_req())
    assert queue.remove(task)
    assert len(queue) == 0


def test_a_running_task_cannot_be_removed(queue):
    """Removing it would leave its worker thread running with no row."""
    task = queue.add(_req())
    queue.start(task)
    assert not task.can_remove
    assert not queue.remove(task)
    assert len(queue) == 1


def test_a_stopped_task_can_be_removed(queue):
    task = queue.add(_req())
    queue.start(task)
    queue.stop(task)
    assert queue.remove(task)


def test_a_finished_task_can_be_removed(queue):
    task = queue.add(_req())
    queue.start(task)
    queue.finish(task)
    assert queue.remove(task)


def test_clear_finished_leaves_waiting_and_running_alone(queue):
    waiting = queue.add(_req("wait.mkv"))
    running = queue.add(_req("run.mkv"))
    done = queue.add(_req("done.mkv"))
    queue.start(running)
    queue.finish(done)
    assert queue.clear_finished() == 1
    assert set(queue.tasks) == {waiting, running}


def test_clear_finished_removes_failed_and_stopped_too(queue):
    a, b = queue.add(_req("a.mkv")), queue.add(_req("b.mkv"))
    queue.fail(a, "boom")
    queue.stop(b)
    assert queue.clear_finished() == 2
    assert len(queue) == 0


def test_get_finds_a_task_by_id(queue):
    task = queue.add(_req())
    assert queue.get(task.id) is task
    assert queue.get(task.id + 999) is None
