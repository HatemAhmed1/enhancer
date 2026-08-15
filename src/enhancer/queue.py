"""Render queue.

Holds a list of jobs and tracks what each one is doing. Contains no Qt and no
rendering, so the rules about what may be started, stopped or removed can be
tested on their own.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

from .requests import RenderRequest


class TaskState(Enum):
    WAITING = "Waiting"
    RUNNING = "Running"
    DONE = "Done"
    STOPPED = "Stopped"
    FAILED = "Failed"


# States a task can be removed from. A running task must be stopped first,
# otherwise its worker thread would outlive the row describing it.
REMOVABLE = {TaskState.WAITING, TaskState.DONE, TaskState.STOPPED, TaskState.FAILED}

_ids = itertools.count(1)


@dataclass(eq=False)
class Task:
    """One queued job.

    Compared by identity, not by value: two jobs for the same file with the
    same settings are still two separate jobs.
    """

    request: RenderRequest
    state: TaskState = TaskState.WAITING
    done_frames: int = 0
    total_frames: int = 0
    message: str = ""
    id: int = field(default_factory=lambda: next(_ids))

    @property
    def name(self) -> str:
        return self.request.source.name

    @property
    def percent(self) -> int:
        if self.total_frames <= 0:
            return 0
        return min(100, int(100 * self.done_frames / self.total_frames))

    @property
    def can_remove(self) -> bool:
        return self.state in REMOVABLE

    @property
    def is_active(self) -> bool:
        return self.state is TaskState.RUNNING


class RenderQueue:
    """An ordered list of jobs, at most one running at a time.

    One at a time is deliberate: two renders would compete for the same 6 GB of
    graphics memory and both would end up slower than running them in turn.
    """

    def __init__(self) -> None:
        self.tasks: list[Task] = []

    def __len__(self) -> int:
        return len(self.tasks)

    def add(self, request: RenderRequest) -> Task:
        task = Task(request=request)
        self.tasks.append(task)
        return task

    def get(self, task_id: int) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    @property
    def running(self) -> Task | None:
        return next((t for t in self.tasks if t.state is TaskState.RUNNING), None)

    def next_waiting(self) -> Task | None:
        return next((t for t in self.tasks if t.state is TaskState.WAITING), None)

    def start(self, task: Task) -> bool:
        """Mark a task running. Refuses if another already is."""
        if self.running is not None or task.state is TaskState.RUNNING:
            return False
        task.state = TaskState.RUNNING
        task.message = ""
        return True

    def finish(self, task: Task, message: str = "") -> None:
        task.state = TaskState.DONE
        task.message = message
        if task.total_frames:
            task.done_frames = task.total_frames

    def stop(self, task: Task, message: str = "Stopped") -> None:
        """Stopping keeps progress: a stopped job resumes where it left off."""
        task.state = TaskState.STOPPED
        task.message = message

    def fail(self, task: Task, message: str) -> None:
        task.state = TaskState.FAILED
        task.message = message

    def requeue(self, task: Task) -> None:
        task.state = TaskState.WAITING
        task.message = ""

    def remove(self, task: Task) -> bool:
        """Remove a task. A running task must be stopped first."""
        if not task.can_remove:
            return False
        self.tasks.remove(task)
        return True

    def clear_finished(self) -> int:
        """Drop everything that has finished, one way or another."""
        gone = [t for t in self.tasks if t.state in
                (TaskState.DONE, TaskState.FAILED, TaskState.STOPPED)]
        for task in gone:
            self.tasks.remove(task)
        return len(gone)
