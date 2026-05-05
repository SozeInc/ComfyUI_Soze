"""Shared helpers for live node-body status text and event-log accumulation.

Native ComfyUI API nodes display status text in a widget at the bottom of the node
via PromptServer.send_progress_text. These helpers expose the same mechanism plus
a small EventLog that nodes can use to build a final `status` STRING output that
contains both a one-line headline and a timestamped transcript of what happened
during the run.

Usage:
    from .status_utils import EventLog, push_node_status, finalize_status

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {...},
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def run(self, ..., unique_id=None):
        log = EventLog()
        push_node_status(unique_id, "Starting...", log)
        # ... work ...
        return (output, finalize_status("OK: produced X", log))
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EventLog:
    """Accumulates timestamped events during a node run."""

    def __init__(self):
        self.events: list[str] = []

    def add(self, text: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.events.append(f"[{ts}] {text}")

    def render(self) -> str:
        return "\n".join(self.events)


def push_node_status(unique_id, text: str, event_log: EventLog | None = None) -> None:
    """Push a status string to the node body widget and append it to the event log.
    Silently no-ops on the widget side if the running ComfyUI doesn't expose the API."""
    if event_log is not None:
        event_log.add(text)
    if unique_id is None:
        return
    try:
        from server import PromptServer
        server = PromptServer.instance
        if hasattr(server, "send_progress_text"):
            server.send_progress_text(text, unique_id)
        else:
            server.send_sync("progress_text", {"node_id": unique_id, "text": text})
    except Exception:
        # Status display is best-effort — never let it break the node.
        logger.debug("Failed to push node status for %s", unique_id, exc_info=True)


def finalize_status(headline: str, event_log: EventLog) -> str:
    """Combine a one-line headline with the full event log."""
    log_section = event_log.render()
    if log_section:
        return f"{headline}\n\nEvents:\n{log_section}"
    return headline
