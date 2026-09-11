from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class TboCommEvents:
    compute_done: torch.cuda.Event
    comm_done: torch.cuda.Event


class TboCommStreamPool:
    """Process-local communication streams shared by TBO operations.

    A single stream is used for all communication operations issued against the
    same process group on a device.  Sharing the stream preserves collective
    ordering and prevents TP collectives and EP communication from competing
    with each other while still allowing them to overlap compute on the default
    stream.
    """

    _streams: Dict[Tuple[int, int], torch.cuda.Stream] = {}
    _events: Dict[Tuple[int, int, int], TboCommEvents] = {}

    @classmethod
    def get_stream_from_pool(cls, group) -> torch.cuda.Stream:
        key = (torch.cuda.current_device(), id(group))
        stream = cls._streams.get(key)
        if stream is None:
            stream = torch.cuda.Stream(priority=0)
            cls._streams[key] = stream
        return stream

    @classmethod
    def get_events(cls, group, subbatch_index: int) -> TboCommEvents:
        key = (torch.cuda.current_device(), id(group), subbatch_index)
        events = cls._events.get(key)
        if events is None:
            events = TboCommEvents(
                compute_done=torch.cuda.Event(blocking=False, interprocess=False),
                comm_done=torch.cuda.Event(blocking=False, interprocess=False),
            )
            cls._events[key] = events
        return events

    @classmethod
    def clear_group(cls, group) -> None:
        device = torch.cuda.current_device()
        group_id = id(group)
        cls._streams.pop((device, group_id), None)
        for key in [key for key in cls._events if key[:2] == (device, group_id)]:
            cls._events.pop(key, None)
