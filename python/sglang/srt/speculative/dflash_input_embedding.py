"""DFlash's constant mask rows and per-step verified-token embeddings.

The graph runner owns the destination storage. Only AITER's out-of-place
draft normalization path may retain mask rows across replays. Eager forwards
materialize a fresh input so residual updates cannot corrupt the cached mask.
"""

from dataclasses import dataclass

import torch


@dataclass
class DFlashInputEmbeddings:
    verified: torch.Tensor  # [batch, hidden], after vocabulary TP reduction
    mask: torch.Tensor  # [1, hidden], cached after loading the trained mask row
    block_size: int

    def materialize(self) -> torch.Tensor:
        """Preserve the ordinary input contract when graph replay is unavailable."""
        batch, hidden = self.verified.shape
        output = self.mask.new_empty((batch, self.block_size, hidden))
        output.copy_(self.mask.view(1, 1, hidden))
        output[:, 0].copy_(self.verified)
        return output.view(batch * self.block_size, hidden)


class DFlashGraphInputCache:
    """Initialize mask rows once, then patch only verified rows on the caller stream.

    No input-sized allocation, device scalar read, or host synchronization is
    needed. Invalidate after a normal full-input copy or graph recapture. The
    single-stream draft runner orders each patch after the preceding replay.
    """

    def __init__(self):
        self.invalidate()

    def invalidate(self):
        self._buffer = None
        self._mask = None
        self._block_size = None

    def update(self, buffer: torch.Tensor, inputs: DFlashInputEmbeddings):
        batch, hidden = inputs.verified.shape
        block = inputs.block_size
        if (
            block <= 0
            or buffer.ndim != 2
            or not buffer.is_contiguous()
            or buffer.shape[1] != hidden
            or buffer.shape[0] % block
            or batch * block > buffer.shape[0]
            or inputs.mask.shape != (1, hidden)
            or inputs.mask.device != buffer.device
            or inputs.verified.device != buffer.device
        ):
            raise ValueError("Invalid DFlash mask-cache graph input layout")
        if (
            self._buffer is not buffer
            or self._mask is not inputs.mask
            or self._block_size != block
        ):
            # Initialize the entire capacity, including currently inactive rows,
            # so a later larger batch/bucket needs no mask fill.
            buffer.copy_(inputs.mask)
            self._buffer = buffer
            self._mask = inputs.mask
            self._block_size = block
        buffer[: batch * block : block].copy_(inputs.verified)
