"""Geometry shared by the DFlash draft-cache budget and physical allocator.

Each request owns two contiguous, page-aligned banks. A working bank contains
the committed suffix followed immediately by the verification block: their
boundary may be inside a page, so a separate scratch page would be incorrect.
Absolute positions (and therefore RoPE) do not change when banks alternate.
This module deliberately has no tensor/runtime imports.
"""

from dataclasses import dataclass


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("value must be nonnegative and alignment positive")
    return (value + alignment - 1) // alignment * alignment


def decode_preallocation_size(max_running_requests: int, configured: int) -> int:
    """Mirror the decode request pool's admission slots, not benchmark C+1."""
    if max_running_requests <= 0 or configured < 0:
        raise ValueError("invalid request-pool capacity")
    return 2 * max_running_requests if max_running_requests <= 32 else configured


def draft_owner_count(
    max_running_requests: int, *, decode: bool, preallocated_requests: int = 0
) -> int:
    if max_running_requests <= 0:
        raise ValueError(
            "compact DFlash requires explicit positive max-running-requests"
        )
    return max_running_requests + (
        decode_preallocation_size(max_running_requests, preallocated_requests)
        if decode
        else 0
    )


@dataclass(frozen=True)
class CompactDFlashLayout:
    owner_count: int
    window_size: int = 1024
    block_size: int = 8
    page_size: int = 64

    def __post_init__(self):
        if (
            min(self.owner_count, self.window_size, self.block_size, self.page_size)
            <= 0
        ):
            raise ValueError("compact DFlash geometry must be positive")
        if self.window_size < self.block_size:
            raise ValueError("draft window must cover the verification block")
        if self.window_size % self.page_size:
            raise ValueError("draft window must be page aligned")

    @property
    def bank_rows(self) -> int:
        return align_up(
            self.window_size + self.page_size - 1 + 2 * self.block_size,
            self.page_size,
        )

    @property
    def request_rows(self) -> int:
        # Index zero has its own dummy banks; it cannot alias any live owner.
        return self.owner_count + 1

    @property
    def physical_tokens(self) -> int:
        return self.request_rows * 2 * self.bank_rows

    def tensor_bytes(self, bytes_per_token: int) -> int:
        if bytes_per_token <= 0:
            raise ValueError("draft bytes per token must be positive")
        # MHATokenToKVPool allocates one additional sentinel page.
        return (self.physical_tokens + self.page_size) * bytes_per_token

    @property
    def request_table_bytes(self) -> int:
        return self.request_rows * self.bank_rows * 4

    def metadata_bytes(self, layer_count: int) -> int:
        # Request table, bank/start/end state, pool pointer/stride arrays, offsets.
        return (
            self.request_table_bytes
            + self.request_rows * 24
            + layer_count * 48
            + self.block_size * 8
        )

    def visible_start(self, prefix_len: int) -> int:
        if prefix_len < 0:
            raise ValueError("prefix length must be nonnegative")
        return max(0, prefix_len - self.window_size) // self.page_size * self.page_size

    def visible_length(self, prefix_len: int) -> int:
        return prefix_len - self.visible_start(prefix_len)

    def bank_base(self, owner: int, bank: int) -> int:
        if not 0 <= owner <= self.owner_count or bank not in (0, 1):
            raise ValueError("invalid draft owner or bank")
        return (2 * owner + bank) * self.bank_rows

    def physical_location(self, owner: int, bank: int, offset: int) -> int:
        if not 0 <= offset < self.bank_rows:
            raise ValueError("draft bank offset out of range")
        return self.bank_base(owner, bank) + offset
