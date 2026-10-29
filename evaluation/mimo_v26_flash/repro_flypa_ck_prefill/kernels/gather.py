import torch
import triton
import triton.language as tl


@triton.jit
def gather_shuffle_5d_to_linear(
    key_cache_ptr,
    value_cache_ptr,
    key_out_ptr,  # (T, num_heads, key_head_size), store dtype
    value_out_ptr,  # (T, num_heads, value_head_size), store dtype
    slot_mapping_ptr,  # (T,) absolute pool slot id per token
    key_out_stride_token,
    value_out_stride_token,
    num_heads,
    key_head_size,
    value_head_size,
    block_size,
    X: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
    KEY_BLOCK_D: tl.constexpr,
    VALUE_BLOCK_D: tl.constexpr,
):
    """Inverse of :func:`reshape_and_cache_shuffle_5d`.

    Gather one token's K/V from the SHUFFLE 5D paged cache into the
    canonical (T, H, D) layout that aiter's ``mha_batch_prefill_func``
    expects in LINEAR mode. Source addressing is identical to the
    writer kernel so any bit-exact round-trip is guaranteed.
    """
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    block_idx = slot_idx // block_size
    slot_in_page = slot_idx % block_size
    page_outer = slot_in_page // X
    page_inner = slot_in_page % X

    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)
    head_mask = head_idx < num_heads
    key_d = tl.arange(0, KEY_BLOCK_D)
    key_d_mask = key_d < key_head_size
    key_d_outer = key_d // X
    key_d_inner = key_d % X
    value_d = tl.arange(0, VALUE_BLOCK_D)
    value_d_mask = value_d < value_head_size

    key_layer_stride = num_heads * key_head_size * block_size
    key_head_stride = key_head_size * block_size

    key_src_mask = head_mask[:, None] & key_d_mask[None, :]
    k_src = (
        block_idx * key_layer_stride
        + head_idx[:, None] * key_head_stride
        + key_d_outer[None, :] * block_size * X
        + slot_in_page * X
        + key_d_inner[None, :]
    )
    k = tl.load(key_cache_ptr + k_src, mask=key_src_mask)

    value_layer_stride = num_heads * value_head_size * block_size
    value_head_stride = value_head_size * block_size
    value_src_mask = head_mask[:, None] & value_d_mask[None, :]
    v_src = (
        block_idx * value_layer_stride
        + head_idx[:, None] * value_head_stride
        + page_outer * value_head_size * X
        + value_d[None, :] * X
        + page_inner
    )
    v = tl.load(value_cache_ptr + v_src, mask=value_src_mask)

    dst_k = (
        token_idx * key_out_stride_token
        + head_idx[:, None] * key_head_size
        + key_d[None, :]
    )
    tl.store(key_out_ptr + dst_k, k, mask=key_src_mask)
    dst_v = (
        token_idx * value_out_stride_token
        + head_idx[:, None] * value_head_size
        + value_d[None, :]
    )
    tl.store(value_out_ptr + dst_v, v, mask=value_src_mask)


def launch_gather_shuffle_5d_to_linear(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    workspace: dict | None = None,
):
    """Inverse of :func:`launch_reshape_and_cache_shuffle_5d`.

    Returns ``(key_out, value_out)`` each shaped
    ``(T, num_heads, key_head_size)`` and
    ``(T, num_heads, value_head_size)`` in ``key_cache.dtype`` /
    ``value_cache.dtype``. The caller is responsible for passing the
    right per-tensor descales downstream when ``store_dtype`` is fp8.

    Args:
        key_cache: (num_blocks, num_heads, key_head_size // X, block_size, X)
        value_cache: (num_blocks, num_heads, block_size // X, value_head_size, X)
        slot_mapping: (T,) per-token absolute slot id in
            ``[0, num_blocks * block_size)``
        workspace: Optional scratch buffers owned by one attention backend.
            Returned tensors alias this storage and are overwritten by the
            next gather with the same layout. Use one workspace per execution
            stream. Buffers grow geometrically to avoid allocator fragmentation
            as chunked prefill extends the context. Graph capture uses separate
            allocations so later workspace growth cannot invalidate a graph.
    """
    assert key_cache.dim() == 5 and value_cache.dim() == 5
    num_blocks, num_heads, kc_D_over_X, block_size, X = key_cache.shape
    vc_blocks, vc_H, vc_page_over_X, vc_D, vc_X = value_cache.shape
    assert vc_blocks == num_blocks and vc_H == num_heads
    assert vc_page_over_X * X == block_size and vc_X == X
    key_head_size = kc_D_over_X * X
    value_head_size = vc_D

    num_tokens = slot_mapping.numel()
    if (
        workspace is not None
        and key_cache.device.type == "cuda"
        and torch.cuda.is_current_stream_capturing()
    ):
        workspace = None
    layout = (
        num_heads,
        key_head_size,
        value_head_size,
        key_cache.dtype,
        value_cache.dtype,
        key_cache.device,
    )
    buffers = workspace.get(layout) if workspace is not None else None
    if buffers is None or buffers[0].shape[0] < num_tokens:
        # Release the previous workspace before allocating its replacement.
        if workspace is not None:
            workspace.pop(layout, None)
            buffers = None
        capacity = (
            triton.next_power_of_2(max(1, num_tokens))
            if workspace is not None
            else num_tokens
        )
        buffers = (
            torch.empty(
                (capacity, num_heads, key_head_size),
                dtype=key_cache.dtype,
                device=key_cache.device,
            ),
            torch.empty(
                (capacity, num_heads, value_head_size),
                dtype=value_cache.dtype,
                device=value_cache.device,
            ),
        )
        if workspace is not None:
            workspace[layout] = buffers
    key_out, value_out = (buffer[:num_tokens] for buffer in buffers)

    HEAD_BLOCK = min(4, triton.next_power_of_2(num_heads))
    KEY_BLOCK_D = triton.next_power_of_2(key_head_size)
    VALUE_BLOCK_D = triton.next_power_of_2(value_head_size)
    grid = (num_tokens, triton.cdiv(num_heads, HEAD_BLOCK))

    gather_shuffle_5d_to_linear[grid](
        key_cache,
        value_cache,
        key_out,
        value_out,
        slot_mapping,
        key_out.stride(0),
        value_out.stride(0),
        num_heads,
        key_head_size,
        value_head_size,
        block_size,
        X=X,
        HEAD_BLOCK=HEAD_BLOCK,
        KEY_BLOCK_D=KEY_BLOCK_D,
        VALUE_BLOCK_D=VALUE_BLOCK_D,
    )
    return key_out, value_out
