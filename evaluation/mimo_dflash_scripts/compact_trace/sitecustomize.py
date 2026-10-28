"""Opt-in validation observer, applied identically to baseline and candidate.

Prepend this directory to PYTHONPATH and set DFLASH_VALIDATION_TRACE_DIR.
Only validation runs should use it: CPU copies deliberately synchronize rank 0.
"""

import os

if os.environ.get("DFLASH_VALIDATION_REPEATABLE_QKV") == "1" and not os.environ.get(
    "DFLASH_VALIDATION_TRACE_DIR"
):
    from copy import copy

    import torch
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    _repeatable_init_backends = DFlashWorkerV2.init_backends

    def repeatable_init_backends(self):
        for layer in self.draft_model.layers:
            projection = layer.self_attn.qkv_proj
            projection.quant_method = copy(projection.quant_method)
            projection.quant_method.apply = lambda layer, x, bias=None: (
                torch.nn.functional.linear(x, layer.weight, bias)
            )
        _repeatable_init_backends(self)

    DFlashWorkerV2.init_backends = repeatable_init_backends

if os.environ.get("DFLASH_VALIDATION_TRACE_DIR"):
    from copy import copy
    from pathlib import Path

    import torch
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
    from sglang.srt.layers.attention.utils import launch_gather_shuffle_5d_to_linear
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    _forward = DFlashWorkerV2.forward_batch_generation
    _sample = DFlashWorkerV2._greedy_sample_from_vocab_parallel_head
    _append = DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc
    _init_backends = DFlashWorkerV2.init_backends

    def init_backends(self):
        if os.environ.get("DFLASH_VALIDATION_REPEATABLE_QKV") == "1":
            for layer in self.draft_model.layers:
                projection = layer.self_attn.qkv_proj
                projection.quant_method = copy(projection.quant_method)
                projection.quant_method.apply = lambda layer, x, bias=None: (
                    torch.nn.functional.linear(x, layer.weight, bias)
                )
        _init_backends(self)
        if os.environ.get("DFLASH_VALIDATION_SHADOW") != "1":
            return
        if os.environ.get("DFLASH_VALIDATION_SHADOW_AFTER_WARMUP") == "1":
            self._validation_pending_shadow = True
            return
        initialize_shadow(self)

    def initialize_shadow(self):
        runner = self.draft_model_runner
        if runner.decode_cuda_graph_runner is not None:
            raise RuntimeError("shadow validation requires eager draft execution")
        compact_pool = runner.token_to_kv_pool
        target_pool = self.target_worker.model_runner
        self._validation_shadow_pool = MHATokenToKVPool(
            size=target_pool.max_total_num_tokens,
            page_size=64,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=128,
            v_head_dim=128,
            layer_num=5,
            device=self.device,
            enable_memory_saver=False,
            kv_cache_layout="vectorized_5d",
        )
        table = torch.zeros_like(runner.req_to_token_pool.req_to_token)
        shadow_runner = copy(runner)
        shadow_runner.token_to_kv_pool = self._validation_shadow_pool
        shadow_runner.req_to_token_pool = copy(runner.req_to_token_pool)
        shadow_runner.req_to_token_pool.req_to_token = table
        shadow_runner.attn_backend = AiterAttnBackend(shadow_runner)
        shadow_runner.decode_attn_backend = shadow_runner.attn_backend
        self._validation_shadow_runner = shadow_runner
        original_forward = runner.forward
        layer_observations = []

        def attention_hook(module, inputs, kwargs, output):
            layer_observations.append(
                (kwargs["hidden_states"].detach().clone(), output.detach().clone())
            )

        for layer in self.draft_model.layers:
            layer.self_attn.register_forward_hook(attention_hook, with_kwargs=True)

        def draft_forward(batch, *args, **kwargs):
            bs = batch.batch_size
            prefixes = batch.positions.reshape(bs, 8)[:, 0]
            target_table = target_pool.req_to_token_pool.req_to_token
            for i, owner in enumerate(batch.req_pool_indices.tolist()):
                prefix = int(prefixes[i])
                length = int(batch.seq_lens[i])
                table[owner, : length + 8] = target_table[
                    owner, prefix - length : prefix + 8
                ]
            shadow_batch = copy(batch)
            # Draft residual+norm kernels can update the input embedding in
            # place. Each run must start with the identical unmodified tensor.
            shadow_batch.input_embeds = batch.input_embeds.clone()
            shadow_batch.out_cache_loc = torch.cat(
                [
                    target_table[int(owner), int(prefix) : int(prefix) + 8]
                    for owner, prefix in zip(batch.req_pool_indices, prefixes)
                ]
            ).to(torch.int64)
            layer_observations.clear()
            shadow_output = shadow_runner.forward(shadow_batch, *args, **kwargs)
            shadow_hidden = shadow_output.logits_output.hidden_states.clone()
            first_layers = list(layer_observations)
            layer_observations.clear()
            shadow_batch.input_embeds = batch.input_embeds.clone()
            repeated_shadow = shadow_runner.forward(
                shadow_batch, *args, **kwargs
            ).logits_output.hidden_states.clone()
            repeat_layers = list(layer_observations)
            layer_observations.clear()
            kv_deltas = []
            compact_locations = torch.cat(
                [
                    runner.req_to_token_pool.req_to_token[int(owner), : int(length)]
                    for owner, length in zip(batch.req_pool_indices, batch.seq_lens)
                ]
            ).to(torch.int64)
            legacy_locations = torch.cat(
                [
                    table[int(owner), : int(length)]
                    for owner, length in zip(batch.req_pool_indices, batch.seq_lens)
                ]
            ).to(torch.int64)
            for ck, cv, lk, lv in zip(
                compact_pool.k_buffer,
                compact_pool.v_buffer,
                self._validation_shadow_pool.k_buffer,
                self._validation_shadow_pool.v_buffer,
            ):
                compact_kv = launch_gather_shuffle_5d_to_linear(
                    ck, cv, compact_locations
                )
                legacy_kv = launch_gather_shuffle_5d_to_linear(lk, lv, legacy_locations)
                for ca, la in zip(compact_kv, legacy_kv):
                    kv_deltas.append((ca.float() - la.float()).abs().max().item())
            if max(kv_deltas) != 0:
                raise RuntimeError(
                    f"shadow committed draft KV mismatch, rank={self.tp_rank}, max_abs={kv_deltas}"
                )
            candidate_output = original_forward(batch, *args, **kwargs)
            compact_layers = list(layer_observations)
            a = shadow_hidden
            b = candidate_output.logits_output.hidden_states
            if self.tp_rank == 0:
                self._validation_shadow_hidden = a.detach().cpu()
                self._validation_shadow_delta = (
                    (a.float() - b.float()).abs().max().item()
                )
            if not torch.allclose(a.float(), b.float(), atol=0.02, rtol=0.02):
                directory = Path(os.environ["DFLASH_VALIDATION_TRACE_DIR"]).parent
                repeat_delta = (a.float() - repeated_shadow.float()).abs().max().item()
                torch.save(
                    {
                        "legacy": a.detach().cpu(),
                        "legacy_repeat": repeated_shadow.detach().cpu(),
                        "compact": b.detach().cpu(),
                        "kv_deltas": kv_deltas,
                        "prefixes": prefixes.detach().cpu(),
                        "input_embeds": batch.input_embeds.detach().cpu(),
                        "attention_repeat_deltas": [
                            [
                                (a.float() - b.float()).abs().max().item()
                                for a, b in zip(first, repeated)
                            ]
                            for first, repeated in zip(first_layers, repeat_layers)
                        ],
                        "attention_compact_deltas": [
                            [
                                (a.float() - b.float()).abs().max().item()
                                for a, b in zip(first, candidate)
                            ]
                            for first, candidate in zip(first_layers, compact_layers)
                        ],
                    },
                    directory / f"shadow-mismatch-rank{self.tp_rank}.pt",
                )
                raise RuntimeError(
                    f"shadow draft hidden mismatch, rank={self.tp_rank}, max_abs={(a.float() - b.float()).abs().max().item()}, legacy_repeat_max_abs={repeat_delta}"
                )
            # Do not create worker-owned reusable sampling buffers as inference
            # tensors: the production sampler reuses them outside this context.
            with torch.inference_mode(False):
                self._validation_shadow_ids = _sample(
                    self,
                    hidden_states=a.reshape(bs, 8, -1)[:, 1:, :]
                    .reshape(bs * 7, -1)
                    .clone(),
                    lm_head=target_pool.model.lm_head,
                )
            return candidate_output

        runner.forward = draft_forward
        self._validation_pending_shadow = False

    def append(self, *args, **kwargs):
        if self.tp_rank == 0:
            # Fixed absolute positions let A/B compare input features even when
            # bounded prefill selects fewer rows before draft projection.
            positions = kwargs["positions"]
            hidden = kwargs["target_hidden"][: positions.numel()]
            mask = (positions % 64 == 0) | (positions >= positions.max() - 7)
            probes = {
                "positions": positions[mask].detach().cpu(),
                "target_hidden": hidden[mask, ::64].detach().cpu(),
            }
        result = _append(self, *args, **kwargs)
        shadow_pool = getattr(self, "_validation_shadow_pool", None)
        if shadow_pool is not None:
            shadow_kwargs = dict(kwargs)
            positions = kwargs["positions"]
            table = self.target_worker.model_runner.req_to_token_pool.req_to_token
            target_locations = []
            offset = 0
            for req, count in zip(
                self._validation_batch.reqs, self._validation_materialize_counts(kwargs)
            ):
                target_locations.append(
                    table[req.req_pool_idx, positions[offset : offset + count]]
                )
                offset += count
            shadow_kwargs["cache_loc"] = torch.cat(target_locations).to(torch.int64)
            if kwargs.get("cache_loc_2d") is not None:
                shadow_kwargs["cache_loc_2d"] = shadow_kwargs["cache_loc"].reshape_as(
                    kwargs["cache_loc_2d"]
                )
            pool = self.draft_model_runner.token_to_kv_pool
            self.draft_model_runner.token_to_kv_pool = shadow_pool
            try:
                _append(self, *args, **shadow_kwargs)
            finally:
                self.draft_model_runner.token_to_kv_pool = pool
        if self.tp_rank == 0:
            self._validation_materializations.append(probes)
        return result

    def materialize_counts(self, kwargs):
        if kwargs.get("cache_loc_2d") is not None:
            return [8] * len(self._validation_batch.reqs)
        return [
            min(count, prefix + count - self._compute_compact_start(prefix + count))
            for prefix, count in zip(
                self._validation_batch.prefix_lens, self._validation_batch.extend_lens
            )
        ]

    def compact_start(self, end):
        return max(0, end - 1024) // 64 * 64

    def sample(self, *args, **kwargs):
        result = _sample(self, *args, **kwargs)
        shadow_ids = getattr(self, "_validation_shadow_ids", None)
        if shadow_ids is not None and not torch.equal(shadow_ids, result):
            raise RuntimeError(f"shadow draft proposals differ, rank={self.tp_rank}")
        if self.tp_rank == 0:
            hidden = kwargs["hidden_states"]
            head = kwargs["lm_head"]
            valid_vocab = int(head.shard_indices.num_org_elements)
            self._validation_observation = {
                "draft_hidden": hidden.detach().cpu(),
                "draft_ids": result.detach().cpu(),
                "draft_logits_rank0": torch.matmul(
                    hidden.to(head.weight.dtype), head.weight[:valid_vocab].T
                )
                .detach()
                .cpu(),
            }
        return result

    def forward(self, model_worker_batch, *args, **kwargs):
        batch = model_worker_batch
        # General warmup uses a generated request ID; shadow begins at the
        # harness's first named request after ordinary communication succeeds.
        if getattr(self, "_validation_pending_shadow", False) and any(
            req.rid == "short" for req in batch.reqs
        ):
            torch.cuda.synchronize(self.device)
            initialize_shadow(self)
            torch.cuda.synchronize(self.device)
        self._validation_batch = batch
        materialization = os.environ.get("DFLASH_VALIDATION_MATERIALIZATION", "auto")
        if materialization == "sequential":
            self._use_fused_kv_materialize = False
        elif materialization == "fused":
            if self._fused_kv_helper is None:
                raise RuntimeError(
                    "validation requires the fused materialization helper"
                )
            self._use_fused_kv_materialize = True
        prepare = os.environ.get("DFLASH_VALIDATION_PREPARE_BLOCK", "auto")
        if prepare != "auto":
            self._use_triton_prepare_block = prepare == "triton"
        if self.tp_rank != 0:
            return _forward(self, batch, *args, **kwargs)
        self._validation_observation = None
        self._validation_materializations = []
        rids = [r.rid for r in batch.reqs]
        prefixes = batch.seq_lens.detach().cpu()
        result = _forward(self, batch, *args, **kwargs)
        if materialization == "fused" and not self._use_fused_kv_materialize:
            raise RuntimeError("fused validation silently fell back to sequential")
        if prepare == "triton" and not self._use_triton_prepare_block:
            raise RuntimeError("Triton prepare validation silently fell back to eager")
        observation = self._validation_observation
        if observation is not None:
            ends = result.new_seq_lens
            lengths = self._compute_compact_draft_seq_lens(ends).to(torch.int64)
            compact = getattr(self, "compact_physical_cache", None)
            locs = []
            for i, req in enumerate(batch.reqs):
                length = int(lengths[i])
                end = int(ends[i])
                if compact is None:
                    table = (
                        self.target_worker.model_runner.req_to_token_pool.req_to_token
                    )
                    locs.append(
                        table[req.req_pool_idx, end - length : end].to(torch.int64)
                    )
                else:
                    owner = req.req_pool_idx
                    base = compact.layout.bank_base(owner, int(compact.banks[owner]))
                    start = int(compact.starts[owner])
                    locs.append(
                        torch.arange(
                            base + end - length - start,
                            base + end - start,
                            device=self.device,
                            dtype=torch.int64,
                        )
                    )
            pool = self.draft_model_runner.token_to_kv_pool
            packed_locs = torch.cat(locs)
            kv = [
                launch_gather_shuffle_5d_to_linear(k, v, packed_locs)
                for k, v in zip(pool.k_buffer, pool.v_buffer)
            ]
            observation.update(
                {
                    "rids": rids,
                    "prefix_lens": prefixes,
                    "commit_lens": result.accept_lens.detach().cpu(),
                    "output_ids": result.next_token_ids.detach().cpu(),
                    "committed_lengths": lengths.detach().cpu(),
                    "committed_k": torch.stack([x[0] for x in kv]).detach().cpu(),
                    "committed_v": torch.stack([x[1] for x in kv]).detach().cpu(),
                    "graph_replay": bool(result.can_run_cuda_graph),
                    "fused_materialization": bool(self._use_fused_kv_materialize),
                    "triton_prepare": bool(self._use_triton_prepare_block),
                    "materialization_inputs": self._validation_materializations,
                    "shadow_hidden_max_abs": getattr(
                        self, "_validation_shadow_delta", None
                    ),
                }
            )
            directory = Path(os.environ["DFLASH_VALIDATION_TRACE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            step = getattr(self, "_validation_step", 0)
            torch.save(observation, directory / f"step-{step:06d}.pt")
            self._validation_step = step + 1
        elif self._validation_materializations:
            directory = (
                Path(os.environ["DFLASH_VALIDATION_TRACE_DIR"]).parent
                / "prefill-traces"
            )
            directory.mkdir(parents=True, exist_ok=True)
            step = getattr(self, "_validation_prefill_step", 0)
            torch.save(
                {
                    "rids": rids,
                    "prefix_lens": prefixes,
                    "materialization_inputs": self._validation_materializations,
                },
                directory / f"step-{step:06d}.pt",
            )
            self._validation_prefill_step = step + 1
        return result

    DFlashWorkerV2._greedy_sample_from_vocab_parallel_head = sample
    DFlashWorkerV2.forward_batch_generation = forward
    DFlashWorkerV2._append_target_hidden_to_draft_kv_by_loc = append
    DFlashWorkerV2.init_backends = init_backends
    DFlashWorkerV2._validation_materialize_counts = materialize_counts
    DFlashWorkerV2._compute_compact_start = compact_start


if os.environ.get("DFLASH_VALIDATION_NUMERICAL_DIR"):
    import json
    from pathlib import Path

    import torch
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    _numerical_forward = DFlashWorkerV2.forward_batch_generation

    def numerical_forward(self, batch, *args, **kwargs):
        if self.tp_rank != 0:
            return _numerical_forward(self, batch, *args, **kwargs)
        rids = [req.rid for req in batch.reqs]
        prefixes = batch.seq_lens.detach().cpu().tolist()
        result = _numerical_forward(self, batch, *args, **kwargs)
        logits = getattr(result.logits_output, "next_token_logits", None)
        if logits is None:
            return result
        _, ranked_ids = torch.topk(logits.float(), 2, dim=-1)
        target_argmax = torch.argmax(logits, dim=-1)
        # torch.topk may return tied maxima in a different order from the
        # target's argmax. Put the actual greedy winner first in diagnostics.
        runner_up = torch.where(
            ranked_ids[:, 0] == target_argmax, ranked_ids[:, 1], ranked_ids[:, 0]
        )
        ids = torch.stack((target_argmax, runner_up), dim=-1)
        values = logits.gather(-1, ids).float()
        verify = result.accept_lens is not None
        rows_per_req = int(self.block_size) if verify else 1
        expected_rows = len(rids) * rows_per_req
        if logits.shape[0] != expected_rows:
            raise RuntimeError(
                "numerical observer received unexpected target logits shape"
            )
        values = values.reshape(len(rids), rows_per_req, 2).detach().cpu().tolist()
        ids = ids.reshape(len(rids), rows_per_req, 2).detach().cpu().tolist()
        lengths = (
            result.accept_lens.detach().cpu().tolist() if verify else [1] * len(rids)
        )
        outputs = (
            result.next_token_ids.reshape(len(rids), rows_per_req)
            .detach()
            .cpu()
            .tolist()
        )
        directory = Path(os.environ["DFLASH_VALIDATION_NUMERICAL_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "target-top2.jsonl").open("a") as stream:
            for rid, prefix, length, output, top_ids, top_values in zip(
                rids, prefixes, lengths, outputs, ids, values
            ):
                stream.write(
                    json.dumps(
                        {
                            "rid": rid,
                            "prefix_len": prefix,
                            "output_start_position": prefix + int(verify),
                            "phase": "verify" if verify else "prefill",
                            "accepted_tokens": int(length),
                            "output_ids": output[:length],
                            "top2_ids": top_ids,
                            "top2_logits": top_values,
                            "graph_replay": bool(result.can_run_cuda_graph),
                        }
                    )
                    + "\n"
                )
        return result

    DFlashWorkerV2.forward_batch_generation = numerical_forward


if os.environ.get("DFLASH_VALIDATION_ALLOCATION_TRACE") == "1":
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.mem_cache.kv_cache_builder import get_draft_kv_pool

    _allocation_internal_state = Scheduler.get_internal_state

    def allocation_internal_state(self, request):
        result = _allocation_internal_state(self, request)
        pool, _ = get_draft_kv_pool(
            draft_worker=self.draft_worker,
            spec_algorithm=self.spec_algorithm,
            server_args=self.server_args,
        )
        if pool is not None:
            runners = (
                self.draft_worker.draft_worker.draft_runner_list
                if self.server_args.enable_multi_layer_eagle
                else [self.draft_worker.draft_worker.draft_runner]
            )
            pools = [runner.token_to_kv_pool for runner in runners]
            leaves = []
            for candidate in pools:
                leaves.extend(
                    [candidate.full_kv_pool, candidate.swa_kv_pool]
                    if hasattr(candidate, "full_kv_pool")
                    else [candidate]
                )
            tensors = {
                tensor.data_ptr(): tensor
                for leaf in leaves
                for tensor in [*leaf.k_buffer, *leaf.v_buffer]
            }
            tables = {
                runner.req_to_token_pool.req_to_token.data_ptr(): runner.req_to_token_pool.req_to_token
                for runner in runners
            }
            result.internal_state["validation_draft_allocation"] = {
                "pool_class": type(pool).__name__,
                "size": pool.size,
                "page_size": pool.page_size,
                "dtype": str(pool.dtype),
                "layer_num": pool.layer_num,
                "runner_count": len(runners),
                "unique_pool_count": len({id(candidate) for candidate in pools}),
                "leaves": [
                    {
                        "size": leaf.size,
                        "layout": leaf.kv_cache_layout,
                        "head_num": leaf.head_num,
                        "head_dim": leaf.head_dim,
                        "v_head_dim": leaf.v_head_dim,
                        "layer_num": leaf.layer_num,
                        "k_shapes": [list(v.shape) for v in leaf.k_buffer],
                        "v_shapes": [list(v.shape) for v in leaf.v_buffer],
                    }
                    for leaf in leaves
                ],
                "kv_tensor_bytes": sum(
                    v.numel() * v.element_size() for v in tensors.values()
                ),
                "request_table_shapes": [list(v.shape) for v in tables.values()],
                "request_table_bytes": sum(
                    v.numel() * v.element_size() for v in tables.values()
                ),
            }
        return result

    Scheduler.get_internal_state = allocation_internal_state


if os.environ.get("DFLASH_VALIDATION_FAKE_DIR"):
    import json
    from pathlib import Path

    import torch
    from sglang.srt.layers.attention.utils import launch_gather_shuffle_5d_to_linear
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    try:
        from sglang.srt.speculative.dflash_compact_cache import CompactDFlashCache
    except ModuleNotFoundError as error:
        if error.name != "sglang.srt.speculative.dflash_compact_cache":
            raise
        # The pinned legacy source tree has no compact implementation. Its
        # forward observer still supplies graph/acceptance evidence.
        CompactDFlashCache = None

    if CompactDFlashCache is not None:
        _fake_initialize = CompactDFlashCache.initialize_fake_prefix
    _fake_forward = DFlashWorkerV2.forward_batch_generation

    def fake_initialize(self, reqs, prefix_lens):
        _fake_initialize(self, reqs, prefix_lens)
        locations = torch.cat(
            [
                self.table[
                    req.req_pool_idx, : self.layout.visible_length(int(prefix))
                ].long()
                for req, prefix in zip(reqs, prefix_lens)
            ]
        )
        for k, v in zip(self.pool.k_buffer, self.pool.v_buffer):
            for tensor in launch_gather_shuffle_5d_to_linear(k, v, locations):
                if torch.count_nonzero(tensor).item():
                    raise RuntimeError(
                        "fake compact prefix was not initialized to zero"
                    )
        if self.pool.layer_num and torch.cuda.current_device() == 0:
            directory = Path(os.environ["DFLASH_VALIDATION_FAKE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "admissions.jsonl").open("a") as stream:
                for req, prefix in zip(reqs, prefix_lens):
                    owner = req.req_pool_idx
                    stream.write(
                        json.dumps(
                            {
                                "rid": req.rid,
                                "owner": owner,
                                "prefix_len": int(prefix),
                                "generation": self.generations[owner],
                                "visible_length": self.layout.visible_length(
                                    int(prefix)
                                ),
                                "zero_suffix_verified": True,
                            }
                        )
                        + "\n"
                    )

    def fake_forward(self, batch, *args, **kwargs):
        result = _fake_forward(self, batch, *args, **kwargs)
        if self.tp_rank == 0 and result.accept_lens is not None and len(batch.reqs):
            directory = Path(os.environ["DFLASH_VALIDATION_FAKE_DIR"])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / "forwards.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "rids": [req.rid for req in batch.reqs],
                            "owners": [req.req_pool_idx for req in batch.reqs],
                            "batch_size": len(batch.reqs),
                            "graph_replay": bool(result.can_run_cuda_graph),
                            "accept_lens": result.accept_lens.detach().cpu().tolist(),
                        }
                    )
                    + "\n"
                )
        return result

    if CompactDFlashCache is not None:
        CompactDFlashCache.initialize_fake_prefix = fake_initialize
    DFlashWorkerV2.forward_batch_generation = fake_forward


if os.environ.get("DFLASH_VALIDATION_SERVICE_DIR"):
    import json
    from pathlib import Path

    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
    from sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner import (
        MultiLayerEagleDraftExtendCudaGraphRunner,
    )
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
        MultiLayerEagleWorkerV2,
    )

    _mtp_service_replay = MultiLayerEagleDraftExtendCudaGraphRunner.replay

    def mtp_service_replay(self, *args, **kwargs):
        result = _mtp_service_replay(self, *args, **kwargs)
        self._validation_service_replays = (
            getattr(self, "_validation_service_replays", 0) + 1
        )
        return result

    MultiLayerEagleDraftExtendCudaGraphRunner.replay = mtp_service_replay

    def observe_service_forward(original, mtp=False):
        def service_forward(self, batch, *args, **kwargs):
            step = getattr(self, "_validation_service_step", 0)
            self._validation_service_step = step + 1
            rank = self.draft_worker.tp_rank if mtp else self.tp_rank
            sampled = rank == 0 and step % 40 == 0 and bool(batch.reqs)
            drafts = []
            if sampled:
                if mtp:
                    composite = self.draft_worker.cuda_graph_runner_for_draft_extend
                    drafts = list(composite.runners) if composite else []
                else:
                    drafts = [self.draft_model_runner.decode_cuda_graph_runner]
                before = [
                    getattr(runner, "_validation_service_replays", 0)
                    for runner in drafts
                ]
            result = original(self, batch, *args, **kwargs)
            if sampled and result.accept_lens is not None:
                target = self.target_worker.model_runner.decode_cuda_graph_runner
                target_replay = bool(result.can_run_cuda_graph)
                draft_replays = [
                    getattr(runner, "_validation_service_replays", 0) > count
                    if mtp
                    else target_replay
                    for runner, count in zip(drafts, before)
                ]
                draft_batches = [
                    getattr(runner, "bs", None) if replay else None
                    for runner, replay in zip(drafts, draft_replays)
                ]
                directory = Path(os.environ["DFLASH_VALIDATION_SERVICE_DIR"])
                directory.mkdir(parents=True, exist_ok=True)
                with (directory / "resident-graphs.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(
                            {
                                "step": step,
                                "method": "mtp" if mtp else "dflash",
                                "resident_batch": len(batch.reqs),
                                "target_graph_replay": target_replay,
                                "target_graph_batch": getattr(target, "bs", None)
                                if target_replay
                                else None,
                                "draft_graph_batch": draft_batches[0]
                                if draft_batches
                                else None,
                                "draft_graph_batches": draft_batches,
                                "draft_graph_replays": draft_replays,
                                "retractions": [
                                    req.retraction_count for req in batch.reqs
                                ],
                            }
                        )
                        + "\n"
                    )
            return result

        return service_forward

    DFlashWorkerV2.forward_batch_generation = observe_service_forward(
        DFlashWorkerV2.forward_batch_generation
    )
    MultiLayerEagleWorkerV2.forward_batch_generation = observe_service_forward(
        MultiLayerEagleWorkerV2.forward_batch_generation, mtp=True
    )
