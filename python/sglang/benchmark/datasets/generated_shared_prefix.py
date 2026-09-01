import math
import pickle
import random
import re
import uuid
from argparse import Namespace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase

from sglang.benchmark.datasets.common import (
    BaseDataset,
    DatasetRow,
    compute_random_lens,
    gen_prompt,
    get_available_tokens,
)


def _zipf_group_probs(num_groups: int, alpha: float) -> np.ndarray:
    """Rank-based Zipf probability vector with rank starting at 1.

    weight(rank)      = 1 / rank ** alpha       (rank in 1..num_groups)
    probability(rank) = weight(rank) / sum_over_all_ranks(weight)

    The returned array has length num_groups; element i corresponds to
    group index i (rank i + 1), so group 0 is the hottest.
    """
    if num_groups <= 0:
        raise ValueError(f"num_groups must be > 0, got {num_groups}")
    ranks = np.arange(1, num_groups + 1, dtype=np.float64)
    weights = 1.0 / (ranks**alpha)
    return weights / weights.sum()


def _controlled_token_pool(tokenizer: PreTrainedTokenizerBase) -> List[int]:
    available = list(dict.fromkeys(get_available_tokens(tokenizer)))
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        available = [token_id for token_id in available if token_id < vocab_size]

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    byte_fallback_pattern = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    token_ids = [
        token_id
        for token_id in available
        if token_id not in special_ids
        and not byte_fallback_pattern.match(
            tokenizer.convert_ids_to_tokens(token_id) or ""
        )
    ]
    if not token_ids:
        raise ValueError("Tokenizer does not contain usable text token IDs")
    return token_ids


def _decode_token_ids(tokenizer: PreTrainedTokenizerBase, token_ids: List[int]) -> str:
    try:
        return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids)


def _generate_controlled_prefixes(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: List[int],
    prefix_lens: List[int],
    rng: np.random.Generator,
) -> List[str]:
    if any(prefix_lens) and len(prefix_lens) > len(token_ids):
        raise ValueError(
            "Controlled GSP generation supports at most "
            f"{len(token_ids)} non-empty prefix groups, but got "
            f"{len(prefix_lens)}. Reduce --gsp-num-groups."
        )

    base_offset = int(rng.integers(len(token_ids)))
    return [
        _decode_token_ids(
            tokenizer,
            [
                token_ids[(base_offset + group_id + position) % len(token_ids)]
                for position in range(prefix_len)
            ],
        )
        for group_id, prefix_len in enumerate(prefix_lens)
    ]


def _common_prefix_len(left: List[int], right: List[int]) -> int:
    matched = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        matched += 1
    return matched


@dataclass
class GeneratedSharedPrefixDataset(BaseDataset):
    num_groups: int
    prompts_per_group: int
    system_prompt_len: int
    question_len: int
    output_len: int
    range_ratio: float
    seed: int
    fast_prepare: bool
    send_routing_key: bool
    num_turns: int
    ordered: bool
    include_cache_prefix: bool = False
    group_distribution: str = "uniform"
    zipf_alpha: Optional[float] = None

    @classmethod
    def from_args(cls, args: Namespace) -> "GeneratedSharedPrefixDataset":
        assert not getattr(args, "tokenize_prompt", False)
        group_distribution = getattr(args, "gsp_group_distribution", "uniform")
        zipf_alpha = getattr(args, "gsp_zipf_alpha", None)

        # Defensive validation for in-process callers that construct a
        # Namespace by hand and bypass the argparse boundary in
        # serving.py. The CLI hook enforces the same rules first.
        if group_distribution not in ("uniform", "zipf"):
            raise ValueError(
                f"--gsp-group-distribution must be 'uniform' or 'zipf', "
                f"got {group_distribution!r}"
            )
        if group_distribution == "zipf":
            if zipf_alpha is None:
                raise ValueError(
                    "--gsp-group-distribution=zipf requires --gsp-zipf-alpha "
                    "(a finite float > 0)"
                )
            if not math.isfinite(zipf_alpha) or zipf_alpha <= 0:
                raise ValueError(
                    f"--gsp-zipf-alpha must be a finite float > 0, got {zipf_alpha!r}"
                )
        elif zipf_alpha is not None:
            raise ValueError(
                "--gsp-zipf-alpha is only meaningful with "
                "--gsp-group-distribution=zipf; remove --gsp-zipf-alpha "
                "or set --gsp-group-distribution=zipf"
            )

        return cls(
            num_groups=args.gsp_num_groups,
            prompts_per_group=args.gsp_prompts_per_group,
            system_prompt_len=args.gsp_system_prompt_len,
            question_len=args.gsp_question_len,
            output_len=args.gsp_output_len,
            range_ratio=getattr(args, "gsp_range_ratio", 1.0),
            seed=args.seed,
            fast_prepare=getattr(args, "gsp_fast_prepare", False),
            send_routing_key=getattr(args, "gsp_send_routing_key", False),
            num_turns=getattr(args, "gsp_num_turns", 1),
            ordered=getattr(args, "gsp_ordered", False),
            include_cache_prefix=getattr(args, "gsp_prewarm_prefixes", False),
            group_distribution=group_distribution,
            zipf_alpha=zipf_alpha,
        )

    def load(
        self, tokenizer: PreTrainedTokenizerBase, model_id=None
    ) -> List[DatasetRow]:
        return sample_generated_shared_prefix_requests(
            num_groups=self.num_groups,
            prompts_per_group=self.prompts_per_group,
            system_prompt_len=self.system_prompt_len,
            question_len=self.question_len,
            output_len=self.output_len,
            range_ratio=self.range_ratio,
            tokenizer=tokenizer,
            seed=self.seed,
            send_routing_key=self.send_routing_key,
            num_turns=self.num_turns,
            fast_prepare=self.fast_prepare,
            ordered=self.ordered,
            include_cache_prefix=self.include_cache_prefix,
            group_distribution=self.group_distribution,
            zipf_alpha=self.zipf_alpha,
        )


def get_gen_prefix_cache_path(
    seed: int,
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    tokenizer,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
    controlled_generation: bool = False,
):
    """Create cache directory under ~/.cache/sglang/benchmark.

    The uniform-mode filename is preserved exactly as before so existing
    on-disk caches remain valid. Non-default sampling modes get an extra
    suffix encoding the parameters that affect the cached payload. Controlled
    prewarm generation uses a separate suffix so it cannot replace legacy GSP
    samples in the shared cache directory.
    """
    cache_dir = Path.home() / ".cache" / "sglang" / "benchmark"

    suffix = ""
    if group_distribution != "uniform":
        suffix = f"_{group_distribution}_{zipf_alpha}"
    if controlled_generation:
        suffix += "_controlled"

    cache_key = (
        f"gen_shared_prefix_{seed}_{num_groups}_{prompts_per_group}_"
        f"{system_prompt_len}_{question_len}_{output_len}{suffix}_"
        f"{tokenizer.__class__.__name__}.pkl"
    )
    return cache_dir / cache_key


def sample_generated_shared_prefix_requests(
    num_groups: int,
    prompts_per_group: int,
    system_prompt_len: int,
    question_len: int,
    output_len: int,
    range_ratio: float,
    tokenizer: PreTrainedTokenizerBase,
    seed: int,
    send_routing_key: bool = False,
    num_turns: int = 1,
    fast_prepare: bool = False,
    ordered: bool = False,
    include_cache_prefix: bool = False,
    group_distribution: str = "uniform",
    zipf_alpha: Optional[float] = None,
) -> List[DatasetRow]:
    """Generate benchmark requests with shared system prompts and caching.

    When group_distribution is "uniform" (default), each group receives exactly
    prompts_per_group requests; behavior matches the legacy generator.

    When group_distribution is "zipf", each request's group is sampled by rank
    with probability 1/rank**zipf_alpha / sum_k(1/k**zipf_alpha); rank starts at
    1 and group index 0 is the hottest. Sampling uses an isolated
    numpy.random.default_rng(seed) so the shared question/system-prompt pool
    stays byte-identical to uniform mode for the same seed and other args.
    Zipf mode is cached on disk under a distinct key per (group_distribution,
    zipf_alpha) value.

    When include_cache_prefix is enabled, system prompts use deterministic
    cyclic token sequences and each active group gets distinct first question
    tokens. Default GSP generation remains unchanged.
    """
    cache_path = get_gen_prefix_cache_path(
        seed,
        num_groups,
        prompts_per_group,
        system_prompt_len,
        question_len,
        output_len,
        tokenizer,
        group_distribution=group_distribution,
        zipf_alpha=zipf_alpha,
        controlled_generation=include_cache_prefix,
    )
    # range_ratio != 1 / num_turns > 1 perturb the payload but are not in the
    # cache key; send_routing_key embeds a per-run uuid + timestamp that is
    # meaningless to cache. Bypass for these pre-existing reasons only.
    should_cache = range_ratio == 1 and not send_routing_key and num_turns == 1

    if should_cache and cache_path.exists():
        print(f"\nLoading cached generated input data from {cache_path}")
        with open(cache_path, "rb") as f:
            cached_rows = pickle.load(f)
        if not include_cache_prefix or all(
            getattr(row, "cache_prefix", None) is not None
            and getattr(row, "cache_prefix_match_len", None) is not None
            for row in cached_rows
        ):
            return cached_rows
        print("Cached data has no prefix metadata; regenerating it for prewarming.")

    if not should_cache:
        print(f"\nCache bypassed ({range_ratio=}, {send_routing_key=}, {num_turns=})")

    print(
        f"\nGenerating new input data... "
        f"({num_groups=}, {prompts_per_group}, {system_prompt_len=}, {question_len=}, {output_len=}, {range_ratio=}, {num_turns=}, {group_distribution=}, {zipf_alpha=})"
    )

    run_random_str = uuid.uuid4().hex[:8]
    run_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    system_prompt_lens = compute_random_lens(
        full_len=system_prompt_len,
        range_ratio=range_ratio,
        num=num_groups,
    )
    question_lens = np.array(
        compute_random_lens(
            full_len=question_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group * num_turns,
        )
    ).reshape(num_groups, prompts_per_group, num_turns)
    output_lens = np.array(
        compute_random_lens(
            full_len=output_len,
            range_ratio=range_ratio,
            num=num_groups * prompts_per_group,
        )
    ).reshape(num_groups, prompts_per_group)
    del system_prompt_len, question_len, output_len

    # Per-slot group assignment. Uniform mode is the identity assignment
    # [0,0,...,1,1,...,N-1,N-1]; zipf mode samples from the rank distribution
    # using an isolated RNG so default GSP generation does not perturb global
    # random state.
    total_slots = num_groups * prompts_per_group
    if group_distribution == "uniform":
        assignment = np.repeat(np.arange(num_groups), prompts_per_group)
    else:  # "zipf"
        assignment_rng = np.random.default_rng(seed)
        probs = _zipf_group_probs(num_groups, zipf_alpha)
        assignment = assignment_rng.choice(
            num_groups, size=total_slots, replace=True, p=probs
        )

    controlled_rng = np.random.default_rng(seed + 1)
    controlled_token_ids = (
        _controlled_token_pool(tokenizer) if include_cache_prefix else None
    )
    if include_cache_prefix:
        system_prompts = _generate_controlled_prefixes(
            tokenizer,
            controlled_token_ids,
            system_prompt_lens,
            controlled_rng,
        )
    else:
        system_prompts = [
            gen_prompt(tokenizer, system_prompt_lens[i]) for i in range(num_groups)
        ]
    cache_prefixes = [
        f"{system_prompt}\n\n" if system_prompt_lens[i] > 0 else None
        for i, system_prompt in enumerate(system_prompts)
    ]
    cache_prefix_lens = [
        len(tokenizer.encode(prefix)) if prefix is not None else None
        for prefix in cache_prefixes
    ]

    # shape: (num_groups, prompts_per_group, num_turns)
    questions = [
        [
            [
                (
                    ""
                    if include_cache_prefix and t == 0
                    else gen_prompt(tokenizer, int(question_lens[g, p, t]))
                )
                for t in range(num_turns)
            ]
            for p in range(prompts_per_group)
        ]
        for g in range(num_groups)
    ]
    if include_cache_prefix and np.any(question_lens[:, :, 0] > 0):
        group_counts = np.bincount(assignment, minlength=num_groups)
        max_group_size = int(group_counts.max(initial=0))
        if max_group_size > len(controlled_token_ids):
            raise ValueError(
                "Controlled GSP suffix isolation supports at most "
                f"{len(controlled_token_ids)} requests per active prefix group, "
                f"but the largest group has {max_group_size}. Increase "
                "--gsp-num-groups or reduce --num-prompts."
            )

        group_offsets = controlled_rng.integers(
            0, len(controlled_token_ids), size=num_groups
        )
        group_local_indices = [0] * num_groups
        for slot_idx, sampled_group in enumerate(assignment):
            src_g, src_p = divmod(slot_idx, prompts_per_group)
            sampled_group = int(sampled_group)
            current_question_len = int(question_lens[src_g, src_p, 0])
            if current_question_len > 0:
                local_index = group_local_indices[sampled_group]
                question_token_ids = controlled_rng.choice(
                    controlled_token_ids,
                    size=current_question_len,
                    replace=True,
                ).tolist()
                question_token_ids[0] = controlled_token_ids[
                    (int(group_offsets[sampled_group]) + local_index)
                    % len(controlled_token_ids)
                ]
                questions[src_g][src_p][0] = _decode_token_ids(
                    tokenizer, question_token_ids
                )
            group_local_indices[sampled_group] += 1

    input_requests = []
    total_input_tokens = 0
    total_output_tokens = 0
    for slot_idx, sampled_g in enumerate(
        tqdm(assignment, desc="Generating shared-prefix prompts")
    ):
        # src_(g,p) walks the question pool in uniform-enumeration order, so
        # per-slot question text is reproducibly identical across modes.
        src_g, src_p = divmod(slot_idx, prompts_per_group)
        sampled_g = int(sampled_g)

        system_prompt = system_prompts[sampled_g]
        routing_key = (
            f"{run_random_str}_{run_start_timestamp}_{sampled_g}"
            if send_routing_key
            else None
        )
        turn_questions = questions[src_g][src_p]
        first_turn_prompt = (
            f"{system_prompt}\n\n{turn_questions[0]}"
            if system_prompt
            else turn_questions[0]
        )
        turn_prompts = [first_turn_prompt] + turn_questions[1:]
        full_prompt = turn_prompts[0] if num_turns == 1 else turn_prompts
        prompt_token_ids = None if fast_prepare else tokenizer.encode(turn_prompts[0])
        prompt_len = 1 if fast_prepare else len(prompt_token_ids)
        output_len_val = int(output_lens[src_g, src_p])
        cache_prefix_match_len = None
        if cache_prefixes[sampled_g] is not None and prompt_token_ids is not None:
            cache_prefix_match_len = _common_prefix_len(
                tokenizer.encode(cache_prefixes[sampled_g]), prompt_token_ids
            )

        input_requests.append(
            DatasetRow(
                prompt=full_prompt,
                prompt_len=prompt_len,
                output_len=output_len_val,
                routing_key=routing_key,
                cache_prefix=cache_prefixes[sampled_g],
                cache_prefix_len=cache_prefix_lens[sampled_g],
                cache_prefix_match_len=cache_prefix_match_len,
            )
        )
        total_input_tokens += prompt_len
        total_output_tokens += output_len_val

    if not ordered:
        random.shuffle(input_requests)

    print(f"\nGenerated shared prefix dataset statistics:")
    print(f"Number of groups: {num_groups}")
    print(f"Prompts per group: {prompts_per_group}")
    print(f"Number of turns: {num_turns}")
    print(f"Group distribution: {group_distribution}")
    if group_distribution == "zipf":
        print(f"Zipf alpha: {zipf_alpha}")
    print(f"Total prompts: {len(input_requests)}")
    if not fast_prepare:
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total output tokens: {total_output_tokens}")
        print(
            f"Average system prompt length: {sum(len(tokenizer.encode(sp)) for sp in system_prompts) / len(system_prompts):.1f} tokens"
        )
        all_questions = [q for group in questions for conv in group for q in conv]
        print(
            f"Average question length: {sum(len(tokenizer.encode(q)) for q in all_questions) / len(all_questions):.1f} tokens\n"
        )

    if should_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Caching generated input data to {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(input_requests, f)

    return input_requests
