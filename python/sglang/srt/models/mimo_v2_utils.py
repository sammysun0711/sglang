# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

def supports_native_mimo_vectorized_v_cache(
    *,
    attention_backend: str,
    kv_cache_layout: str,
    head_dim: int,
    swa_head_dim: int | None,
    v_head_dim: int | None,
    swa_v_head_dim: int | None,
    full_num_kv_heads: int,
    swa_num_kv_heads: int,
) -> bool:
    """Return whether every MiMo attention pool can store native V128."""
    return (
        attention_backend == "aiter"
        and kv_cache_layout == "vectorized_5d"
        and head_dim == 192
        and swa_head_dim == 192
        and v_head_dim == 128
        and swa_v_head_dim == 128
        and full_num_kv_heads == 1
        and swa_num_kv_heads == 1
    )
