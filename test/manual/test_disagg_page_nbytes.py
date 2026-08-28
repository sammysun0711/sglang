"""CPU unit tests for PD disagg item_len of NHD vs vectorized_5d KV.

Mooncake strides registered MRs by ``page_index * item_len``. NHD stores one
token in ``buf[0]``, so one page is ``buf[0].nbytes * page_size``. vectorized_5d
is page-major: ``buf[0]`` is already one page, and multiplying by page_size
again overshoots the MR.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

PAGE_SIZE = 64


def _bare_pool(layout: str, page_size: int = PAGE_SIZE) -> MHATokenToKVPool:
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.kv_cache_layout = layout
    pool.page_size = page_size
    pool.start_layer = 0
    pool.layer_num = 1
    return pool


class TestDisaggPageNbytes(unittest.TestCase):
    def test_nhd_item_len_is_token_times_page_size(self):
        pool = _bare_pool("nhd")
        buf = torch.zeros(128, 2, 64, dtype=torch.bfloat16)
        self.assertEqual(pool._disagg_page_nbytes(buf), buf[0].nbytes * PAGE_SIZE)

    def test_vectorized_5d_item_len_is_one_page_block(self):
        pool = _bare_pool("vectorized_5d")
        # K: (num_pages, H, Dk//X, page, X) with X=8 for bf16
        buf = torch.zeros(4, 2, 24, PAGE_SIZE, 8, dtype=torch.bfloat16)
        self.assertEqual(pool._disagg_page_nbytes(buf), buf[0].nbytes)
        self.assertNotEqual(pool._disagg_page_nbytes(buf), buf[0].nbytes * PAGE_SIZE)

    def test_contiguous_item_lens_follow_layout(self):
        k_nhd = torch.zeros(128, 2, 64, dtype=torch.bfloat16)
        v_nhd = torch.zeros(128, 2, 64, dtype=torch.bfloat16)
        pool = _bare_pool("nhd")
        pool._get_key_buffer = lambda i: k_nhd
        pool._get_value_buffer = lambda i: v_nhd
        _, _, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(
            item_lens,
            [k_nhd[0].nbytes * PAGE_SIZE, v_nhd[0].nbytes * PAGE_SIZE],
        )

        k_5d = torch.zeros(4, 2, 24, PAGE_SIZE, 8, dtype=torch.bfloat16)
        v_5d = torch.zeros(4, 2, 8, 128, 8, dtype=torch.bfloat16)
        pool = _bare_pool("vectorized_5d")
        pool._get_key_buffer = lambda i: k_5d
        pool._get_value_buffer = lambda i: v_5d
        _, _, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual(item_lens, [k_5d[0].nbytes, v_5d[0].nbytes])


if __name__ == "__main__":
    unittest.main()
