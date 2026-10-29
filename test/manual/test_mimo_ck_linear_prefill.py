"""CK BF16 full-prefill integration: ragged slots, tails, padding and streams."""
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention.aiter_utils import (
    is_gfx942,
    run_mimo_ck_linear_prefill,
)


@pytest.mark.parametrize("physical_rows", [0, 4])
def test_empty_attention_group(physical_rows):
    q = torch.empty(physical_rows, 16*192, dtype=torch.bfloat16)
    batch = SimpleNamespace(batch_size=0, extend_seq_lens_cpu=[], seq_lens_sum=0)
    layer = SimpleNamespace(tp_q_head_num=16, v_head_dim=128)
    output = run_mimo_ck_linear_prefill(None, q, layer, batch, None, None, None)
    assert output.shape == (physical_rows, 16*128)
    assert torch.count_nonzero(output) == 0


@pytest.mark.parametrize("qlens,kvlens,padding", [([257], [8193], 17), ([1,63,257], [17,129,4097], 19), ([128,256], [128,256], 0)])
def test_ck_linear_ragged_graph_and_streams(qlens, kvlens, padding):
    if not torch.cuda.is_available() or not is_gfx942():
        pytest.skip("gfx942 required")
    torch.manual_seed(714)
    torch.backends.cuda.matmul.allow_tf32 = False
    page, heads, qdim, vdim = 64, 16, 192, 128
    npages = [(n+page-1)//page for n in kvlens]
    total_pages = sum(npages)
    kd = torch.randn(total_pages+2, page, 1, qdim, dtype=torch.bfloat16, device="cuda")*.2
    vd = torch.randn(total_pages+2, page, 1, vdim, dtype=torch.bfloat16, device="cuda")*.2
    order = torch.randperm(total_pages, device="cuda")
    slots, offset = [], 0
    for pages, length in zip(npages, kvlens):
        physical = order[offset:offset+pages]
        local = (physical[:,None]*page+torch.arange(page,device="cuda")[None,:]).flatten()[:length]
        slots.append(local)
        if length % page:
            kd[physical[-1], length%page:].fill_(float("nan"))
            vd[physical[-1], length%page:].fill_(float("nan"))
        offset += pages
    kd[-2:].fill_(float("nan"))
    vd[-2:].fill_(float("nan"))
    key = kd.view(-1,page,1,qdim//8,8).permute(0,2,3,1,4).contiguous()
    value = vd.view(-1,page//8,8,1,vdim).permute(0,3,1,4,2).contiguous()
    cuq = [0]
    cuk = [0]
    for ql, kl in zip(qlens,kvlens):
        cuq.append(cuq[-1]+ql)
        cuk.append(cuk[-1]+kl)
    packed_qkv = torch.randn(cuq[-1]+padding, heads*qdim+qdim+vdim,
                             dtype=torch.bfloat16, device="cuda")*.1
    query = packed_qkv[:, :heads*qdim]
    metadata = SimpleNamespace(kv_indices=torch.cat(slots).to(torch.int32),
        kv_indptr=torch.tensor(cuk,device="cuda",dtype=torch.int32),
        max_q_len=max(qlens),max_kv_len=max(kvlens))
    backend = SimpleNamespace(qo_indptr=torch.tensor(cuq,device="cuda",dtype=torch.int32))
    batch = SimpleNamespace(batch_size=len(qlens),extend_seq_lens_cpu=qlens,seq_lens_sum=sum(kvlens))
    layer = SimpleNamespace(tp_q_head_num=heads,qk_head_dim=qdim,v_head_dim=vdim,scaling=qdim**-.5)

    def run(q):
        return run_mimo_ck_linear_prefill(backend,q,layer,batch,key,value,metadata)

    def check(output, q):
        assert torch.isfinite(output).all()
        assert torch.equal(output[cuq[-1]:], torch.zeros_like(output[cuq[-1]:]))
        for index, ql in enumerate(qlens):
            sample = torch.tensor(sorted({0,ql//2,ql-1}),device="cuda")
            selected = metadata.kv_indices[cuk[index]:cuk[index+1]].long()
            k = kd.reshape(-1,qdim)[selected].float()
            v = vd.reshape(-1,vdim)[selected].float()
            qq = q[cuq[index]+sample].view(-1,heads,qdim).float()
            scores = (qq.reshape(-1,qdim)@k.T).view(-1,heads,len(selected))*layer.scaling
            limit = kvlens[index]-ql+sample+1
            scores.masked_fill_(torch.arange(len(selected),device="cuda")[None,None,:]>=limit[:,None,None],-float("inf"))
            expected = torch.softmax(scores,-1).reshape(-1,len(selected))@v
            actual = output[cuq[index]+sample].view(-1,vdim).float()
            torch.testing.assert_close(actual,expected,atol=.02,rtol=.02)

    eager = run(query)
    check(eager,query)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = run(query)
    graph.replay()
    torch.testing.assert_close(replay,eager,atol=0,rtol=0)
    query.neg_()
    for a,b in zip(cuk,cuk[1:]):
        metadata.kv_indices[a:b] = metadata.kv_indices[a:b].flip(0)
    graph.replay()
    check(replay,query)

    opposite = -query
    torch.cuda.synchronize()
    s1,s2 = torch.cuda.Stream(),torch.cuda.Stream()
    with torch.cuda.stream(s1):
        output1 = run(query)
    with torch.cuda.stream(s2):
        output2 = run(opposite)
    torch.cuda.synchronize()
    check(output1,query)
    check(output2,opposite)
    workspace1 = backend._mimo_ck_linear_workspaces[s1.cuda_stream]
    workspace2 = backend._mimo_ck_linear_workspaces[s2.cuda_stream]
    assert {x.data_ptr() for buffers in workspace1.values() for x in buffers}.isdisjoint(
        {x.data_ptr() for buffers in workspace2.values() for x in buffers})
