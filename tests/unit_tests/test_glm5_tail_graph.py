"""私有插件tail算子的逐元素参考：多请求、未对齐chunk、padding和动态graph replay。"""

import json
import torch
from vllm_fl.kernels.glm5_next.indexer_backend import INDEXER_BACKEND


def slots_for(lengths, iteration):
    slots = []
    count = len(lengths)
    for i, length in enumerate(lengths):
        page = (i + iteration) % count + 1
        start = (i + iteration) % 4
        slots.extend(page * 4 + (start + j) % 4 for j in range(length))
    return slots + [-1, -1, 999999]


def reference(k, gates, slots, pages):
    result = torch.full((pages, 2, 4, 128), -97.0, dtype=k.dtype)
    # Independent per-request segment traversal, not the Triton lookahead.
    begin = 0
    while begin < len(slots):
        page = slots[begin] // 4
        if slots[begin] < 0 or page >= pages:
            begin += 1
            continue
        end = begin + 1
        while end < len(slots) and slots[end] >= 0 and slots[end] // 4 == page:
            end += 1
        remainder = (slots[end - 1] % 4 + 1) % 4
        if remainder:
            for row in range(max(begin, end - remainder), end):
                offset = slots[row] % 4
                result[page, 0, offset] = k[row]
                result[page, 1, offset] = gates[row]
        begin = end
    return result


def test_prefill_tail_eager_graph():
    gen = torch.Generator().manual_seed(5347)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for batch in (1, 2, 4, 8, 16, 32):
            lengths = [1 + (i * 3 + 4) % 11 for i in range(batch)]
            n = sum(lengths) + 3
            pages = batch + 2
            k = torch.empty(n, 256, dtype=dtype, device="cuda")[..., ::2]
            gate = torch.empty(n, 256, dtype=dtype, device="cuda")[..., ::2]
            slots = torch.empty(
                n, dtype=torch.int32 if batch % 4 else torch.int64, device="cuda"
            )
            cache = torch.empty(pages, 2, 4, 256, dtype=dtype, device="cuda")[..., ::2]

            def set_inputs(iteration):
                data = torch.randn(n, 128, generator=gen).to(dtype)
                scores = torch.randn(n, 128, generator=gen).to(dtype)
                slot_values = slots_for(
                    lengths if iteration % 2 == 0 else list(reversed(lengths)),
                    iteration,
                )
                k.copy_(data)
                gate.copy_(scores)
                slots.copy_(torch.tensor(slot_values))
                cache.fill_(-97.0)
                return reference(data, scores, slot_values, pages)

            expected = set_inputs(0)
            INDEXER_BACKEND.persist_prefill_tail(k, gate, slots, cache)
            torch.cuda.synchronize()
            torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                INDEXER_BACKEND.persist_prefill_tail(k, gate, slots, cache)
            for iteration in (1, 2, 3):
                expected = set_inputs(iteration)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(cache.cpu(), expected, rtol=0, atol=0)
            print(
                json.dumps(
                    dict(
                        dtype=str(dtype),
                        requests=batch,
                        rows=n,
                        noncontiguous=True,
                        eager="PASS",
                        changed_boundary_graph_replays=3,
                    )
                ),
                flush=True,
            )
    print("PREFILL_TAIL_MULTI_REQUEST_EAGER_GRAPH_PASS", flush=True)
