# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform graph lifecycle; metadata computation uses normal plugin dispatch."""

from collections.abc import Callable
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_fl.compilation.graph import Graph
from vllm_fl.dispatch import resolve_op

logger = init_logger(__name__)
compute_common_slot_mapping = resolve_op("qwen38_compute_common_slot_mapping")


class CommonSlotMappingGraphRunner:
    """Platform graph wrapper for the common paged-KV slot producer."""

    def __init__(self) -> None:
        self.graphs: dict[tuple[int, int], Any] = {}
        self.graph_pool = current_platform.get_global_graph_pool()
        self._missing_graph_keys: set[tuple[int, int]] = set()

    def clear(self) -> None:
        self.graphs.clear()
        self._missing_graph_keys.clear()

    def run(
        self,
        block_table: Any,
        num_reqs: int,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        *,
        use_graph: bool,
        capture: bool,
        compute: Callable[
            [
                Any,
                int,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
            None,
        ] = compute_common_slot_mapping,
    ) -> bool:
        if not use_graph:
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            return False

        key = (id(block_table), num_reqs)
        graph = self.graphs.get(key)
        if capture:
            if graph is not None:
                graph.replay()
                return True

            graph = Graph.graph()
            with current_platform.torch_device_fn.graph(graph, pool=self.graph_pool):
                compute(
                    block_table,
                    num_reqs,
                    query_start_loc,
                    positions,
                    seq_lens,
                    num_computed_tokens,
                )
            self.graphs[key] = graph
            return True

        if graph is None:
            if key not in self._missing_graph_keys:
                logger.warning(
                    "Common slot mapping graph for %d requests was not captured; "
                    "falling back to eager execution.",
                    num_reqs,
                )
                self._missing_graph_keys.add(key)
            compute(
                block_table,
                num_reqs,
                query_start_loc,
                positions,
                seq_lens,
                num_computed_tokens,
            )
            return False

        graph.replay()
        return True
