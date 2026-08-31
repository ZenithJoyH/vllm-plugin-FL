# SPDX-License-Identifier: Apache-2.0
"""Hybrid reasoning parser preserving XingChen4's special boundary tokens."""

from typing import TYPE_CHECKING

from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class XingChen4ReasoningParser(Qwen3ReasoningParser):
    """XingChen4 marks <think>/</think> as special tokenizer tokens."""

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request = super().adjust_request(request)
        # Preserve token 10 until the parser separates reasoning from content.
        request.skip_special_tokens = False
        return request
