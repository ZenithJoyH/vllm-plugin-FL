# SPDX-License-Identifier: Apache-2.0
"""Shape-owned compressed block tables: captured addresses must stay alive."""


def update_compressed_block_table(owner, table, factor):
    compressed = table[:, ::factor] // factor
    buffers = getattr(owner, "_sparse_indexer_block_tables", None)
    if buffers is None:
        buffers = {}
        owner._sparse_indexer_block_tables = buffers
    shape = tuple(compressed.shape)
    buffer = buffers.get(shape)
    if buffer is None:
        buffer = compressed.new_empty(compressed.shape)
        buffers[shape] = buffer
    buffer.copy_(compressed)
    return buffer
