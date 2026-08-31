# 算子提供者与调用契约

模型只调用 `CachedOp`。FlagGems 实现注册为 `DEFAULT/default.flagos`，原有厂商
实现注册为 `VENDOR/vendor.cuda` 或 `vendor.thead`，数学参考为 `REFERENCE`。
默认顺序为 FlagGems → vendor → reference；显式 `op_backends` 和黑名单仍优先。
vendor 内没有另一个 FlagGems/reference 选择器。

## 已有 FlagGems 优先

| 插件契约 | FlagGems 入口与兼容封装 | 保底 / 限制 |
| --- | --- | --- |
| `fused_recurrent_kda` | `FLA/fused_recurrent` 通用 kernel，设置 `IS_KDA=True`；不使用写死False的GDN公开wrapper | PPU原有Triton保留；当前绑定BF16 QKV、FP32 gate/state、H4 D128、单token decode；非连续state走vendor |
| `sparse_indexer_rotate_indexer_query` | FP32 Hadamard128，再转回输入dtype | CUDA原生或独立reference |
| `sparse_indexer_prepare_query` | `per_token_group_quant_fp8`；映射`use_ue8m0`到`scale_ue8m0` | 输出dtype显式来自存储契约；PPU BF16+identity scale，不伪称FP8 |
| `sparse_indexer_indexer_k_quant_and_cache` | 已有量化writer；明确uint8缓存、128维+4字节scale | 当前PPU不支持原生e4m3算术，调用前拒绝；BF16 kpool用独立writer |
| `sparse_indexer_gather_cache` | uint8量化页调用已有gather；scale接口按字节宽度适配 | BF16页调用PPU原有Triton；FP8字节搬运不要求FP8算术 |
| `sparse_indexer_mqa_logits` | 已有`fp8_fp4_mqa_logits`，BF16同样兼容；Q scale预先合入weights | 独立reference / CUDA DeepGEMM；不将FP4或多group scale误当作单scale |
| `sparse_indexer_paged_mqa_logits` | 已有BF16 64页、H32/64 D128 kernel；关闭公开wrapper的CPU清理，以设备侧mask完成clean_logits | 32页等不兼容布局走PPU原有Triton；量化CUDA页走CUDA |
| `sparse_indexer_topk_prefill/decode` | 已有per-row top-k；统一额外max_seq_len元数据契约 | CUDA原生 / reference |
| `sparse_indexer_pack_seq/unpack_seq` | 已有公开封装，正常eager调用 | 输出shape依赖设备长度，所有现有wrapper均禁止capture；只用于eager-break区间 |
| `mhc_pre_with_norm` / `mhc_fused_post_pre_with_norm` | 现有MHC数学内核，明确追加且仅追加一次RMSNorm | CUDA原生 / reference；不是全局替换上游MHC |
| `silu_and_mul_with_clamp` | 现有kernel，eager预热并缓存device scalar供graph重放 | 非BF16或非默认alpha/beta使用reference；无PPU专属FlagGems封装 |

KDA非空请求的state ID必须合法、互不重复；空padding请求通过重复cu_seqlens边界
在内核中跳过，可携带-1。插件不通过`.item()`验证设备元数据。该条件是调度器输入
契约，不能将任意越界ID当作受支持输入。speculative、多token recurrent、其他head
配置不在当前FlagGems绑定的验收范围内。已有vendor的额外padding防护继续保留。

## 调度与副作用

`OpImpl.supports`只检查shape/dtype/layout/执行模式等调用前元数据，False只跳过本次
候选，不将实现加入失败缓存。`CachedOp`对带守卫的算子每次重新检查，不能将64页
选出的实现错误复用到32页。strict仍禁止执行失败后的重试，但不禁止调用前排除
不兼容实现。显式只允许FlagGems时，不兼容输入会报错，不越过用户后端列表。

本次注册的实现均设置`allow_runtime_fallback=False`。已启动内核抛错时立即传播，
不能假设缓存尚未修改而换另一份实现重新执行。没有吞掉运行错误或改变原通用算子
的默认错误回退行为。

## 版本证据与保留实现

本次核验安装FlagGems5.3.5，源码SHA256：

- recurrent: `c756b775eadb90866a9857a22be9e8d4dd0f6c7fb13d84ec108ee7ac7c2a6735`
- BF16 paged-MQA: `4864884c70face79c3eb9688a59f69bc92f5acfb8bfe0892b875126bc8d73ec7`
- gather: `30893e8d131154c02bbc6358e983e7b969e64267bc42c684ad5352ae4190fd3a`
- query quant: `6ec0fd142804f30fb8a3768b44583b9c3b5d1268636ed5a1094e3eea4d968666`

causal-conv状态更新、raw K+gate双平面尾池及kpool没有在该安装版本找到等价入口，
原实现保留。不泛称所有FlagGems版本都缺失这些能力。GLM模型、配置、processor、
NoPE attention subclass和v0.24 cache/runner适配仍保留模型名称，通用算子不再加glm5前缀。

测试中的原生FP8 query quant/cache writer在PPU上验证的是**调用前拒绝**，不是
FP8数值通过。相关FlagGems绑定需要在兼容CUDA设备上进一步数值/graph验证。
未修改现有安装、服务、权重、评分或上游vLLM；新分支整模型和正式精度尚未验收。
