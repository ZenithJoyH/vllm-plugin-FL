# GLM-5.3-Flash / GLM5-Next

本分支提供 GLM-5.3-Flash checkpoint（`glm5_next` / `glm5_next_text`）的插件实现。
基于 `030plugin-for-day0` 的 `e2f51dcd0e03ea81e6f156b4c5cba920e5ce9d36`，
不是将运行容器的整个脏工作树合入 `main`。此路径依赖 **vLLM 0.24**，不能照用
仓库通用 Quick Start 中较旧的 vLLM 版本。安装沿用该平台已有的插件构建流程。

## 验证状态

2026-08-31 在 T-Head PPU-ZW810E、Python 3.12、Torch 2.10.0、
vLLM 0.24.0+empty、FlagGems 5.3.5、Transformers 5.8.1 环境完成：

| 范围 | 结果 |
| --- | --- |
| GLM 配置、导入、注册、黑名单、平台边界、MoE clamp | 27 项通过 |
| 原框架 dispatch 单元测试 | 275 项通过：265 项原测试 + 10 项输入能力/错误传播测试 |
| H128、MHC、尾池、clamp、KDA、MQA、cache、metadata | 28 项通过：26 项组件检查，2 项FP8能力拒绝检查；不替代模型验收 |
| NVIDIA | 仅惰性注册与模拟 top-k 分派检查；未运行 CUDA kernel 或模型 |
| 新分支整模型启动、正式 GPQA、多模态、MTP、性能 | 未验收 |

测试没有安装到现有插件、启动模型服务或发送模型请求。历史正式 GPQA 198 题未完成，
不能用组件结果或历史错误子集成绩代替最终精度。旧环境中 QKV/MHC FP32 候选已有
局部因果和组件证据，但完整模型精度尚未闭环；本分支不携带其临时环境开关或全局
替换，也不默认启用这些候选。**本分支不是运行环境所有数值实验的等价快照。**

## 代码边界

| 层次 | 职责与入口 |
| --- | --- |
| 模型 / 配置 | `models/glm5_next*.py`、`configs/glm5_next.py`：KDA、稀疏 MLA、MoE、MHC、权重与状态流 |
| 算子契约 | `ops/glm5_next.py`、`kernels/glm5_next/indexer_backend.py`：通过 `CachedOp` 请求同名算子，不判断 NVIDIA/非 NVIDIA |
| 分派与注册 | `dispatch/backends/model_ops.py`：沿用 registry、policy、backend availability 和惰性载入 |
| 通用实现 | `flaggems/impl/{mhc,sparse_indexer,recurrent_kda,bounded_activation}.py`；reference 独立注册 |
| 厂商实现 | `vendor/cuda/impl/{model_ops,sparse_indexer}.py` 原生 CUDA；`vendor/thead/impl/{recurrent_kda,gather_cache,paged_mqa,causal_conv1d}.py` 保底 |
| 公共模型 kernel | `kernels/glm5_next/`：kpool 压缩、逐请求尾池、bounded KDA gate、metadata buffer 生命周期 |
| 版本兼容 | `patches/glm5_next*_v024.py`：GLM 配置、cache spec、metadata 和 reshape 的 v0.24 适配 |

- 厂商必须通过注册明确支持；未知厂商不能落入 PPU 分支。ROCm 的普通稀疏索引
  不具备这里的 kpool/tail 语义，因此明确报不支持，不提供表面可用的替代路径。
- cache/query dtype、压缩页约束由后端提供。PPU 的 CUDA 兼容 capability 不当作
  NVIDIA SM；PPU 页约束为 32，CUDA SM90 为 64、SM100 为 32。
- CUDA persistent top-k 和 DeepGEMM 依赖检查在 CUDA 后端；公共索引流程不直调 `_C`。
- 不修改上游 vLLM 源文件，不全局替换 MHC `forward_oot` 或 `concat_mla_q`。
  GLM 专属 attention subclass 处理 NoPE；runner wrapper 对非 GLM 原样委派。
  v0.24 的 cache/metadata 兼容入口带 GLM 类型或专属 backend 标记检查。
- PPU 的 MLA prefill 注册只在 GLM 配置校验入口启用；它仍是进程级注册，尚未验证
  同一进程轮换不同模型。不能据此宣称所有模型的整服务兼容性。
- MoE 公共代码只补齐已有 `gemm1_clamp_limit` 的传播：未配置 clamp 时保留原路径；
  已配置时绕开会丢失 clamp 的整段 fused 快路径，经中央算子分派执行有界激活。

## 保留的正确性修复

1. Q 与压缩 K 使用相同归一化 H128 基底；PPU 旋转以 FP32 计算再转回原 dtype。
2. packed-prefill 按每条请求的槽位持久化未满池的 raw K 与 gate，不能按整个 batch
   的余数推导；decode 延续正确的请求尾池。
3. 每种 graph shape 保留自己的 block-table buffer，replay 只原位更新，防止旧 graph
   引用后续捕获已释放的地址。
4. MHC 契约包含恰好一次 RMSNorm；FlagGems / reference 的 pre 与 fused post-pre
   都遵守该契约，不遗漏 checkpoint 的 norm weight。
5. 有界 KDA gate、SwiGLU clamp、NoPE query assembly 和压缩缓存布局保持模型语义。

本次排除了源容器中与此模型无关的 Qwen/W8A8 改动、trace/探针、失败候选、备份、
跳过 KV 写入的诊断开关、`provider.py` 环境 A/B 分支及不可达重复实现。
没有删除源容器中的原文件，也没有改变其他模型的全局黑名单。

## 后端与 graph 限制

当前实际 FlagGems 包为5.3.5，安装树无Git元数据；历史revision不能当作本次身份。
本次以安装源码SHA256核验，详见 [`BACKEND_CONTRACTS.md`](BACKEND_CONTRACTS.md)。
已存在且接口匹配的 Hadamard/MHC/MLA/top-k 等由相应 backend 接入；不在模型里
绕过分派直接调用 FlagGems。raw K+gate 双平面逐请求尾池没有兼容实现，使用
`kernels/glm5_next/prefill_tail.py`；pinned FP8 gather 与 PPU BF16 kpool cache 不兼容，
由 `vendor/thead/impl/gather_cache.py` 负责。算子来源见
[`PROVENANCE.md`](../../vllm_fl/kernels/glm5_next/PROVENANCE.md)。

外层索引和部分 KDA 路径依赖 v0.24 的 `eager_break_during_capture`。graph 模式需要
`VLLM_USE_BREAKABLE_CUDAGRAPH=1`，不代表整个模型所有操作都在一张无中断 graph 内。
FlagGems及reference的公开pack/unpack会读取CPU长度，只能在这类eager区间使用。
调用前守卫在capture中拒绝它们，不会尝试CPU回退。FlagGems clamp 的
标量 buffer 必须在正常 eager warm-up 时准备。后端在调用前确定实现；不捕获 kernel
`RuntimeError` 后再重试写缓存操作。新增 Triton 的全面 dtype/布局覆盖仍需逐算子补充，
本轮只声称测试中列出的组合通过。

尾池测试覆盖 BF16/FP16/FP32 × B1/2/4/8/16/32，包含改变请求边界、页面、偏移、
非连续布局、padding/越界槽位和每组 3 次 replay。其他独立夹具使用 H128 数学参考、
MHC CPU 参考和改变输入的 graph replay；不是只检查 eager 等于 graph。

## 启动示例（需要单独整模型验证）

显式选择模型 profile；`VLLM_FL_CONFIG` 是框架原有配置入口，不是 plugin 搜索路径。
`thead.yaml` 保留固定环境的黑名单，仅对使用该文件的进程生效，不建议复制为
所有平台的默认配置。原量化gather黑名单已由显式缓存布局守卫替代，BF16仍走vendor。
两个示例均以FlagGems为首选；NVIDIA硬件验证仍待完成。

以下从插件仓库根目录执行；模型路径、TP、内存及通信设置按实际硬件调整：

```bash
VLLM_FL_CONFIG="$PWD/examples/glm5_next/thead.yaml" \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve /path/to/GLM-5.3-Flash-BF16 \
  --served-model-name glm-5.3-flash \
  --trust-remote-code \
  --tensor-parallel-size 16 \
  --distributed-executor-backend mp \
  --host 127.0.0.1 --port 18021 \
  --max-model-len 100000 --max-num-seqs 32 \
  --gpu-memory-utilization 0.90 \
  --reasoning-parser glm45 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

eager 验证将 `--compilation-config ...` 换为 `--enforce-eager`。无需 overlay、
`PYTHONPATH` 或 `PYTHONDONTWRITEBYTECODE`。本次没有执行以上模型启动命令。

## 复现聚焦测试

在已有兼容运行时的插件源码目录执行，不需要模型权重：

```bash
python -m pytest tests/unit_tests/test_glm5_support.py -q --confcutdir=tests/unit_tests
python -m pytest tests/unit_tests/dispatch -q --confcutdir=tests/unit_tests/dispatch
python -m pytest tests/unit_tests/test_glm5_tail_graph.py tests/unit_tests/test_glm5_precision_graph.py tests/unit_tests/test_flaggems_bindings_graph.py -q --confcutdir=tests/unit_tests
```

最后一组需要空闲 PPU 设备；CPU-only 或 NVIDIA 不算作这组 PPU 数值验证。
后续应先做此新分支的整模型 eager/graph 回归，再进行正式 198 题 C32 GPQA；正式
精度达标前不开始最终性能验收。
