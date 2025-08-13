# SGLang 完整学习指南

## 1. 项目概述

SGLang (Structured Generation Language) 是一个高效的大语言模型服务框架，专为结构化文本生成而设计。它通过创新的调度算法、内存管理和计算优化技术，提供了比传统推理框架更优越的性能。

## 2. 核心代码结构分析

### 2.1 主要目录结构

```
sglang/
├── python/sglang/           # 主要Python包
│   ├── lang/               # 前端语言API
│   │   ├── api.py         # 公共API接口
│   │   ├── ir.py          # 中间表示(IR)
│   │   ├── compiler.py    # 编译器
│   │   └── backend/       # 后端抽象
│   └── srt/               # SGLang Runtime (推理运行时)
│       ├── entrypoints/   # 引擎入口点
│       ├── managers/      # 各种管理器
│       ├── layers/        # 模型层实现
│       ├── models/        # 支持的模型
│       └── mem_cache/     # 内存缓存
├── sgl-kernel/            # C++/CUDA内核库
├── sgl-router/            # Rust路由器
├── benchmark/             # 基准测试
├── examples/              # 使用示例
└── test/                  # 测试代码
```

### 2.2 核心模块职责

#### 2.2.1 前端语言模块 (lang/)
- **api.py**: 用户面向的高级API，包含 `gen()`, `select()`, `function()` 等核心函数
- **ir.py**: 定义中间表示，包含 `SglFunction`, `SglGen`, `SglSamplingParams` 等核心类
- **compiler.py**: 将用户程序编译为执行图，支持并行执行和批处理优化

#### 2.2.2 运行时模块 (srt/)
- **managers/scheduler.py**: 核心调度器，负责请求调度和批处理管理
- **managers/schedule_batch.py**: 批处理数据结构和状态管理
- **entrypoints/engine.py**: 引擎入口，管理多进程协调
- **layers/attention/**: 注意力机制后端，支持FlashInfer、Triton等
- **mem_cache/**: 内存缓存系统，包含Radix树缓存

## 3. 核心技术实现详解

### 3.1 Prompt处理与编译流程

#### 关键代码位置：`python/sglang/lang/api.py:75-137`

```python
def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    # ... 其他参数
):
    """调用模型生成文本的核心函数"""
    return SglGen(
        name, max_tokens, min_tokens, n, stop, stop_token_ids,
        temperature, top_p, top_k, min_p, frequency_penalty,
        presence_penalty, ignore_eos, return_logprob,
        logprob_start_len, top_logprobs_num, return_text_in_logprobs,
        dtype, regex, json_schema,
    )
```

**技术亮点**：
1. **统一的生成接口**：通过 `SglGen` 对象封装所有生成参数
2. **类型安全**：支持 `dtype` 参数进行类型约束
3. **多样化约束**：支持regex、JSON schema等结构化约束

#### 编译器实现：`python/sglang/lang/compiler.py:11-148`

```python
class CompiledFunction:
    def __init__(self, tracer, function):
        self.function = function
        self.last_node = CompGraphNode(tracer.last_node)
        self.expr_to_node = {}
        self.build_graph(tracer)        # 构建计算图
        self.topological_sort()         # 拓扑排序优化执行顺序
```

**设计思想**：
- 将用户程序转换为有向无环图(DAG)
- 通过拓扑排序确定最优执行顺序
- 支持并行执行不相关的计算节点

### 3.2 推理调度核心逻辑

#### 调度器架构：`python/sglang/srt/managers/scheduler.py:1-100`

SGLang采用三进程架构：
1. **TokenizerManager** (主进程): 处理tokenization
2. **Scheduler** (子进程): 负责批处理调度和模型前向
3. **DetokenizerManager** (子进程): 处理detokenization

**核心调度算法**：
- **RadixAttention**: 基于前缀树的KV缓存共享
- **Chunked Prefill**: 将长序列分块处理，减少内存峰值
- **动态批处理**: 根据序列长度和计算需求动态调整批大小

#### 批处理数据结构：`python/sglang/srt/managers/schedule_batch.py:19-32`

```python
# 数据流转过程
ScheduleBatch -> ModelWorkerBatch -> ForwardBatch

# ScheduleBatch: 高级调度数据 (CPU)
# ModelWorkerBatch: 模型前向子集 (CPU->GPU转换)
# ForwardBatch: 低级张量数据 (GPU)
```

### 3.3 内存管理与缓存系统

#### RadixAttention缓存：`python/sglang/srt/mem_cache/radix_cache.py`

**核心特性**：
- **前缀共享**: 多个请求共享相同前缀的KV缓存
- **动态分配**: 根据序列长度动态分配内存
- **LRU淘汰**: 使用LRU策略管理缓存容量

## 4. sgl-kernel 深度解析

### 4.1 功能定位

sgl-kernel是SGLang的CUDA内核库，提供高性能的底层计算原语：

#### 核心模块：`sgl-kernel/python/sgl_kernel/__init__.py:1-103`

```python
# 主要功能模块
from sgl_kernel.attention import (
    cutlass_mla_decode,          # MLA解码注意力
    lightning_attention_decode,   # Lightning注意力
    merge_state,                 # 状态合并
)
from sgl_kernel.gemm import (
    fp8_scaled_mm,              # FP8矩阵乘法
    int8_scaled_mm,             # INT8量化GEMM
    awq_dequantize,             # AWQ反量化
)
from sgl_kernel.moe import (
    moe_fused_gate,             # MOE融合门控
    topk_softmax,               # TopK Softmax
)
```

### 4.2 核心优化技术

#### 4.2.1 注意力计算优化
- **CUTLASS MLA**: 使用NVIDIA CUTLASS库优化Multi-Latent Attention
- **Lightning Attention**: 针对长序列优化的注意力实现
- **状态合并**: 高效的注意力状态合并算法

#### 4.2.2 量化与压缩
- **多精度支持**: FP8, INT8, INT4等多种精度
- **块级量化**: Block-wise quantization减少量化误差
- **动态量化**: 运行时动态确定量化参数

#### 4.2.3 MOE优化
- **专家调度**: 高效的专家路由和负载均衡
- **融合计算**: 门控函数与专家计算的融合优化
- **通信优化**: 减少专家间的数据传输开销

### 4.3 编译构建系统

#### CMake配置：`sgl-kernel/CMakeLists.txt`

**主要特性**：
- **模块化构建**: 可选择性编译不同功能模块
- **自动依赖管理**: 自动下载和集成第三方库
- **多GPU架构支持**: 支持从sm_80到sm_90a的所有架构

## 5. FlashInfer集成分析

### 5.1 集成架构

FlashInfer作为SGLang的主要注意力后端，提供以下功能：

#### 后端实现：`python/sglang/srt/layers/attention/flashinfer_backend.py:74-100`

```python
class FlashInferAttnBackend(AttentionBackend):
    def __init__(self, model_runner, skip_prefill=False):
        # 解析常量配置
        self.decode_use_tensor_cores = should_use_tensor_core(...)
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
```

### 5.2 核心应用场景

#### 5.2.1 解码阶段优化
- **PagedKV缓存**: 分页管理KV缓存，提高内存利用率
- **批量解码**: `BatchDecodeWithPagedKVCacheWrapper` 支持高效批处理
- **Tensor Core加速**: 自动检测并使用Tensor Core加速

#### 5.2.2 预填充阶段优化
- **分块处理**: `BatchPrefillWithPagedKVCacheWrapper` 支持大序列分块
- **Ragged批处理**: 处理不同长度序列的高效批处理
- **内存优化**: 全局workspace buffer复用

### 5.3 性能优化策略

```python
# 全局工作空间缓冲区复用
global_workspace_buffer = None

# 快速路径优化：减少host-to-device拷贝
global_override_indptr_cpu = None
```

## 6. 编译与环境配置

### 6.1 依赖要求

#### 基础依赖：`python/pyproject.toml:16-50`

```toml
# 核心依赖
dependencies = [
    "aiohttp", "requests", "tqdm", "numpy", 
    "IPython", "setproctitle"
]

# 运行时依赖
runtime_common = [
    "torch",
    "transformers", 
    "flashinfer",
    "triton",
    # ... 更多依赖
]
```

#### 系统要求
- **Python**: >= 3.10
- **CUDA**: >= 11.8 (推荐12.1+)
- **PyTorch**: >= 2.0
- **GPU**: 支持Compute Capability >= 8.0

### 6.2 编译流程

#### 6.2.1 从源码编译

```bash
# 1. 克隆仓库
git clone https://github.com/sgl-project/sglang.git
cd sglang

# 2. 安装Python包
pip install -e "python[runtime]"

# 3. 编译sgl-kernel
cd sgl-kernel
pip install -e .

# 4. 可选：编译sgl-router
cd ../sgl-router
pip install -e .
```

#### 6.2.2 性能编译选项

```bash
# 启用FP4内核
python -m uv build --wheel -Ccmake.define.SGL_KERNEL_ENABLE_FP4=1

# 并行编译
CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) make build

# 使用ccache加速
export CCACHE_DIR=/path/to/ccache
make build
```

### 6.3 常见编译问题

#### CUDA 12.6兼容性问题
```bash
# 更新ptxas到12.8版本
sudo bash update_ptxas.sh 12.6
ptxas --version  # 验证版本
```

## 7. 基础使用流程

### 7.1 快速开始

#### 离线引擎使用：`examples/runtime/engine/launch_engine.py`

```python
import sglang as sgl

# 启动引擎
llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")

# 生成文本
result = llm.generate("What is the capital of France?")
print(result)

# 关闭引擎
llm.shutdown()
```

#### 前端语言API：`examples/frontend_language/quick_start/local_example_chat.py`

```python
import sglang as sgl

@sgl.function
def multi_turn_question(s, question_1, question_2):
    s += sgl.user(question_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256))
    s += sgl.user(question_2)
    s += sgl.assistant(sgl.gen("answer_2", max_tokens=256))

# 设置后端
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-chat-hf")
sgl.set_default_backend(runtime)

# 执行函数
state = multi_turn_question.run(
    question_1="What is the capital of the United States?",
    question_2="List two local attractions.",
)

# 获取结果
print(state["answer_1"])
```

### 7.2 高级功能

#### 7.2.1 结构化生成

```python
# JSON Schema约束
@sgl.function
def structured_gen(s, query):
    s += sgl.user(query)
    s += sgl.assistant(sgl.gen(
        "response", 
        max_tokens=200,
        json_schema="""{
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }"""
    ))

# 正则表达式约束
s += sgl.assistant(sgl.gen(
    "phone", 
    regex=r"\d{3}-\d{3}-\d{4}"
))
```

#### 7.2.2 批处理与流式输出

```python
# 批处理
states = multi_turn_question.run_batch([
    {"question_1": "Q1", "question_2": "Q2"},
    {"question_1": "Q3", "question_2": "Q4"},
])

# 流式输出
state = multi_turn_question.run(
    question_1="What is AI?",
    question_2="Explain it simply.",
    stream=True
)

for output in state.text_iter():
    print(output, end="", flush=True)
```

### 7.3 服务器部署

```bash
# 启动HTTP服务器
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --port 30000 \
    --host 0.0.0.0
```

## 8. 技术差异对比分析

### 8.1 vs. vLLM

| 特性 | SGLang | vLLM |
|------|--------|------|
| **核心设计** | 结构化生成优先 | 通用LLM服务 |
| **内存管理** | RadixAttention + 前缀缓存 | PagedAttention |
| **调度策略** | 动态图执行 | 静态批处理 |
| **约束生成** | 原生支持JSON/Regex | 需要额外库 |
| **长上下文** | Chunked Prefill | 标准处理 |
| **多轮对话** | 高效前缀共享 | 重复计算 |

### 8.2 vs. TensorRT-LLM

| 特性 | SGLang | TensorRT-LLM |
|------|--------|--------------|
| **编程模型** | Python-first | C++/CUDA |
| **灵活性** | 高度可定制 | 固定Pipeline |
| **部署复杂度** | 简单 | 复杂 |
| **性能优化** | 算法创新 | 底层优化 |
| **开发效率** | 快速迭代 | 编译周期长 |

### 8.3 核心技术优势

#### 8.3.1 RadixAttention
传统框架每个请求独立计算，SGLang通过前缀树共享KV缓存：
```
传统方式：
Request1: [system] + [user1] + [assistant1] 
Request2: [system] + [user2] + [assistant2]  // 重复计算system

SGLang方式：
         [system] (共享)
           /    \
    [user1]    [user2]
      |          |
 [assistant1] [assistant2]
```

#### 8.3.2 动态图执行
- **编译期优化**: 将用户程序编译为执行图
- **运行期调度**: 根据依赖关系并行执行
- **内存复用**: 智能的内存分配和复用策略

#### 8.3.3 多级缓存体系
1. **L1 - Token缓存**: 细粒度的token级别缓存
2. **L2 - Prefix缓存**: 序列前缀的KV缓存
3. **L3 - 模型缓存**: 模型参数和中间状态缓存

## 9. 深入学习建议

### 9.1 代码阅读路径

**初学者路径**：
1. `examples/` - 从示例开始理解用法
2. `python/sglang/lang/api.py` - 理解前端API设计
3. `python/sglang/lang/ir.py` - 学习中间表示
4. `python/sglang/srt/entrypoints/engine.py` - 理解引擎架构

**进阶路径**：
1. `python/sglang/srt/managers/scheduler.py` - 深入调度算法
2. `python/sglang/srt/mem_cache/radix_cache.py` - 理解缓存机制
3. `sgl-kernel/csrc/` - 学习CUDA内核实现
4. `python/sglang/srt/layers/attention/` - 注意力机制实现

### 9.2 关键技术点深入研究

#### 9.2.1 内存管理 (`python/sglang/srt/mem_cache/`)
- **RadixCache**: 基于前缀树的KV缓存实现
- **ChunkCache**: 分块内存管理策略
- **MemoryPool**: 内存池分配器设计

#### 9.2.2 计算优化 (`sgl-kernel/csrc/`)
- **Attention优化**: CUTLASS、FlashAttention集成
- **GEMM优化**: 多精度矩阵乘法实现
- **MOE优化**: 专家网络调度算法

#### 9.2.3 系统架构 (`python/sglang/srt/managers/`)
- **多进程协调**: 进程间通信设计
- **负载均衡**: 请求分发策略
- **故障恢复**: 错误处理和恢复机制

### 9.3 实验与调试

#### 性能分析工具
```python
# 开启性能分析
import sglang as sgl
sgl.set_profile_path("/tmp/sglang_profile")

# 内存使用监控
print(runtime.get_server_info())

# 缓存命中率统计
print(runtime.get_cache_stats())
```

#### 调试技巧
```bash
# 详细日志
export SGLANG_LOG_LEVEL=DEBUG

# 内存调试
export CUDA_LAUNCH_BLOCKING=1

# 内核性能分析
nsys profile python your_script.py
```

## 10. 总结

SGLang通过以下创新技术实现了卓越的性能：

1. **架构创新**: 三进程分离架构，专业化分工
2. **算法创新**: RadixAttention前缀缓存共享
3. **系统创新**: 动态图执行和智能调度
4. **工程创新**: 模块化设计和高度可扩展性

学习SGLang不仅能掌握现代LLM服务技术，更能深入理解高性能系统设计的精髓。建议从基础API开始，逐步深入核心算法和系统实现，最终形成对整个框架的全面理解。

---

*本指南基于SGLang v0.5.0版本，持续更新中。如有问题或建议，欢迎反馈。*