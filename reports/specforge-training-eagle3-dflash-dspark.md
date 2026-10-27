# SpecForge 源码报告：EAGLE3、DFlash、DSpark 的输入、计算与训练

- **EAGLE3**：三层目标特征融合，单层草稿网络自回归展开；训练中回灌自己的 hidden state，学习多步预测。
- **DFlash**：目标特征注入每个草稿层的 K/V；一次前向预测整个 masked block，以带位置权重的 token CE 训练。
- **DSpark**：沿用 DFlash 并行骨干，增加轻量串行 Markov/RNN head 和接受概率预测；训练结合 CE、词表分布 L1 与 confidence BCE。

本文依据 **2026-10-27** 拉取的 [SpecForge `e3ab28d681df457f2a04e59dfac8117c2c4cbfef`](https://github.com/sgl-project/SpecForge/tree/e3ab28d681df457f2a04e59dfac8117c2c4cbfef)。SGLang 运行时索引依据 [`8252e3af630c103e960e13a77e570e810f8e3eb8`](https://github.com/sgl-project/sglang/tree/8252e3af630c103e960e13a77e570e810f8e3eb8)。这是静态源码走读，未执行 GPU 训练、导出兼容性实验或性能复现。下文具体层数、宽度和超参数均来自指定 Qwen3-8B 配方，不代表所有模型的固定设置。

## 1. 推理时真正输入什么

先固定一个边界：目标模型已经计算完 `x1 ... x(a-1)`，并生成确定的下一个 token `x(a)`。这个刚生成、尚未作为输入经过目标模型的 token，称为 **anchor / bonus token**。草稿模型现在要预测 `x(a+1), x(a+2), ...`。

此时可用的是：

- 目标模型在位置 `< a` 的中间 hidden states；包括 prompt，以及此前被接受的回答前缀。
- 确定的 anchor token `x(a)`。
- 草稿模型自己的 KV cache、position IDs 和 attention mask。

**还没有 `x(a)` 的目标 hidden state，也没有未来 token 的目标特征。** 首轮 target prefill 提供的是上述前缀特征和新生成的 anchor token；后续轮通过 target verification 更新。被拒绝的后缀不能当作已确认前缀继续保留，相关缓存需要按接受边界处理。

### 1.1 符号、shape 与 dtype

| 符号 / 张量 | 含义 | 典型 shape / dtype |
| --- | --- | --- |
| `B` | batch 中的样本或请求数 | 整数 |
| `S` | 目标特征序列长度 | 整数 |
| `Ht`, `Hd` | 目标、草稿 hidden width | Qwen3-8B 配方均为 4096 |
| `M` | 抽取的目标层数 | EAGLE3 为 3；本文 DFlash/DSpark 配方为 5 |
| `K` | `block_size`，草稿输入块长度 | DFlash 为 16；DSpark 为 7 |
| `A` | 每条训练序列采样的 anchor 数 | 最多 `num_anchors`，不等于 batch size |
| `V`, `Vd` | 目标、草稿输出词表大小 | 本文目标为 151936；EAGLE3 草稿为 32000 |
| `input_ids` | 文本 token IDs | `[B,S]`，整数，常用 int64 |
| `target_hidden` | 多个目标层沿最后一维拼接的特征 | `[B,S,M*Ht]`，配方使用 bfloat16 |
| `noise_embedding` | anchor 和 mask tokens 的 embedding | 推理 `[B,K,Hd]`；训练 `[B,A*K,Hd]` |
| `loss_mask` | 哪些位置参与监督 | `[B,S]`，0/1；不是 attention mask |
| `p`, `q` | 同一位置的目标、草稿词表概率分布 | `[V]` 或相应的缩减词表分布 |

不同方法的关键区别如下：

| 项目 | EAGLE3 | DFlash | DSpark，本文默认 Markov 配方 |
| --- | --- | --- | --- |
| token 输入 | 当前已知 token，逐步推进 | anchor + `K-1` 个 mask | anchor + `K-1` 个 mask |
| 特征输入 | 首步用目标多层特征；后续复用草稿状态 | 目标多层特征作为每层 K/V 上下文 | 同 DFlash |
| 重骨干执行 | 每个草稿深度一次 | 整块一次 | 整块一次 |
| 实例骨干 | 1 个 decoder layer | 5 个 decoder layers | 5 个 decoder layers |
| 输出 head | 自己训练的缩减词表 LM head | 冻结目标 LM head | 冻结目标 LM head + 训练的 Markov head |
| 块内 token 采样 | 自回归 | 给定骨干输出后，各位置没有已采样 token 的显式条件依赖 | Markov head 从左到右采样 |
| 主要监督 | 目标软分布，多步展开 | 序列 token 的 hard-label CE | hard-label CE + 目标概率分布 L1 + confidence BCE |

## 2. EAGLE3：融合目标特征，再逐步回灌草稿状态

### 2.1 模块和实际调用链

模型是 `LlamaForCausalLMEagle3`。核心接口分成三段：

```python
project_hidden_states(hidden_states)
backbone(input_embeds, hidden_states, cache_hidden,
         attention_mask, position_ids, past_key_values=None, use_cache=True)
compute_logits(hidden_states)
```

对应模块：

1. `embed_tokens`：完整目标词表的 embedding，通常从目标模型复制，训练中冻结。
2. `fc`：把三层目标特征从 `3*Ht` 投影到 `Hd`；可配置投影前分别做 RMSNorm。
3. `midlayer`：一个 decoder layer，包含 attention、RoPE、RMSNorm、SwiGLU MLP 和 residual。
4. `norm`、`lm_head`：输出归一化与可训练草稿词表投影。
5. `t2d`、`d2t`：目标词表与草稿词表映射。

目标层由 `resolve_eagle_capture_layers()` 解析。没有显式配置时，零基 decoder layer IDs 为 `[1, L//2-1, L-4]`；目标有 36 层时为 `[1,17,32]`。不能把这组 hook 层号直接当作 Hugging Face `output.hidden_states` 的 tuple 下标，后者还包含 embedding 输出。

来源：[模型定义](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/llama3_eagle.py#L1493-L1719)、[模型装配和层选择](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/model_providers.py#L89-L213)。

### 2.2 一步如何计算

用列向量记号表示单个位置，先融合目标特征：

$`\displaystyle g_{a-1}=W_f[\,h_{a-1}^{\mathrm{low}};h_{a-1}^{\mathrm{mid}};h_{a-1}^{\mathrm{high}}\,],\qquad W_f\in\mathbb{R}^{H_d\times 3H_t}`$

将 anchor embedding 和融合后的状态分别归一化，再拼接：

$`\displaystyle u_a=[\,\operatorname{RMSNorm}(E(x_a));\operatorname{RMSNorm}(g_{a-1})\,]\in\mathbb{R}^{2H_d}`$

这个 `2*Hd` 输入进入 Q/K/V 投影。attention 输出回到 `Hd`，与原状态相加，再经过 MLP 和第二次 residual：

$`\displaystyle r_a=g_{a-1}+\operatorname{Attention}(u_a;\mathrm{draft\ cache}),\qquad z_a=r_a+\operatorname{MLP}(\operatorname{RMSNorm}(r_a))`$

$`\displaystyle q(x_{a+1}\mid\mathrm{prefix},x_a)=\operatorname{softmax}\!\left(W_{\mathrm{draft}}\operatorname{RMSNorm}(z_a)\right)`$

上式对应默认 `norm_output=false` 的归一化放置；代码也支持把输出 norm 放进 backbone。抽取 token 后，下一步将 `z_a` 与新 token 的 embedding 继续送入同一个草稿层。**草稿状态不要求等于目标模型的下一位置 hidden state。**

Qwen3-8B 配方的维度变化为：

```text
3 × 4096 目标特征 → fc → 4096
4096 token embedding 与 4096 状态拼接 → 8192
Q/K/V + attention + residual + MLP → 4096
草稿 LM head → 32000 logits → d2t 映射到目标 token ID
```

缩减词表按训练监督位置的 token 频率选取，`t2d` 是目标词表上的选择 mask。**`d2t` buffer 保存的是偏移量**：`target_id = draft_id + d2t[draft_id]`，不能直接把 `d2t[draft_id]` 当作目标 ID。缩减词表节省输出投影成本，也使训练数据的词汇覆盖影响草稿可提案范围；token 输入 embedding 仍覆盖完整目标词表。

来源：[Qwen3-8B EAGLE3 配置](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/configs/qwen3-8b-eagle3.json)、[词表统计及映射](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/data/preprocessing.py#L599-L652)。

### 2.3 怎么训练：TTT 到底展开什么

`OnlineEagle3Model.forward()` 接受 token IDs、三层拼接特征、目标 logits、padding mask 与 loss mask。这里的类名 `Online` 不等于只能在线训练；离线特征同样进入这个训练模型。

训练流程为：

1. 冻结目标模型，对完整训练序列做 causal forward，获得辅助特征和每个位置的目标分布。
2. 首次用 `fc` 融合目标特征，然后执行草稿 backbone。
3. 将本步草稿输出 `hidden_states_out` 作为下一展开步的 hidden 输入。
4. token IDs、目标分布和监督 mask 随展开步移位，并用配套 attention/cache 逻辑维护各条展开路径的依赖。
5. 每一步都计算 token 分布监督，再加权求和、反向传播。

**当前实现回灌的是草稿 hidden state；token 输入仍来自训练序列的移位，不能描述成“把草稿采样 token 完整地在线 rollout 后再训练”。** 代码中没有把本步 `argmax` token 写回下一步 `input_ids`。这也是 TTT 与普通一次 teacher-forcing forward 的区别：后续步骤必须适应草稿自己产生的状态。

默认 token loss 是目标软标签交叉熵，与最小化目标到草稿的 KL 对草稿参数有相同梯度。省略 batch/位置归一化后：

$`\displaystyle \mathcal{L}_j=-\sum_t m_{t,j}\sum_{v\in\mathcal{V}_d}\widetilde p_{t,j}(v)\log q_{t,j}(v),\qquad \mathcal{L}=\sum_{j=0}^{T-1}0.8^j\mathcal{L}_j`$

其中 `m` 是有效监督 mask，`T` 是 `ttt_length`。缩减词表训练使用在选中词表上重新归一化的目标分布；目标 top-1 不在草稿词表内的位置被排除出该监督 mask。接受率计算则保留目标在草稿词表上的原概率质量，不把它误当作完整目标分布。

本文配方 `ttt_length=7`。训练 `fc`、decoder、norm 和草稿 LM head；目标模型与复制的 embedding 冻结。TTT 不是推理时更新权重。

可选 **LK loss** 将接受率直接纳入目标。对同一上下文中的规范化分布：

$`\displaystyle \alpha(p,q)=\sum_v\min(p_v,q_v)=1-\frac12\lVert p-q\rVert_1`$

源码支持例如 `-log(alpha)` 和 KL/接受率混合目标；这些是可选 loss，不是 EAGLE3 模型必须新增的结构。

来源：[TTT 循环](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/eagle3/model.py#L266-L627)、[多步 loss 汇总](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/training/strategies/base.py#L270-L340)、[LK loss](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/core/lk_loss.py)。

## 3. DFlash：一次前向生成整个块

### 3.1 输入和模块

骨干类 `DFlashDraftModel` 的接口为：

```python
forward(position_ids, attention_mask=None, noise_embedding=None,
        target_hidden=None, past_key_values=None, use_cache=False, **kwargs)
```

其本体包括：多层特征融合 `fc`、`hidden_norm`、若干 `Qwen3DFlashDecoderLayer`、RoPE 和输出 RMSNorm。输入 embedding 与输出 LM head 由外部提供，训练 wrapper 从目标模型加载这两部分并冻结。

本文配方抽取目标层 `[1,9,17,25,33]`，得到 `[B,S,20480]`，再压缩成 `[B,S,4096]`。块输入为：

```text
[ E(anchor), E(mask), E(mask), ... ]     # 总长 K=16
```

`noise_embedding` 在这里指 mask token 的 embedding，不是高斯噪声。该路径一次前向完成 masked block 预测，不需要逐 token 重跑 Transformer，也不是必须多轮迭代去噪。

### 3.2 KV injection 如何计算

设多层目标特征融合结果为 `C`，某个草稿层的块表示为 `Z`。融合与 attention 的主要计算如下，矩阵记号使用行向量布局：

$`\displaystyle C=\operatorname{RMSNorm}\!\left(\operatorname{Concat}_{\mathrm{feature}}(H^{(1)},\ldots,H^{(M)})W_f^{\mathsf T}\right)`$

$`\displaystyle Q=ZW_Q,\qquad \mathsf{K}=\operatorname{Concat}_{\mathrm{seq}}(CW_K,ZW_K),\qquad \mathsf{V}=\operatorname{Concat}_{\mathrm{seq}}(CW_V,ZW_V)`$

$`\displaystyle \operatorname{Attn}(Z,C)=\operatorname{softmax}\!\left(\frac{\operatorname{RoPE}(Q)\operatorname{RoPE}(\mathsf{K})^{\mathsf T}}{\sqrt{d_h}}+\mathcal{M}\right)\mathsf{V}`$

这里直立体 Key/Value 符号表示张量，区别于 block size `K` 和词表大小 `V`；`d_h` 是单个 attention head 的宽度，`M` 的花体符号是允许位置为 0、禁止位置为负无穷的加性 mask。实际实现还有 Q/K RMSNorm、GQA 多头展开和输出投影。每层随后执行 attention residual、MLP residual。

**Q 只来自草稿块；K/V 同时来自目标上下文和草稿块。** 目标特征经过草稿层自己的 K/V 投影，不是直接把目标模型原始 KV cache 拷贝过来。每层都能重新读取目标信息，而非只在第一层融合一次。

默认全局 attention 层允许同一草稿块内双向注意力。它仍然不能读取 anchor 及之后的目标特征，也不能读取其他训练块。因而“双向”不等于看到未来真实答案。

来源：[模型及 Q/K/V 计算](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/dflash.py#L381-L442)、[decoder 和骨干 forward](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/dflash.py#L594-L939)、[attention mask](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/dflash_mask.py)。

### 3.3 训练如何构造 block

`OnlineDFlashModel._sample_anchor_positions()` 在 loss mask 中选择 **anchor 与下一个 token 均受监督** 的位置，最多选 `num_anchors` 个，随机选择后按位置排序。本文配方最多 512 个，实际数量受每条回答长度限制。

对每个 anchor `a`：

```text
原文：     ... x(a-1) | x(a)  x(a+1) x(a+2) ...
块输入：                x(a)  [MASK] [MASK] ...
监督标签：              不计   x(a+1) x(a+2) ...
目标特征可见范围：位置 < a
```

训练时可以一次算出完整干净序列的目标特征，因为目标本身是 causal，且训练 attention mask 将每个块的可见上下文限制为 `< anchor`。所有 anchor 块拼到 `[B,A*K,Hd]`，用稀疏 mask 一次处理，块之间互不可见。

`forward()` 对 block offset `k` 使用标签 `x(a+k)`，排除 `k=0` 的 anchor；其余位置同时检查越界、padding 和 assistant loss mask。**所以 DFlash 配置 `block_size=16`，这个接口实际监督 15 个未来 token。**

### 3.4 loss 和训练参数

默认目标是 hard-label CE。设 `m(a,k)` 为有效 mask，位置权重为：

$`\displaystyle w_k=\exp\!\left(-\frac{k-1}{\gamma}\right),\qquad k=1,\ldots,K-1`$

$`\displaystyle \mathcal{L}_{\mathrm{DFlash}}=\frac{\sum_{a,k}m_{a,k}w_k[-\log q_{a,k}(x_{a+k})]}{\sum_{a,k}m_{a,k}w_k}`$

`gamma` 对应 `loss_decay_gamma`；不配置或非正时不做这种衰减。靠前 token 权重较高，因为投机验证按连续前缀接受，前面失败后，后面的正确 token 也无法使用。

训练融合投影、5 层草稿骨干和相关 norm；冻结目标模型、目标 embedding 与目标 LM head。**标准 DFlash loss 不需要拿目标最终 hidden state 做特征回归或软分布蒸馏。** 当前 online capture 可额外记录最终 hidden state 计算 teacher telemetry，但这不改变标准训练目标；离线标准 DFlash 数据可以没有它。

可选 **D-PACE** 改变 CE 的位置权重。源码从草稿对正确 token 的概率 `s(k)` 出发，使用 `[0,1]` 内的平滑参数 `lambda_s`，先平滑，再累计前缀概率和后缀贡献。无效槽保持前缀积不变、不贡献后缀奖励：

$`\displaystyle \bar s_i=\begin{cases}\lambda_s+(1-\lambda_s)s_i,&m_i=1,\\1,&m_i=0,\end{cases}\qquad P_j=\prod_{i=1}^{j}\bar s_i`$

$`\displaystyle w_k^{\mathrm{D\text{-}PACE}}=\sum_{j=k}^{K-1}m_jP_j`$

权重在 `no_grad` 中计算，再参与 CE；另有仅前缀置信度、仅 continuation value 的消融变体。D-PACE 的归一化是每条序列先按有效 anchor 数平均，再按有效序列数平均，不是除以整个 batch 的有效 anchor 总数，也不能套用前文普通 DFlash 的加权 token 分母。它不是再加一套草稿网络。

来源：[anchor、块构造和训练 forward](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/common/dflash_family_model.py#L335-L407)、[block 前向](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/common/dflash_family_model.py#L555-L645)、[标签及 mask](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/common/dflash_family_model.py#L1274-L1349)、[D-PACE 权重](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/core/position_weights.py)。

## 4. DSpark：并行骨干 + 串行小 head + 接受率预测

### 4.1 相比 DFlash 增加哪些模块

`DSparkDraftModel` 继承 `DFlashDraftModel`，保留相同的特征融合和 KV injection 骨干，新增：

- `markov_head`：默认 `VanillaMarkovHead`，也有 gated、RNN 变体。
- `confidence_head`：`AcceptRatePredictor`，本质为一个输出标量的 Linear。

本文 Qwen3-8B 配置使用 5 层 GQA 骨干、`markov_rank=256`、启用 confidence，并把 Markov embedding 纳入 confidence 输入。

### 4.2 Markov head 怎样建立 token 依赖

本节用块内局部编号：`x0` 为 anchor，`x1 ... xK` 为待预测 token，区别于全文开头的绝对序列编号。并行骨干一次输出各位置状态 `z(k)`，冻结目标 LM head 先给出基础 logits：

$`\displaystyle u_k=W_{\mathrm{target}}z_k`$

Markov head 增加一个依赖“前一个实际 token”的词表 bias：

$`\displaystyle b_k=W_2\,\operatorname{Embed}_{W_1}(x_{k-1}),\qquad W_1\in\mathbb{R}^{V\times r},\quad W_2\in\mathbb{R}^{V\times r}`$

$`\displaystyle q_k=\operatorname{softmax}(u_k+b_k)`$

这里 `W2` 按 PyTorch Linear 的权重存储形状书写；等价的行向量矩阵乘法是 `W1[token] @ W2.T`。相当于用秩为 `r` 的分解表示一个巨大的 `V × V` token 转移矩阵。

推理时先采样 `x1`，再用它计算下一位置 bias，依次向右。**顺序执行的是 embedding lookup、小 head 和采样；5 层 Transformer 骨干不按 token 重跑。** 这就是“半自回归”。训练则给 Markov head 使用真实前驱 token，因此默认 Markov 版本可以并行计算所有位置的 bias；RNN 变体需要维护自身的递归状态。

### 4.3 DFlash 与 DSpark 的标签错位不同

当前 DSpark 训练使用 `anchor + 1 ... anchor + K` 作为 K 个标签，连 anchor 输入槽的输出也参与“下一 token”预测：

| 4 槽块示意 | 槽 0 | 槽 1 | 槽 2 | 槽 3 |
| --- | --- | --- | --- | --- |
| 输入 token | A，anchor | MASK | MASK | MASK |
| DFlash 的训练标签 | 不计 loss | B | C | D |
| DSpark 的训练标签 | B | C | D | E |
| DSpark Markov head 的训练前驱 | A | B | C | D |

因此不能把“两个配置都是 `block_size=K`”理解成相同的预测槽数。DSpark 还对监督有效性做 `cumprod`，遇到无效位置后，块内后续监督全部停止。

此版本 SpecForge 的 `DSparkDraftModel._sample_draft_tokens()` 辅助函数仍截取最后 `K-1` 个 hidden，配合继承的 `spec_generate()` 使用，不能把它当作 K 个训练槽的等价推理路径。

另查本文固定版本的 SGLang DSpark worker：`sample_from_anchor` 默认为 true，未覆盖提案长度时从配置 `block_size` 解析出 `gamma`。这里运行时变量 `gamma` 是提案长度，与 loss 公式中表示 `loss_decay_gamma` 的衰减参数无关；本文配方二者分别为 7 和 4。此时 `query_token_num=gamma`，保留包含 anchor 输入槽在内的全部 `gamma` 个 hidden，采样 `gamma` 个未来 token；验证输入为 `[anchor | gamma 个草稿 token]`，长度 `gamma+1`。因此默认 K=7 对应 7 个输入槽、7 个草稿 token、最多 8 个验证输入槽，与训练的 next-token 对齐相符。若显式设置 `sample_from_anchor=false` 或覆盖提案长度，必须按相应配置重新核对。

运行时证据：[长度解析及默认开关](https://github.com/sgl-project/sglang/blob/8252e3af630c103e960e13a77e570e810f8e3eb8/python/sglang/srt/speculative/dspark_components/dspark_config.py#L29-L133)、[保留所有预测行](https://github.com/sgl-project/sglang/blob/8252e3af630c103e960e13a77e570e810f8e3eb8/python/sglang/srt/speculative/dspark_components/dspark_draft.py#L648-L667)、[采样与验证块装配](https://github.com/sgl-project/sglang/blob/8252e3af630c103e960e13a77e570e810f8e3eb8/python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py#L146-L178)。这是静态槽位核对，尚非导出权重后的数值等价测试。

来源：[Markov/confidence 模块](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/dspark.py)、[DSpark 标签、loss 和真实前驱构造](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/common/dflash_family_model.py#L1890-L2273)。

### 4.4 三项训练 loss

目标最终 hidden state 经冻结目标 LM head 恢复 `p(k)`。若标签位于序列位置 `t`，教师分布来自位置 `t-1` 的最终 hidden，不能错取位置 `t`。

第一项为序列 token 的 hard-label CE：

$`\displaystyle \ell_{\mathrm{CE},k}=-\log q_k(x_k^*)`$

第二项为 **整个词表概率分布** 的 L1：

$`\displaystyle \ell_{\mathrm{L1},k}=\sum_{v=1}^{V}|p_k(v)-q_k(v)|`$

这不是对 hidden state 做 L1 回归。它直接关联标准投机采样的期望单步接受概率：

$`\displaystyle c_k^*=1-\frac12\ell_{\mathrm{L1},k}=\sum_v\min(p_k(v),q_k(v))`$

第三项为接受概率预测。启用 `confidence_head_with_markov` 时：

$`\displaystyle c_k=\sigma\!\left(w_c^{\mathsf T}[z_k;\operatorname{Embed}_{W_1}(x_{k-1})]+b_c\right)`$

$`\displaystyle \ell_{\mathrm{conf},k}=\operatorname{BCE}\!\left(c_k,\operatorname{stopgrad}(c_k^*)\right)`$

代码使用 `binary_cross_entropy_with_logits`，直接输入 confidence 的原始 logit。监督标签是 **软接受概率**，不是“这次采样是否接受”的 0/1，也不是草稿最大 softmax 概率。

默认三项权重为 `0.1 / 0.9 / 1.0`。以第一个未来 token 为 `k=1`，设有效 mask 为 `m(k)`：

$`\displaystyle w_k=\exp\!\left(-\frac{k-1}{\gamma}\right),\qquad \mathcal{L}_{\mathrm{DSpark}}=\frac{\sum_{a,k}m_{a,k}w_k\left(0.1\ell_{\mathrm{CE},a,k}+0.9\ell_{\mathrm{L1},a,k}+\ell_{\mathrm{conf},a,k}\right)}{\sum_{a,k}m_{a,k}w_k}`$

本文配方 `loss_decay_gamma=4.0`。反向传播更新融合投影、草稿骨干、Markov head、confidence head；目标模型、目标 embedding 与目标 LM head 冻结。草稿可以按配置初始化，也支持 `draft_checkpoint_path` warm start，并非必须先单独训练一个 DFlash 再启动 DSpark。

来源：[三项 loss 的实现](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/common/dflash_family_model.py#L1990-L2200)、[CE/L1 数值定义](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/core/fused_loss.py#L789-L825)、[冻结目标 embedding/head](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/target/target_utils.py#L95-L167)。

### 4.5 confidence 怎样用于推理调度

confidence 估计“前面已全部接受时，这个位置仍能接受”的条件概率。整个前缀生存概率为：

$`\displaystyle a_j=\prod_{i=1}^{j}c_i,\qquad \mathbb{E}[\mathrm{progress}(\ell)]=1+\sum_{j=1}^{\ell}a_j`$

例如条件接受率为 `0.9, 0.8, 0.5`，三个前缀的生存概率是 `0.9, 0.72, 0.36`。验证三个草稿 token 的期望推进量为 `1+0.9+0.72+0.36=2.98`，其中 1 对应每轮目标模型保证的推进。

完整 DSpark 方法会校准 confidence，并结合引擎的 verification 成本曲线选择各请求验证前缀长度；不是一律把整块送给 target。论文的调度还包含提前停止约束，避免利用未来候选信息造成选择偏差。参见 [DSpark 论文 3.2 节](https://arxiv.org/html/2607.05147v1#S3.SS2)。

SpecForge 负责训练相关 head；SGLang 才负责把输出接入实际调度。本文核对版本中的入口包括：

- [运行时 DSpark 模型](https://github.com/sgl-project/sglang/blob/8252e3af630c103e960e13a77e570e810f8e3eb8/python/sglang/srt/models/dspark.py)：基础 logits、Markov head 和 confidence。
- [草稿采样器](https://github.com/sgl-project/sglang/blob/8252e3af630c103e960e13a77e570e810f8e3eb8/python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py)：串行 Markov block 及 confidence 输出。
- [验证规划器](https://github.com/sgl-project/sglang/blob/8252e3af630c103e960e13a77e570e810f8e3eb8/python/sglang/srt/speculative/dspark_components/dspark_planner.py)：confidence 校准表、成本表和验证预算规划。

这些入口是源码定位，不代表所有硬件、worker 或服务模式都默认开启相同的 confidence scheduler。

## 5. SpecForge 的完整训练流水线

### 5.1 回答重生成与特征采集是两道工序

```text
选择 prompts / 对话历史
        ↓
用目标模型重生成 assistant 回答，或使用现成回答
        ↓
套用目标 chat template，tokenize，建立 assistant loss mask
        ↓
冻结目标模型，对这条干净 token 序列做 causal forward
        ↓
采集多层特征；EAGLE3/DSpark 还需要相应教师输出信息
        ↓
EAGLE3：展开 TTT          DFlash/DSpark：采样多个 anchor，构造 masked blocks
        ↓
草稿 forward → 计算方法对应 loss → backward → optimizer step
        ↓
保存 checkpoint → 导出 → 使用真实 serving worker 评估
```

当前 online capture adapter 的请求参数是 `temperature=0.0, max_new_tokens=0`：把已经准备好的完整 token 序列交给 SGLang 做 prefill 特征采集。**“online training”不代表每个 step 自动生成新的 assistant 回答，也不代表 RL。** 回答重生成由 `scripts/regenerate_train_data.py` 等数据流程单独完成。

来源：[online capture 请求](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/inference/adapters/server_capture.py#L380-L417)、[数据准备指南](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/docs/sections/basic_usage/data_preparation.md)。

### 5.2 online 与 offline 的准确区别

| 方式 | 目标特征何时计算 | 数据流 | 适用取舍 |
| --- | --- | --- | --- |
| online | 训练期间采集 | producer → SGLang capture server → Mooncake → trainer consumer | 避免落盘全部特征；目标推理、传输、草稿训练需要配平 |
| offline colocated | 训练前计算 | 特征文件 → trainer | 重复实验方便；需要存储与 I/O |
| offline disaggregated | 训练前计算 | 文件 → producer/feature store → trainer consumer | 将文件读取和训练解耦；增加分布式运行组件 |

在核对版本中，**online 统一采用 producer/consumer 分离，不再有旧的 online colocated target 路径**。同机 `managed-local` 也仍是分离角色；`external` 只表示服务由用户或调度器启动，不表示必须在远程机器。

统一训练入口为 `specforge train --config ...`，旧的 `scripts/train_*.py` 已移除。目标推理的并行由 SGLang server 管理，trainer 的 DP/离线 EAGLE3 USP 由训练配置管理。

来源：[训练入口和拓扑](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/docs/sections/basic_usage/training.md)、[分离训练](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/docs/sections/basic_usage/disaggregated_training.md)。

### 5.3 离线文件具体保存什么

| 方法 | `.ckpt` / `.ckpt.gz` 必需字段 | 哪些用于教师监督 |
| --- | --- | --- |
| EAGLE3 | `input_ids`, `loss_mask`, `aux_hidden_state`, `hidden_state` | `aux_hidden_state` 是三层拼接输入；`hidden_state` 是目标最终状态，用来恢复教师 logits |
| DFlash | `input_ids`, `loss_mask`, `hidden_states` | 标准 CE 直接用 token labels；`hidden_states` 是选定目标层的拼接输入 |
| DSpark | `input_ids`, `loss_mask`, `hidden_states`, `target_last_hidden_states` | 最后一项用于恢复目标分布，构造 L1 与 confidence 标签 |

这些字段是磁盘原始格式；进入统一训练 batch 后可能改名。例如 EAGLE3 磁盘的 `aux_hidden_state` 与最终教师状态，不应只凭名称混同。不同策略的特征目录不能直接混用，capture 使用的目标层和草稿配置也必须一致。

以 `S=4096, Ht=4096`、bfloat16、忽略 metadata 和压缩估算单条记录：

- 三层辅助特征约 96 MiB；加最终状态后的 EAGLE3 记录主体约 128 MiB。
- 五层辅助特征约 160 MiB；DSpark 再加最终状态约 192 MiB。

因此数据准备的成本不能只按“多少条 JSONL”衡量，要同时按有效 token 数、捕获层数和存储字节估算。

## 6. 怎么选择训练数据

### 6.1 首选覆盖实际服务分布的 prompts

训练草稿的直接目的，是提高它与 **指定目标模型** 在服务上下文中的一致性。选择数据时应优先匹配：

| 服务特征 | 数据应覆盖什么 |
| --- | --- |
| 通用助手 | 闲聊、指令、多轮历史、不同语言与回答格式 |
| 代码助手 | 代码生成、补全、修改、真实代码上下文与缩进格式 |
| 数学 / 推理 | 实际长度的推理过程、公式和最终答案；匹配 thinking 开关 |
| Agent / 工具调用 | tool schema、调用参数、工具返回和多轮边界 |
| 长上下文服务 | 与线上相近的 prompt 长度、回答长度和长程依赖 |

框架预置包括聊天类 `sharegpt / ultrachat / perfectblend`，以及 `opc / opencodeinstruct / magicoder-evol-instruct / gsm8k / hendrycks_math` 等。它们是可用来源，不代表任何一个都能替代真实流量。

推荐用目标模型重生成 assistant 回答，同时保持 tokenizer、chat template、system prompt、thinking 模式和采样策略与部署接近。原始回答也可以训练，但它与目标模型的输出分布可能不一致。不能只按“参考答案是否更优质”筛选，而忽略草稿要模仿哪个目标。

作为已发表配方的例子，DSpark 论文使用 Open-PerfectBlend prompts，并由各目标模型重新生成回答，实验采用 non-thinking 模式；这说明目标回答重生成的做法，不意味着该模式适合所有线上服务。参见 [论文训练数据说明](https://arxiv.org/html/2607.05147v1#S4.SS1)。

### 6.2 样本格式与监督边界

自定义对话可以使用以下 JSONL，每行一条记录：

```json
{"id":"sample-0001","conversations":[{"role":"user","content":"解释分页 KV cache"},{"role":"assistant","content":"目标模型生成的回答"}]}
```

现成文本也支持 `{"id":"...","text":"已经套好模板的完整文本"}`，需配置 `data.is_preformatted: true`，并指定正确 `data.chat_template`，用于识别 assistant span 和构建 loss mask。

处理时应检查：

1. 先按原始 prompt/对话分 train/eval，再重生成；近重复 prompt、同一长对话的不同切片不要跨集合。`prepare_data.py --split-eval` 提供 seed=42 的 95/5 随机拆分，但不能替代语义去重。
2. 统计 token 数、assistant 监督 token 数、长短样本分布和各领域占比；不要只看样本条数。
3. 过滤空回答、生成失败、模板损坏、截断后没有监督位置的样本。DFlash/DSpark 还要求至少两个连续受监督 token 才能构造合法 anchor。
4. `max_length` 截断后重新检查监督覆盖；长推理不能只有 prompt 被保留，回答与思考过程却都被截掉。
5. 保留部署实际会生成的 thinking、特殊结束符和工具格式；不要把隐藏推理、可见历史和当前待生成回答混在同一监督边界里。

多轮 reasoning 若只训练最后一轮，可用 `expand_reasoning_conversations.py` 拆成“每个 assistant 生成事件一条样本”，当前轮保留 reasoning，历史轮保留服务时可见内容，并配合 `data.train_only_last_turn: true`。

EAGLE3 缩减词表还需要检查对线上领域的词频覆盖；领域变化较大时，旧词表映射也可能不再合适。DFlash/DSpark 用完整目标输出 head，没有这项相同的缩减词表约束。

上述去重、分层评估和分布匹配是数据选择建议；框架并不会自动完成全部这些工作。格式、预置数据和 reasoning 转换的现有支持见 [数据准备源码指南](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/docs/sections/basic_usage/data_preparation.md)。

## 7. 具体配方、启动条件和评估

### 7.1 Qwen3-8B 示例的实际设置

| 配方字段 | EAGLE3 | DFlash | DSpark |
| --- | --- | --- | --- |
| 草稿层数 | 1 | 5 | 5 |
| TTT / block | `ttt_length=7` | `block_size=16`，15 个未来监督槽 | `block_size=7`，7 个未来监督槽 |
| 数据路径引用 | `sharegpt_train.jsonl` | `sharegpt_train.jsonl` | `perfectblend_raw_900k.jsonl` |
| `max_length` | 4096 | 3072 | 4096 |
| `learning_rate` | 1e-4 | 6e-4 | 6e-4 |
| `num_epochs` | 10 | 6 | 10 |
| `max_steps` | 10000 | 10000 | 配方未设置 |
| anchor 数上限 | 不使用这种 anchor block 采样 | 512 | 512 |
| `loss_decay_gamma` | TTT loss 默认逐步乘 0.8 | 7.0 | 4.0 |
| 默认主要 loss | 目标软分布 CE | 位置加权 hard-label CE | 0.1 CE + 0.9 分布 L1 + 1.0 confidence BCE |

数据文件名不等于已核实的数据内容或实际样本数。这三份示例的数据、最大长度、层数和训练预算不同，不能拿它们直接宣称算法公平比较或某种方法必然更快。`num_epochs` 也不是唯一停止条件，设置了 `max_steps` 时还受优化器步数预算限制。

对应固定版本配置：[EAGLE3](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/examples/configs/online/disaggregated/external/qwen3-8b-eagle3.yaml)、[DFlash](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/examples/configs/online/disaggregated/external/qwen3-8b-dflash.yaml)、[DSpark](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/examples/configs/online/disaggregated/external/qwen3-8b-dspark.yaml)。

### 7.2 启动入口与前置条件

以下命令对应上述源码版本的 NVIDIA CUDA/PyTorch 环境，模型 ID 为 `Qwen/Qwen3-8B`，所有路径相对 SpecForge 仓库根目录。先按该版本安装指南准备环境、模型与数据；online external 模式还需要启动与 SpecForge capture 协议匹配的 SGLang 服务和 Mooncake，配置好 server URLs 与 feature capture 层。对受限模型，按模型托管方要求在本地配置 `HF_TOKEN`，不要写入 YAML 或提交文件。

```bash
# 分别选择一种算法执行；不是依次训练三个阶段。
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-eagle3.yaml
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-dflash.yaml
specforge train --config examples/configs/online/disaggregated/external/qwen3-8b-dspark.yaml
```

这不是“三条命令启动相同资源实验”：EAGLE3 示例 trainer 为 1 rank，DFlash 为 8 ranks，DSpark 为 1 rank、梯度累计 512 次。目标服务使用的 TP/DP 另行由 capture server 配置；三者实际显存和并行资源需求不同。`external` 示例不会代替你启动外部服务。

离线流程则先执行 `scripts/prepare_hidden_states.py`，传入匹配的 `--strategy`、`--target-model-path`、`--draft-model-config`、`--chat-template` 和输出目录，再执行对应 `examples/configs/offline/colocated/` 配方。不要用为另一组目标层准备的特征训练当前草稿。

依赖与启动的权威入口：[安装指南](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/docs/sections/get_started/installation.md)、[训练拓扑与并行约束](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/docs/sections/basic_usage/training.md)、[导出和服务说明](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/examples/configs/README.md)。本文没有把这些命令作为已执行成功的训练记录。

### 7.3 训练结束看什么

| 层次 | 指标 | 能回答的问题 |
| --- | --- | --- |
| 训练诊断 | 按草稿深度/块位置的 loss、accuracy、分布一致性 | 后续位置是否退化，监督是否有效 |
| 采样行为 | 真实目标验证的平均连续接受长度，明确是否含 bonus | 草稿究竟能让每轮推进多少 token |
| DSpark 校准 | 条件接受率与前缀生存概率的校准误差 | confidence 能否用于定量预算规划 |
| 系统性能 | drafting/verification 时间、TPOT、吞吐、不同并发和上下文长度 | 接受率改进是否抵消了草稿与验证成本 |

一般的速度取舍可表示为：

$`\displaystyle T_{\mathrm{per\ token}}\approx\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}+T_{\mathrm{overhead}}}{\mathbb{E}[\mathrm{progress\ per\ round}]}`$

token accuracy 不能替代连续接受长度；teacher-forced 接受率估计也不能替代实际自由运行评估。对比时应固定 target 权重/量化、数据、采样参数、硬件、并发、上下文长度与服务实现，再说明草稿预算是否一致。

## 8. 仓库其他方法放在哪里理解

| 方法 | 相对上述主线的变化 | 当前源码入口 |
| --- | --- | --- |
| P-EAGLE | 对 anchor/depth 进行 Chain-of-Draft 采样，使用真实/learned mask 特征和专用 attention mask 并行学习多深度预测 | [训练模型](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/algorithms/peagle/model.py) |
| DFlash2 | DFlash 层增加分组动态卷积，另用低秩 transition selector 对 top-k 候选路径打分；仍用 `strategy: dflash` | [骨干与候选 selector](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/dflash2.py) |
| Domino | DFlash 骨干基础 logits 加上 GRU 前缀状态与当前 hidden 产生的修正 logits | [GRU 修正模块](https://github.com/sgl-project/SpecForge/blob/e3ab28d681df457f2a04e59dfac8117c2c4cbfef/specforge/modeling/draft/domino.py) |

这些方法共享部分数据、capture、优化器和运行时设施，但模型结构、标签对齐和 loss 不能只按统一入口名推断。本文详细核对的主线是 EAGLE3、DFlash 和默认 Markov 版本 DSpark。

## 9. 原始论文与证据边界

- [EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840)：多层特征融合、移除特征回归约束和 TTT 的方法来源。
- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)：KV injection、随机 anchor block 和单步并行预测的方法来源。
- [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147)：半自回归 head、confidence 监督与硬件感知验证调度的方法来源。
- [D-PACE](https://arxiv.org/abs/2605.18810)、[LK Losses](https://arxiv.org/abs/2602.23881)：可选训练目标来源。

本文涉及“代码当前做什么”的结论，以固定 commit 的实现为依据；涉及完整方法的调度设计，以论文和单独列出的 SGLang 运行时入口为依据。未将论文加速比、不同配方的结果或代码存在本身写成当前机器上的运行证明。
