# Xiaomi-Robotics-0 论文解读：视觉语言能力保留与实时异步执行

论文原题：Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution

- 作者：Rui Cai 等，Xiaomi Robotics
- 发表：arXiv，2026
- 论文：[arXiv:2602.12684](https://arxiv.org/abs/2602.12684)
- 代码：[XiaomiRobotics/Xiaomi-Robotics-0](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)
- 项目：[Xiaomi-Robotics-0](https://xiaomi-robotics-0.github.io/)

这篇笔记关注两个问题：VLA 学会动作之后，怎样保留原有视觉语言能力；模型推理比控制周期更慢时，怎样让机器人持续运动并及时响应观测。正文依次梳理两阶段预训练、动作前缀与 Lambda 注意力、数据和训练配置，以及仿真与真机实验。

:::{note}
本文依据随阅读笔记保存的论文正文、实验表格和原图整理。文中分别标注论文陈述、个人理解与待验证问题；“未披露”指所核对论文的描述范围，不代表开源代码中没有相关实现。实验排名均指论文报告的对比范围。
:::

::::{important}
**一句话核心观点**：
大参数量 VLA 模型推理延迟导致连续动作 chunk 之间出现不连贯/抖动的 jerky motion，而训练时 condition on 先前 action prefix（Training RTC）虽然改善连续性，却可能导致模型走 shortcut（直接复制 prefix 而不是关注视觉和语言输入），策略反应性变差。Xiaomi-Robotics-0 将 Lambda-shape 注意力掩码、RoPE offset 和损失重加权结合起来，限制后期 action token 对 prefix 的直接访问，改善连续性与反应性的平衡。
::::

::::{tip}
**一句话工作思路**：
当前图像与语言指令 → VLM 编码 → VLM KV cache、本体状态与 action prefix 共同约束 DiT → 5 步 flow matching 生成 action chunk → 机器人以 30 Hz 执行动作，下一轮推理在后台进行
::::

## 一、论文为什么存在？

### 1.1 研究背景

**论文背景**

VLA 模型（如 RT-2、OpenVLA、$\pi_0$、$\pi_{0.5}$）已成为机器人策略学习的新范式。它们建立在预训练 VLM 之上，将观测和语言指令直接映射为动作。然而这些模型的参数量通常在数十亿级别，推理延迟成为部署瓶颈——在同步执行中机器人必须等待推理完成才能执行下一步，导致停顿和动作不连续。

- 当前主流范式：VLM backbone + action head（离散 token 预测或连续 flow-matching/diffusion）
- 现有方案的局限：同步执行需要等待；异步 chunk 衔接还需处理连续性与反应性的冲突
- 大规模 VLA 阶段更严重的原因：参数量增加，单次推理延迟 > 实际控制周期
- 问题类型：推理部署瓶颈（体现为 latency-reactivity-stability 三角）

（来源：论文 §1 Introduction 与 Related Work）

---

### 1.2 现有方法的问题

1. **问题一：同步执行导致暂停和 discontinuous actions**
   - 论文观察：大参数量 VLA 模型推理延迟在 real-robot rollout 中不可忽略，同步执行时机器人 idle 等待推理完成，导致停顿。
   - 我的理解：完整 VLA 模型的推理延迟（本文部署为 80 ms）超过 30 Hz 控制周期，同步执行会增加等待时间。
   - 后果：jumpy/jerky motion，动作执行进入 out-of-distribution 状态。
   （来源：论文 Introduction）

2. **问题二：Training RTC 改善连续性，但可能引入 shortcut learning**
   - 论文观察：用已承诺执行的动作作为 prefix condition 到 DiT 的 noisy action token 之前，虽然改善了跨 chunk 的连续性，但让后期 timestep 的动作预测可以利用"连续动作通常相似"的时间相关性走捷径——直接复制 prefix 而非关注视觉和语言信号。
   - 我的理解：causal attention 下，任意后续 token 可以 attend 到 prefix token，prefix token 提供了高相关性的"trivial answer"，模型可能过度依赖这个信号，而忽视其他条件。
   - 后果：策略反应性下降，真机部署时 Training RTC 变体会陷入重复性动作循环（如反复 flinging towel 而不重新抓取）。
   （来源：论文 Model & Training，§Post-training；论文 Experiments，§Results）

::::{important}
**矛盾总结**：
VLA 大模型的高延迟使同步执行出现等待；不连续的动作还可能把机器人带到训练分布之外。异步执行通过 action prefix 改善连续性，但 causal attention 也让 prefix 成为潜在的信息捷径，削弱策略对视觉和语言的响应。作者选择从 attention masking 角度突破：如何在保持 prefix 连续性收益的同时，限制后期动作 token 对 prefix 的直接访问，迫使模型回到视觉-语言条件。
::::

---

### 1.3 作者的核心观察 / 假设

> **原文锚点**：
> "While this conditioning method ensures continuity across consecutively generated chunks, it allows the generation of later-timestep actions to exploit the temporal correlation that successive actions tend to be similar."（论文 Introduction）

**中文解释**：

Training RTC 的 action prefix 帮助新旧 action chunk 平滑过渡，但也提供了一个潜在的信息捷径：因为 causal attention 让所有后续 token 都能直接看到 prefix token，而 prefix token 本身就是与"下一步"高度相似的 action，模型可能过度依赖拷贝，而减少对其他模态的利用。

**我的理解：为什么可能形成捷径？**

可以借用 autoregressive 生成任务来理解：每一步的前序 token 给出了"几乎正确的答案"。如果因果注意力不设限，优化可能倾向于利用高度相关的 prefix。这里是对 shortcut 的直觉解释；论文没有测量视觉语言特征的噪声方差，也没有据此证明梯度路径。

**待验证问题：哪些条件可能减弱这种捷径？**

- 当 action 在高频任务中变化剧烈（如碰撞后反弹、快速动态抓取），prefix 与真实所需 action 差异大，即使 causal attention 下模型也必须依赖视觉信号。
- 当 prefix 长度 $\Delta t_c$ 很小（如 ≤2），prefix 提供的捷径可能变弱，反应性下降可能不明显。

---

## 二、论文到底提出了什么？

### 2.1 方法总览

::::{important}
**整体流程**：
图像与语言 → Qwen3-VL-4B-Instruct → 最后 16 层 KV cache；KV cache + 本体状态 + noisy action（异步时加入 clean prefix）→ 16 层 DiT → action chunk → 30 Hz 真机控制
::::

XR-0 采用 Mixture-of-Transformers (MoT) 架构，分两阶段 pre-training：（一）先训练 VLM 在机器人数据上通过 Choice Policies 预测 action 同时 co-train 在 VL 数据上保持视觉语言能力；（二）冻结 VLM，以 VLM 的 KV cache 为条件训练 DiT 通过 flow-matching 生成 action chunk。post-training 阶段引入 Lambda attention mask + RoPE offset + loss reweighting 来训练异步执行能力。



```{figure} images/Xiaomi-Robotics-0/xiaomi-training.png
:alt: VLM 共训练、冻结 VLM 训练 DiT、加入动作前缀的三个阶段
:width: 100%
:align: center

**原论文图 3：模型与训练流程**。先训练 VLM 的动作预测与视觉语言能力，再冻结 VLM 训练 DiT；异步后训练在 noisy action 前加入 clean prefix。
```

#### 输入与输出

| 项目 | 内容 | 形状 / 频率 / 坐标系 | 备注 |
|---|---|---|---|
| 输入 1 | 观测图像 $o_t$ | 真机为 2 个腕部相机 + 1 个外部相机；部署输入按 30 Hz 时间线聚合 | 分辨率未明确 |
| 输入 2 | 语言指令 $l$ | token 序列 | 任务级指令 |
| 输入 3 | 本体状态 $s_t$ | 维度未明确；真机部署按 30 Hz 聚合 | 通过 MLP 编码 |
| 输出 | Action chunk $a_{t:t+T}$ | 真机 $T=30$（1 秒）；LIBERO/CALVIN 为 10；SimplerEnv 为 4 | 30 Hz 指真机设置；动作编码维度未明确 |
| 内部接口 | VLM 最后 16 层 KV cache | 作为 DiT 条件 | 不等同于机器人的动作输出 |

::::{caution}
以下关键信息在本文所核对的论文正文中未明确披露（不代表代码中也不存在）：
**动作表示（绝对/相对/末端位姿或关节角）、动作维度总数、相机分辨率、本体状态的维度与物理量。**
::::

---

### 2.2 模块一：Co-trained VLM (Pre-training Step 1)

#### 作用

让 VLM 同时具备 action generation 能力（通过 Choice Policies 多模态动作预测）并保持视觉语言知识（通过 VL co-training 防止灾难性遗忘）。

#### 输入与输出

- 输入：$o_t$（图像）、$l$（语言）、$s_t$（本体状态）
- 输出：N 个 action chunk candidates + N 个 scores
- 是否可训练：是（VLM 全量更新）
- 是否冻结：否
- 与前后模块的接口：输出经过训练后，VLM 的参数承载了从视觉语言到动作的映射能力；Step 2 中 VLM 作为 frozen conditioner

#### 具体过程

1. VLM 接收 $o_t,l$ 作为标准输入，$s_t$ 通过 MLP 编码后注入
2. 追加 T 个 learnable action token `[A_i]` + 1 个 score token `[S]`
3. 每个 `[A_i]` 的输出映射为 N 个 action timestep i 的预测值
4. `[S]` 输出映射为 N 个 scores
5. 训练: score 用各 candidate 与 ground truth 的 $L_1$ 距离作为监督目标; action 用 winner-takes-all（仅 $L_1$ 最小的 candidate 通过 BP 更新）
6. 同时 co-train VL 数据（visual grounding, VQA, captioning, embodied reasoning）用 next-token-prediction 目标，采样比例 VL:robot = 1:6

#### 为什么这样设计？

**论文中的解释**：

Choice Policies 处理轨迹的 multi-modality（不同示教者可能给出不同路径到达同一目标），winner-takes-all 机制用最接近当前示教的 candidate 接收动作监督；它不保证全局最优路径。Co-training VL 数据防止 VLM 在 action 训练中丢失预训练视觉语义知识。

**我的理解**：

相当于 VLM 同时做两个任务——理解世界（VL data）和干活（robot trajectory data）。Choice Policies 同时输出动作候选，以及对候选与示教之间距离的估计。score 不是任务成功概率，论文也没有用它为后续 DiT 筛选高质量样本。1:6 的比例意味着每 7 个训练 sample 中仅有 1 个 VL sample，机器人数据主导但 VL 数据充当正则化防遗忘。

**待验证问题**：

- VLM 训练时的 action chunk 长度 T 是否与 DiT 训练的 T 一致？
- Choice Policies 中 N 的取值未说明——如果 N 太小，multi-modal 覆盖不完整；太大则训练开销显著增加
- Winner-takes-all 是否会让部分候选长期收不到动作梯度，影响多模态覆盖？论文没有报告候选利用率。

（来源：论文 Model & Training，§Pre-training）

---

### 2.3 模块二：DiT with Flow-Matching (Pre-training Step 2)

#### 作用

冻结 VLM 后，单独训练 DiT 通过 flow-matching 生成高精度的连续 action chunk，利用 VLM 的视觉语言 KV cache 作为条件。

#### 输入与输出

- 输入：VLM KV cache（最后 16 层）, $s_t$（MLP 编码）, noisy action $\tilde{a}^{\tau}$
- 输出：去噪后的 action chunk $a_{t:t+T}$
- 是否可训练：是（仅 DiT + MLPs）
- 是否冻结：VLM 完全冻结
- 与前后模块的接口：DiT 通过 attention 读取冻结 VLM 的 KV cache；此阶段 VLM 只接收图像和语言，不再追加 Step 1 的 action/score token

#### 具体过程

1. 用 VLM 前向得到所有层的 KV cache，仅取最后 16 层
2. 初始噪声：$\tilde{a}^{\tau=0} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$。噪声调度：$\tilde{a}^{\tau} = \tau \cdot a_{\text{true}} + (1-\tau) \cdot \varepsilon$（$\varepsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$）
3. DiT 预测向量场 $\mathbf{v}_{\theta}$，loss 为 $\lVert \mathbf{v}_{\theta} - (a_{\text{true}} - \varepsilon) \rVert_{2}^{2}$
4. $\tau$ 从 Beta 分布采样，偏向更高噪声区域
5. 推理时：初始化为 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 高斯噪声，经 5 步 flow-matching 从 $\tau=0$ 积分到 1
6. DiT 使用 causal attention，输入序列：$\texttt{[SINK]}, s_t, \tilde{a}_t, \dots, \tilde{a}_{t+T-1}$
7. adaLN 将 $\tau$ 注入 DiT

#### 为什么这样设计？

**论文中的解释**：

冻结 VLM 防止 action generation 训练的梯度反向传播到 VLM 导致 VL 能力遗忘。仅取 16 层 KV cache 为 DiT 设计 16 层以降低推理延迟。

**我的理解**：

相当于把 VLM 当作"冻结的特征提取器"——VLM 看到了当前场景+指令，生成了一系列表示。DiT 的任务是"在这些表示和本体状态的条件下，把随机噪声变形为正确的动作轨迹"。本文使用 5 步积分控制推理成本，但没有提供足以断言 flow matching 普遍比 diffusion 更高效的对照。Beta 采样偏向高噪声区，意味着这些状态在训练目标中得到更多关注。

**待验证问题**：

- 为什么选择 16 层作为 KV cache 传递？是否更多的层能提供更好的条件信息？
- 5 步去噪对于 30 步 action chunk 是否充分？是否有精度损失？
- Causal attention 在 DiT 中仅考虑 action 的时间顺序（temporal causality），不涉及 VLM 侧的因果性

（来源：论文 Model & Training，§Pre-training，Eq. 1）

---

### 2.4 模块三：Post-training for Asynchronous Execution


```{figure} images/Xiaomi-Robotics-0/xiaomi-lambda-attention.png
:alt: Lambda 注意力掩码与 clean prefix、noisy action 的 RoPE 位置索引
:width: 100%
:align: center

**原论文图 4：Lambda 注意力掩码**。动作 token 只直接读取局部历史动作窗口，同时保留 VLM、sink 和 state 条件。图中 noisy token 的 RoPE 位置索引增加 10。
```


#### 作用

让模型在异步执行时既能维持连续 chunk 之间的平滑过渡，又不丧失反应性。

#### 输入与输出

- 输入：当前 chunk 中已承诺执行、用于覆盖下一次推理窗口的 clean action prefix → 接到 noisy action 前
- 新增机制：Lambda attention mask + RoPE position offset + loss reweighting
- 输出：异步推理就绪的 action chunk 生成能力

#### 具体过程

1. 将 $\Delta t_c$ 个已承诺执行的 clean action 作为 prefix 加到 noisy action token 前面
   - DiT 输入序列变为：$\texttt{[SINK]}, s_t, a_t, \dots, a_{t+\Delta t_c - 1}, \tilde{a}_{t+\Delta t_c}^{\tau}, \dots, \tilde{a}_{t+T-1}^{\tau}$
2. 对于 noisy action token 的位置编码（RoPE），在原始 position index 上加一个 offset（如 +10），使模型能区分 clean prefix 和 noisy token
3. 使用 Lambda-shape attention mask（限制动作历史访问范围的 causal mask）：
   - 早期 noisy action token（紧靠 prefix 的 timestep）可以 attend 到 prefix → 帮助连续过渡
   - 后期 noisy action token **不能直接** attend 到 prefix → 减少直接复制的捷径，促使模型利用视觉语言等条件
   - 每个 noisy token 可读取 VLM KV cache、`[SINK]`、$s_t$，以及时间上之前 $w$ 步的 action token；这是滑动窗口，靠近边界时可以包含 clean prefix
4. 训练时动态 loss reweighting：当 $\Delta t_c > 0$，基于 online-predicted action 与 ground truth 的 $L_1$ 误差做 reweight，误差大的样本权重更大
5. $\Delta t_c$ 从 $\{0, 1, \dots, 6\}$ 采样

#### 为什么这样设计？

**论文中的解释**：

Lambda mask 在早期 timestep 保留与 prefix 的直接联系以帮助平滑过渡，在后期限制直接访问以减轻 shortcut learning。RoPE offset 让模型在 positional embedding 层面区分"已承诺的 clean 动作"与"待预测的 noisy 动作"。Loss reweighting 让训练更关注动作预测偏差大的样本；大误差不一定全部来自 prefix 复制。具体权重函数未披露。

**我的理解**：

Lambda mask 把直接读取 prefix 的范围限制在局部过渡区，让更远的动作更多利用观测条件。但多层 attention 仍可能间接传递 prefix 信息，因此不能说后期动作完全从零规划。RoPE offset 进一步区分 clean 与 noisy token；损失重加权属于监督学习目标的调整，不是强化学习。

**待验证问题**：

- Lambda mask 的 w 取值未明确。论文说"previous w timesteps"，但具体 w=?
- RoPE offset 的具体数值（论文说 +10）是否对不同 T 敏感？
- Loss reweighting 是否可能过拟合到某些"特别困难的样本"，导致一般性性能下降？
- 没有单独的 ablation 证明 Lambda mask 独立于 RoPE offset 和 loss reweighting 的贡献

（来源：论文 Model & Training，§Post-training，Fig.4）


## 三、数学建模与公式理解

### 3.1 Flow-matching：从噪声到动作

用 $\varepsilon\sim\mathcal{N}(\mathbf{0},\mathbf{I})$ 表示高斯噪声，真实动作 chunk 为 $\mathbf{a}_{t:t+T}$。训练的线性插值路径为：

```{math}
\tilde{\mathbf{a}}^{\tau}_{t:t+T}
=\tau\mathbf{a}_{t:t+T}+(1-\tau)\varepsilon,
\qquad \tau\in[0,0.999].
```

因此，$\tau=0$ 对应纯噪声；沿条件路径走到 $\tau=1$ 才是目标动作。该路径的目标速度为 $\mathbf{u}=\mathbf{a}_{t:t+T}-\varepsilon$。DiT 预测速度场，最小化：

```{math}
\begin{aligned}
\mathcal{L}_{\mathrm{FM}}(\theta)
&=\left\|\mathbf{v}_{\theta}
\left(\mathbf{o}_t,l,\mathbf{s}_t,
\tilde{\mathbf{a}}^{\tau}_{t:t+T},\tau\right)
-\mathbf{u}\right\|_2^2,\\
\mathbf{u}&=\mathbf{a}_{t:t+T}-\varepsilon.
\end{aligned}
```

**从公式到算法**：

1. 从轨迹中采样真实 action chunk，再采样噪声和 $\tau$。
2. 线性混合动作与噪声，得到 noisy action。Beta 分布让训练更多覆盖高噪声、较小 $\tau$ 的区域。
3. DiT 在 VLM KV cache、本体状态和 $\tau$ 的条件下预测速度场。
4. 计算预测与目标速度的平方误差。预训练 Step 2 中 VLM 冻结，更新 DiT 和相关 MLP。
5. 推理时从标准高斯噪声开始，用 5 步 flow matching 从 $\tau=0$ 积分到 1。

**我的理解**：把一段搬运 Lego 的动作看成动作空间中的轨迹，模型学习的是怎样把带噪轨迹向示教分布移动。这里的向量场定义在动作表示空间中，不等于真实物理空间里一条保证最优的运动路径。

:::{note}
固定观测条件、模型、初始噪声和确定性积分器时，输出也确定；改变初始噪声仍可产生不同动作。不能由此推出 flow matching 缺乏随机性，也不能把它作为与所有 diffusion 方法的分界。
:::

来源：论文 Model & Training / Pre-training，公式 1。

### 3.2 Choice Policies：候选评分与 winner-takes-all

VLM 同时预测 $N$ 个 action chunk 和 $N$ 个 scores。对第 $k$ 个候选，评分的监督目标是它与真实动作的距离：

```{math}
d^{(k)}=\left\|\hat{\mathbf{a}}^{(k)}_{t:t+T}
-\mathbf{a}^{\mathrm{true}}_{t:t+T}\right\|_1,
\qquad k^*=\arg\min_k d^{(k)}.
```

- 各候选的 $d^{(k)}$ 用来监督 score prediction。
- 动作分支只对 $k^*$ 对应的候选做反向传播；winner 由**实际 L1 距离**确定，不是由模型预测的 score 确定。
- score 学习估计与示教的距离，不是任务成功概率。只有预测准确时，较低 score 才意味着候选更接近示教。
- 论文没有明确 score regression 的具体损失形式，也没有给出 action/score loss 的系数，不能擅自写成固定 MSE 或等权求和。

**直观例子**：抓取同一块 Lego 可以有不同接近路径。多候选允许模型保留多个可能模式；当前示教主要监督最接近它的那个候选。它并不意味着该路径在时间、能耗或任务成功率上全局最优。

**待验证问题**：若部分候选长期不是 winner，它们是否会缺少动作梯度？判断多模态覆盖还需要候选使用频率、多样性和任务成功率，本文没有单独报告这些指标。

来源：论文 Model & Training / Pre-training Step 1。

## 四、数据：从原始数据到训练样本

### 4.1 数据来源

| 数据来源 | 规模 | 主要内容 | 需要注意的边界 |
|---|---|---|---|
| DROID、MolmoAct | 未分别给出本文使用量 | 公开机器人轨迹 | 各数据源采样权重未详细列出 |
| 自采 Lego 数据 | 338 小时 | 遥操作双臂拆解与按颜色分拣 | 双 6-DoF 机械臂 |
| 自采 Towel 数据 | 400 小时 | 遥操作双臂毛巾折叠 | 可变形物体操作 |
| 通用视觉语言数据 | 包含在 VL 总量内 | grounding、VQA、captioning 等 | 部分重新标注或交叉验证 |
| 机器人场景派生 VL 数据 | 包含在 VL 总量内 | 具身问答、推理、任务规划与点轨迹预测 | 用于加强对机器人视角的理解 |

总量约为 **200M robot timesteps** 和 **超过 80M VL samples**。Step 1 的 **VL:robot = 1:6** 是训练采样比，适用于整体 VL 数据与机器人数据，不是通用 VL 数据的独立占比，也不是数据集原始规模之比。



```{figure} images/Xiaomi-Robotics-0/xiaomi-data.png
:alt: 机器人轨迹、视觉定位、问答、描述和具身推理的数据组成
:width: 100%
:align: center

**原论文图 2：训练数据组成**。机器人轨迹与多种视觉语言任务共同参与训练；VL 数据既包括通用场景，也包括机器人相关的理解与推理。
```



**待验证问题**：不同平台的动作空间、维度、单位和坐标系怎样统一？各来源的混合权重是什么？这些工程细节不能只由总数据量推出来，论文正文没有完整展开。

### 4.2 数据处理与样本组织

论文给出的处理可以分为三部分：

1. **VL 标注与质量处理**：视觉 grounding 使用 Grounded SAM、Grounding DINO 1.5、LLMDet 的跨模型共识；VQA 和 caption 数据使用基础 VLM 重新标注。机器人轨迹还派生出具身推理和规划等 VL 任务。
2. **训练采样**：Step 1 混合 VL 和 robot 数据，VL 使用 next-token prediction，robot 使用动作与候选评分监督。论文没有为全部数据集列出统一的轨迹过滤、padding 和缺失模态策略。
3. **部署时对齐**：各传感器数据按 timestamp 聚合到统一的 30 Hz 时间线；每个时刻选择时间最近的测量值。作者说这是为了与训练分布保持一致，但没有逐项列出各训练数据源的重采样实现。

**我的理解**：最近邻时间采样不等于插值。若相机频率不足 30 Hz，可能重复使用同一帧或增加观测陈旧程度；仍需检查跨传感器时间差，不能把“统一时间线”直接理解为完全同时采集。

下面仅表示机器人监督样本的概念结构，**不是官方数据格式或可直接运行的数据加载器**：

```python
sample = {
    "observation": images_t,
    "language_instruction": instruction,
    "proprioception": state_t,
    "action": actions[t:t + chunk_length],
}
```

Step 1 的 $N$ 个 candidates 和 scores 是模型输出，不是额外的 ground-truth 字段。时间戳、标定、坐标系与 mask 如何存储，应以真实数据格式和实现为准。

### 4.3 数据质量与实验边界

| 问题 | 论文描述的处理 | 可以支持的判断 |
|---|---|---|
| grounding 标注噪声 | 多模型共识 | 用交叉验证提高标注可信度，仍可能保留共同偏差 |
| VL 文本标注质量 | 使用基础 VLM 重新标注 | 质量仍受标注模型能力限制 |
| LIBERO 失败示教 | 使用过滤后的专家示教，去掉失败轨迹 | 此设置不能推广为所有数据源都按同样规则过滤 |
| 长尾、冗余与次优轨迹 | 没有详细的统一治理说明 | 无法评估稀有失败和纠错行为的覆盖程度 |

**我的理解**：只按成功与否筛选，可能丢失“接近成功但需要纠错”的片段；它们对恢复行为或许有价值。不过这是数据设计上的疑问，本文没有比较保留和删除这类样本的效果。

LIBERO、CALVIN 和 SimplerEnv 使用论文所述标准评估协议；判断 train-test leakage 仍需数据级审计，仅凭“遵循协议”不足以完成证明。

来源：论文 Data 与 Simulation Benchmarks。

## 五、模型架构与训练

### 5.1 模型组成

| 组件 | 结构与输入 | 输出或作用 | 训练方式 |
|---|---|---|---|
| VLM | Qwen3-VL-4B-Instruct；图像与语言 | 多模态表征；Step 1 还承担动作与评分预测 | 从预训练模型初始化 |
| VLM KV-cache 条件接口 | VLM 最后 16 层的 keys 和 values | 供 DiT 读取视觉语言条件；不是额外的视觉编码器 | Step 2 冻结 VLM |
| DiT | 16 层；本体状态、noisy action、KV cache | 预测 flow-matching 速度场 | Step 2 从零训练 |
| 状态和动作 MLP | 本体状态与连续动作 | 映射到对应 token 表示 | 随相应阶段训练 |
| Choice / score heads | Step 1 的 action token 与 score token | $N$ 组候选与 $N$ 个 scores | 用于 VLM 动作预训练 |

总参数量为 **4.7B**。论文没有单列 DiT、MLP 和预测头的精确参数量，不能把总量与 VLM 名义规模之差直接当成 DiT 参数量。

**表示与 attention**：

- 视觉和文本沿用 VLM 的输入接口；本文没有重新展开视觉编码器内部所有细节。
- Step 1 将本体状态和可学习 action/score token 加入 VLM 序列；Step 2 的 VLM 只接收图像和语言，本体状态进入 DiT。
- 连续 action 与本体状态通过 MLP 编码；flow 时间 $\tau$ 通过 adaLN 注入。
- DiT 序列包含 `[SINK]`、state 与 action tokens；预训练使用 causal attention，异步后训练使用 Lambda mask。
- VLM KV cache 是同一次动作生成的条件接口，不能据此断言系统缓存了多轮历史观测。

### 5.2 训练目标

**Step 1 的监督构成**：

| 分支 | 监督内容 | 论文明确的信息 | 未明确的信息 |
|---|---|---|---|
| Action prediction | ground-truth action chunk | 只更新实际 L1 距离最小的候选 | 动作损失的完整形式与系数 |
| Score prediction | 候选与真实动作的 L1 距离 | 对每个候选预测距离评分 | score loss 的具体形式与系数 |
| Vision-language | VL 文本序列 | next-token prediction；VL:robot 采样比 1:6 | 相对于动作分支的损失系数 |

:::{important}
采样比例与损失权重是不同的量。每 7 个采样样本中约 1 个 VL 样本，不代表 VL 的 cross-entropy 系数就是 $1/7$。
:::

**Step 2 与后训练**使用前文的 flow-matching 目标。异步后训练中，当 $\Delta t_c>0$ 时，根据在线预测动作与真实动作的 L1 误差动态重加权，让偏差大的样本获得更多关注。论文没有给出可直接复现的权重函数，因此不写成“权重严格正比于 L1”。

### 5.3 训练配置与阶段

| 项目 | 论文配置 |
|---|---|
| 预训练 | 40k steps，batch size 32,768；未分别列出两个子阶段的预算分配 |
| Lego 后训练 | 40k steps，batch size 2,048 |
| Towel 后训练 | 80k steps，batch size 2,048 |
| 优化器与分布式训练 | AdamW、DeepSpeed ZeRO-2 |
| 训练硬件、精度、scheduler | 本文未详细列出；ZeRO-2 本身不能确定数值精度 |
| 预训练 Step 1 | robot + VL 共训练，更新 VLM 与相关动作预测模块 |
| 预训练 Step 2 | robot 数据；冻结 VLM，训练 DiT 与相关 MLP |
| 同步后训练 | 明确解冻整个 VLM 和 DiT，在目标机器人轨迹上继续训练 |
| 异步后训练 | 加入 prefix、Lambda mask、RoPE offset 与误差重加权；冻结范围未单独明确 |
| Prefix 长度 | 训练时从 $\{0,1,\ldots,6\}$ 采样 |
| 真机 action chunk | $T=30$，对应 1 秒动作 |

**我的理解**：两阶段预训练将“让 VLM 理解动作”与“让 DiT 学会连续动作生成”分开；后训练再适配机器人和部署方式。这种划分有助于理解能力保留的设计意图，但各阶段的独立贡献仍需要对应消融来验证。

来源：论文 Model & Training、Implementation Details。

## 六、推理与机器人部署

### 6.1 异步推理闭环



```{figure} images/Xiaomi-Robotics-0/xiaomi-async-execution.png
:alt: 相邻 action chunk 在推理窗口内的执行和拼接时间线
:width: 100%
:align: center

**原论文图 5：异步执行与 chunk 衔接**。机器人在推理期间继续执行当前 chunk，下一轮生成以承诺动作作为 prefix，并按已流逝的推理时间对齐。
```



1. 执行当前 chunk 的前 $T_e$ 步。
2. 使用最新图像、本体状态和语言指令触发下一次推理；同时继续执行当前 chunk 的剩余动作。
3. 取当前 chunk 的 $[T_e,T_e+\Delta t_c)$ 段作为下一轮 clean prefix。它代表**已承诺执行**的动作，其中一些会在后台推理期间执行，而不是都已执行完毕。
4. VLM 生成 KV cache，DiT 以 prefix 和其他条件做 5 步 flow matching。
5. 推理完成后，从新 chunk 的第 $\Delta t_{\mathrm{inf}}$ 步开始执行，跳过推理期间已流逝的时间，使两个 chunk 对齐。

同步模式也先执行 $T_e$ 步，再发起推理；区别在于等待期间机器人停下。异步模式避免这段空转，同时增加了衔接和反应性的要求。

| 指标 | 数值或含义 |
|---|---|
| 模型推理延迟 | NVIDIA RTX 4090 上约 80 ms |
| 真机控制频率 | 30 Hz，约 33.3 ms 一个控制周期 |
| 真机 chunk | $T=30$，1 秒动作 |
| 新推理触发 | 同步和异步均先执行 $T_e$ 步；论文未给出其具体取值 |
| 推理窗口 | 80 ms 约为 2.4 个控制周期；按整步覆盖通常至少需要 3 步 |
| Prefix 长度 | 论文要求 $\Delta t_c\geq\Delta t_{\mathrm{inf}}$，覆盖推理窗口 |
| 积分步数 | 5 |

**我的理解**：prefix 覆盖窗口是动作连续性的一个条件，同时还需要当前 chunk 的剩余动作足以支撑这段等待。论文没有详细说明离散步取整、推理超时和延迟抖动的调度策略；不能把平均 80 ms 当成所有轮次的固定上界。

30 Hz 是动作执行频率，不代表完整模型每秒重规划 30 次。80 ms 推理让观测在动作落地时变旧，但这只是端到端延迟的一部分，不能直接等同于传感器延迟。快速动态任务对这种观测陈旧程度是否敏感，仍需专门评估。

### 6.2 状态、动作接口与适用范围

- **观测历史**：论文描述的是当前图像和状态；跨 chunk 连续性依赖 action prefix。没有明确描述多轮观测历史缓存。
- **后处理与 IK**：论文未详述；不能从 end-to-end 表述推断完全没有动作后处理，也不能擅自断定输出是关节角或笛卡尔位姿。
- **真机平台**：双 6-DoF 机械臂，两个腕部相机与一个外部相机；夹爪动作的具体编码未详述。
- **跨 embodiment**：预训练混合了不同机器人数据，但动作维度、坐标系和单位如何统一需要查实现。
- **Human-to-robot**：自采数据来自遥操作；本文没有提出从人类视频或人手姿态到机器人动作的重定向流程。

来源：论文 Data、Deployment、Real-Robot Experiments。

## 七、实验设计与结果分析

### 7.1 实验要验证什么？

| 主张 | 对应实验 | 支持范围 |
|---|---|---|
| 机器人策略性能 | LIBERO、CALVIN、SimplerEnv | 报告的主要汇总指标领先所比较方法；不代表每个子项都领先 |
| 异步后训练提高效率 | Lego、Towel；与 Sync、Training RTC、$\pi_{0.5}$ 比较 | 在这些真机设置中吞吐更高，评估规模有限 |
| Prefix 会引入 shortcut | Training RTC 的毛巾重复动作案例 | 定性证据；没有单独量化 prefix 复制率 |
| 改进后训练减轻 shortcut | 完整方法与 Training RTC 对比 | 验证的是多项技术组合，未隔离 Lambda mask 的独立贡献 |
| 保留视觉语言能力 | 10 项 VL benchmark、去掉 VL 数据的变体 | 比所比较 VLA 在 9 项上更好；与原始 VLM 仍存在能力差距 |

### 7.2 实验设置与指标

**仿真**：

- **LIBERO**：使用过滤后的专家示教，联合四个 splits 训练，遵循 OpenVLA 评估协议；$T=10$。
- **CALVIN**：ABCD→D 测试分布内性能，ABC→D 测试未见环境泛化；1000 条指令链，每链 5 个任务；$T=10$。
- **SimplerEnv**：Google Robot 使用 RT-1 Fractal 训练，分别评估 Visual Matching 和 Variant Aggregation；WidowX 使用 Bridge 训练；$T=4$。

**真机**：

- **Lego**：LA-5、LA-10、LA-20 分别有 5、10、20 块砖，每种规模 3 个装配配置、每配置 3 次试验；MA 共 34 块砖，3 次试验。
- **Towel**：6 类毛巾，每个方法进行两次连续 30 分钟 rollout；单次折叠超过 2 分钟记为失败。

| 对比方法 | 可以确认的对比范围 | 需要保留的限制 |
|---|---|---|
| $\pi_{0.5}$ 真机基线 | 按 OpenPi 官方协议微调到本文任务，使用相同训练设置 | 不代表预训练数据和 backbone 相同 |
| Ours (Sync) | 同步部署与相应后训练 | 与异步版本不仅运行调度不同，训练方式也有区别 |
| Ours (Training RTC) | 使用 action prefix 的异步基线 | 完整方法同时增加 mask、offset 和 loss reweighting |
| EO-1、OpenVLA、FLOWER 等 | 相应仿真表格中的公开比较 | 不能默认所有预训练数据、架构与动作表示一致 |
| MolmoAct、$\pi_0$、$\pi_{0.5}$ | 视觉语言能力比较 | 能力、数据与模型规模并非严格控制变量 |

| 指标 | 定义 | 解释边界 |
|---|---|---|
| 仿真 success rate | 按 benchmark 协议统计任务成功比例 | 不直接反映动作平滑性或实时性 |
| CALVIN average length | 每条链连续完成的任务数，最多 5 个 | 同时受感知、执行与泛化影响，不能只归因于规划 |
| Lego success rate | 正确分拣的砖块数 / 总砖块数 | 不是整个装配任务全部成功的 trial 比例 |
| Lego throughput | 正确分拣的砖块数 / rollout 总时间 | 同时反映成功率和耗时 |
| Towel throughput | 成功折叠的毛巾数 / rollout 总时间 | 包含失败和恢复耗时 |

### 7.3 主要结果

#### 结果一：LIBERO

| 方法 | Spatial | Object | Goal | Long | 平均 |
|---|---:|---:|---:|---:|---:|
| EO-1 | 99.7 | 99.8 | 99.2 | 94.8 | 98.2 |
| Xiaomi-Robotics-0 | 98.8 | 100.0 | 98.8 | 97.2 | 98.7 |

单位为成功率（%）。论文还报告 $\pi_{0.5}$ 平均 96.9%，OpenVLA 平均 76.5%。

**我的理解**：XR-0 在 Long 上比 EO-1 高 **2.4 个百分点**，但 Spatial 和 Goal 更低，不能写成每个 split 都领先。Long 的任务成功率说明长程任务表现更好，单靠这个指标不能证明增益来自 chunk 连续性或 flow matching。不同数据与训练设置也可能影响结果。

来源：论文 LIBERO 结果表。

#### 结果二：CALVIN

| 设置 | 方法 | 平均连续完成任务数 | 连续完成 5 个任务 |
|---|---|---:|---:|
| ABCD→D | FLOWER | 4.67 | 88.3% |
| ABCD→D | Xiaomi-Robotics-0 | 4.80 | 91.8% |
| ABC→D | FLOWER | 4.53 | 77.8% |
| ABC→D | Xiaomi-Robotics-0 | 4.75 | 88.1% |

**我的理解**：未见环境的 ABC→D 设置中，平均链长增加 0.22，连续完成 5 个任务的比例增加 **10.3 个百分点**，比只看接近上限的平均链长更直观。不过它也不是所有子项都第一：ABCD→D 的 3-task 成功率为 96.7%，略低于 FLOWER 的 96.9%。

来源：论文 CALVIN 结果表。

#### 结果三：SimplerEnv

| 平台与设置 | EO-1 | Xiaomi-Robotics-0 | 差值 |
|---|---:|---:|---:|
| Google Robot / Visual Matching | 76.5% | 85.5% | +9.0 个百分点 |
| Google Robot / Variant Aggregation | 63.0% | 74.7% | +11.7 个百分点 |
| WidowX | 72.7% | 79.2% | +6.5 个百分点 |

**我的理解**：Variant Aggregation 引入视觉变化，结果支持整个模型在该协议下的鲁棒性优势。VL co-training 可能有帮助，但没有隔离变量的机器人任务消融，不能把全部增益直接归因于它。

Drawer Apple 在 Visual Matching 下为 EO-1 52.8% → XR-0 75.0%；在 Variant Aggregation 下则是 23.8% → 66.7%。这两个设置不应混用。

来源：论文 Google Robot 与 WidowX 的 SimplerEnv 结果表。

#### 结果四：真机吞吐量与成功率



```{figure} images/Xiaomi-Robotics-0/xiaomi-real-robot-results.png
:alt: 双臂机器人 Lego 和毛巾任务设置，以及各方法的成功率和吞吐量柱状图
:width: 100%
:align: center

**原论文图 6：真机任务与结果**。上方为 Lego 拆解和毛巾折叠设置；下方展示按砖块统计的成功率及两类任务的吞吐量。
```



**论文结果**：Lego 上各方法按砖块统计的成功率接近，同步方法略好；XR-0 异步完整版吞吐量最高。Towel 上，$\pi_{0.5}$、Ours (Sync)、Ours (Training RTC) 约为 **1.0 pcs/min**，完整版为 **1.2 pcs/min**，即相对提高 **20%**。

Training RTC 有时在抓住多层毛巾后反复执行 flinging，没有重新抓取；作者用这个现象说明 prefix 可能形成捷径。完整版减轻了这类重复失败。这支持组合改造的有效性，但不能把全部吞吐增益只归因于 Lambda mask 或只归因于减少等待时间。

**待验证问题**：毛巾每个方法仅评估两次 30 分钟，MA 仅 3 次试验；正文未给出置信区间。还需要更长评估、更多任务与失败恢复统计，判断效果是否稳定。

来源：论文 Real-Robot Experiments、原图 6。

#### 结果五：视觉语言能力保留到什么程度？

| Benchmark | 原始 Qwen3-VL-4B-Instruct | MolmoAct | Xiaomi-Robotics-0 |
|---|---:|---:|---:|
| ERQA | 40.0 | 33.5 | 40.8 |
| SEED | 78.8 | 72.7 | 78.6 |
| POPE | 89.7 | 86.6 | 88.5 |
| AI2D | 81.6 | 72.0 | 78.7 |
| MMBench | 88.7 | 80.1 | 84.4 |
| MME | 87.1 | 69.5 | 81.8 |
| MMMU | 51.7 | 38.0 | 46.2 |
| TextVQA | 78.0 | 67.3 | 72.0 |
| SciQA | 92.7 | 91.1 | 79.4 |
| ChartQA | 76.8 | 57.1 | 59.2 |

表中沿用论文各 benchmark 的分数口径，不跨任务比较绝对值。XR-0 在 10 项中有 9 项领先所比较的 VLA，例外是 SciQA。相比原始 Qwen3-VL，只有 ERQA 略高，其余都下降；SciQA 和 ChartQA 分别低 13.3 和 17.6 分，不能笼统称为“完全保持”或“只有微小退化”。

移除 VL 数据的变体在论文评估中 10 项均为零，说明联合训练对保留可用的 VL 表现很重要；这些评估分数也可能受到输出格式遵循等因素影响，不能把零分直接解释为内部所有知识都消失。

来源：论文 Preservation of Vision-Language Capabilities 及 VL benchmark 结果表。

### 7.4 消融能说明什么？

| 对比 | 观察 | 能得出的结论与限制 |
|---|---|---|
| 有 / 无 VL 数据 | 去掉 VL 数据后，VL 测试得分严重退化 | 支持共训练的重要性；不能替代其对机器人成功率贡献的独立评估 |
| Sync / Async | Lego 同步成功率略好，异步完整版吞吐更高 | 体现精度与效率的取舍；后训练和调度均有变化 |
| Training RTC / 完整版 | 前者出现毛巾重复动作，后者吞吐更高 | 验证 mask、RoPE offset、重加权的组合，不是单因素消融 |

尚缺少仅改变 Lambda mask、仅改变 RoPE offset、仅改变 loss reweighting 的对照。论文也未用定量指标直接测量 prefix 复制程度或对新观测的响应延迟。

### 7.5 失败案例与局限

| 现象 | 论文解释或观察 | 仍需验证的方向 |
|---|---|---|
| Lego 异步成功率略低 | 反应性较差可能导致抓取不准、夹爪与砖块间张力较高，砖块被弹出工作区 | 缩短推理延迟、改进接触控制是否有效 |
| Training RTC 重复 flinging | 抓住多层毛巾后没有重新抓取，反复执行同一动作 | 完整版减轻了观察到的失败，但不等于彻底解决 shortcut |
| VL 复杂数值推理错误 | 附录展示密集图表推理的失败 | 改变数据配比是否能恢复能力，是否影响动作性能 |
| VL 格式遵循错误 | 如要求数字却输出单词 | 格式规范与任务理解对评估分数各有多大影响 |

:::{caution}
相对提升不等于问题已经解决。1.2 pcs/min 对应平均约 50 秒完成一条毛巾；是否满足应用需求取决于实际节拍、成功率和恢复成本。本文没有给出工业产线验证，真机任务与评估规模也有限。
:::
