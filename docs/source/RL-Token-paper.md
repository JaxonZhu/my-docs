# RL Token 论文解读：冻结 VLA 的在线强化学习

论文原题：RL Token: Bootstrapping Online RL with Vision-Language-Action Models

- 作者：Charles Xu、Jost Tobias Springenberg、Michael Equi、Ali Amin、Adnan Esmail、Sergey Levine、Liyiming Ke
- 机构：Physical Intelligence
- 发表：arXiv，2026 年 4 月 24 日首次提交；4 月 30 日更新至 v2
- 论文：[arXiv:2604.23073](https://arxiv.org/abs/2604.23073)
- 官方介绍：[Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt)（2026 年 3 月 19 日）
- 原文：[官方 PDF](https://www.pi.website/download/rlt.pdf)

RLT 将 VLA 的内部表征压缩为一个 RL token，并让轻量 actor 参考 VLA 生成的动作块，通过在线强化学习改进精密操作。它保留大模型的感知与动作先验，同时把在线更新集中到小型 actor–critic 上。

我最关注的是这套接口如何工作：什么信息进入 RL token，critic 怎样评价整段动作，reference 如何约束探索，以及关键阶段训练和人工切换为样本效率贡献了什么。

:::{note}
**阅读范围**：正文依据所提供的论文 PDF 与主笔记整理，保留方法分析、实验解读与待验证问题。论文陈述与“我的理解”分开表述；“未披露”均指所核对的论文内容，不代表当前代码或项目资料的状态。这里没有进行训练代码复核，也不将本地 PDF 直接认定为某个 arXiv 版本。
:::

:::{important}
**核心观点**：在线 RL 阶段冻结 VLA 与 token 模块，用 2048 维 RL token 表示状态，让小型 Gaussian actor 在 VLA reference action 附近学习完整的 10 步动作块。critic 指导的是 actor 的参数更新，执行时由 actor 直接产生动作，并不进行在线 Q 搜索。
:::

## 一、为什么需要这项工作？

### 1.1 通用操作能力与最后几毫米的精度

大规模 VLA 能从多任务数据中获得视觉、语言和行为先验，但其单任务表现仍受到遥操作数据质量与覆盖度的约束。在螺丝安装等精密装配中，接触结果对位置、角度和时序十分敏感，base policy 可能停顿、试探、回撤或反复重试。（来源：§I–II；图 3）

作者讨论了三类相关路线：

1. **全模型 RL**：更新整个 VLA 可以改变行为，但在线优化数十亿参数，使得短时间内充分复用每一次真机交互更困难。
2. **轻量真机 RL**：HIL-SERL、RL100 一类系统能训练小网络，却通常从常规视觉 encoder 开始，没有直接利用 VLA 已学到的任务语义和行为先验。
3. **单步残差与 latent steering**：PLD、Policy Decorator、DSRL、GR-RL 等避免全模型更新，但不同方案仍可能面对较长的信用分配跨度、手动设置残差尺度，或受限于 VLA 可生成的动作模式。（来源：§II）

**我的理解**：在线 RL 希望网络小、更新快、数据复用率高；VLA 的价值则来自大模型内部丰富的表征与动作先验。RLT 通过一个紧凑接口，把两者连接起来。

### 1.2 大模型提供先验，小模型进行局部改进

在线阶段，冻结的 VLA 提供 RL 状态表征与 reference action，引导探索从已有的可行行为附近开始；小型 actor–critic 负责改进目标任务中的困难阶段。（来源：§III；图 1）

**我的理解**：这是一种非对称适配。大模型判断当前情境大致应该做什么，小模型利用本机经验，学习在特定夹具、摩擦、相机误差和接触几何下怎样做得更快、更稳。

**待验证问题**：如果 base VLA 从未覆盖正确动作模式，reference conditioning 与 L2 锚定也可能限制策略离开错误邻域。论文主要展示已有策略“能做但慢、偶尔失败”时的改进，没有证明能普遍修复结构性错误的基础策略。

## 二、方法如何组成一个训练系统？

### 2.1 从任务适配到在线强化学习

流程分成两个阶段：先用任务示范联合进行 VLA SFT 与 RL-token 重建训练；随后冻结 VLA 和 token 模块，采集 warmup 数据，再异步执行机器人交互、经验回放和 actor–critic 更新。

```{figure} images/RL-Token/rlt-overview.png
:alt: RLT 从 VLA 提取 RL token 和参考动作，由轻量 actor–critic 改进机器人关键操作阶段
:width: 100%
:align: center

**原论文图 1：RLT 整体流程**。任务适配后冻结 VLA 与 token 模块，在线更新集中于轻量 actor–critic。
```

| 接口 | 内容 | 形状或频率 | 说明 |
| --- | --- | --- | --- |
| 视觉输入 | 两个腕部相机、一个基座相机 | 输入分辨率未披露 | 任务设置见 §VI-A |
| 指令 | 任务级语言指令 | 每个实验任务固定 | 仅在 RL-token 重建时舍弃 language embeddings；VLA 仍接收指令 |
| VLA action | $\tilde{\mathbf{a}}_{t:t+H-1}$ | $H=50$，单步 $d=14$ | 50 Hz 下覆盖 1 s |
| RL action | $\mathbf{a}_{t:t+C-1}$ | $C=10$，共 140 维 | 50 Hz 下覆盖 0.2 s |
| RL state | $x_t=(\mathbf{z}_{\mathrm{rl}},\mathbf{s}^{p}_t)$ | RL token 为 2048 维 | 另接本体状态，具体维数未披露 |
| Reward | 稀疏二值奖励 | 成功终止时 $+1$，否则 $0$ | 由操作员标记 |
| Replay sample | 动作块转移与 reference | stride 为 2 个控制步 | 在 50 Hz 数据中约形成 25 条样本/s |

本体状态的描述分散在正文与附录：§VI-A 将 screw 与关节位置、其他任务与末端位姿关联，附录 B 还提到位置与速度。论文没有给出逐任务完整的状态维度表，不宜据此写出唯一确定的输入张量。（来源：§VI-A；附录 B）

:::{note}
**动作接口的边界**：论文说明比较方法使用 14 维 delta action space，但没有完整列出增量所在坐标系、各维单位、归一化、裁剪和夹爪通道定义。动作块的维数可核对，完整的真机控制接口仍需结合实现确认。
:::

### 2.2 RL token：通过重建训练一个信息瓶颈

Transformer VLA 的最终层 token 数量多、总维度高，也没有预先指定哪个 token 最适合 value learning。RLT 通过重建目标学习一个单 token readout，将这些内部 embeddings 压缩成紧凑表示。（来源：§IV-A；图 2）

设 VLA 的最终层特征为 $\mathbf{z}_{1:M}=f(s,\ell;\theta_{\mathrm{vla}})$。系统追加 learned embedding $\mathbf{e}_{\mathrm{rl}}$，送入轻量 encoder transformer $g_\phi$，特殊位置的输出就是 $\mathbf{z}_{\mathrm{rl}}$。decoder 再以这个 bottleneck 和**原始 embedding 的前缀**为条件，因果地重建各位置的 embedding。

这里的前缀使用 stop-gradient 后的目标特征，属于 teacher forcing，并非先前已经重建出的预测值。重建目标的梯度被截断；任务适配阶段还同时进行 VLA SFT，不能把这一阶段也描述成 VLA 全程冻结。

```{figure} images/RL-Token/rlt-token.png
:alt: RL token 编码器读取 VLA 最终层特征，解码器以 RL token 和目标特征前缀重建原始 embeddings
:width: 100%
:align: center

**原论文图 2：RL token 的抽取与重建**。重建目标鼓励单个 token 保留 VLA 特征信息；在线 RL 阶段不再更新这套表征模块。
```

**我的理解**：重建训练鼓励 RL token 保留对 VLA 内部表示有用的信息。冻结表征还能减少在线 value learning 面对的输入漂移，但这不构成 critic 稳定性的保证，也不证明该 token 是充分的 RL 状态。例如，VLA 本身无法观察的接触力不会因重建目标而自动出现。

论文以 frozen ResNet-10 替换 RL token 的消融，支持 VLA 表征在该设置下有用；它没有比较简单的 VLA feature pooling、learned probe、reward-aware bottleneck 或不同 token 数量。因此，收益中有多少来自 VLA 预训练、多少来自具体的重建读出方式，仍待区分。

### 2.3 Chunk-level TD critic：缩短奖励传播的决策链

critic 直接评估整段动作：$Q_\psi(x_t,\mathbf{a}_{t:t+C-1})$。TD target 累积动作块内的奖励，再从下一个动作块起点 bootstrap。（来源：§IV-B，式 3）

在 50 Hz 下，5–20 s 的关键阶段包含约 250–1000 个控制步。取 $C=10$ 后，名义上的决策链约为 25–100 个动作块。这有助于缩短稀疏终点奖励的传播跨度，但不意味着每条轨迹恰好只需这么多次优化更新。

论文采用 target critic，附录进一步说明使用两个 Q 函数，并取二者较小值构造 target，以减轻价值高估。这些是已披露的 TD3 风格设计，其他完整 TD3 配置不能直接补入。（来源：§IV-B；附录 B）

**我的理解**：chunking 同时改变信用分配跨度、actor 输出结构和 VLA 查询频率。论文的 w/o Chunk 还因 50 Hz 查询 VLA 不可行，将 RL token 换为 ResNet-10，因此这个消融无法单独量化 chunk-level TD 的贡献。

### 2.4 Reference-conditioned actor：以参考动作作为条件

actor 接收 RL state 与 reference 的前 $C$ 步，输出固定小方差的 Gaussian 动作块分布：

$$
\pi_\theta(\mathbf{a}_{1:C}\mid x,\tilde{\mathbf{a}}_{1:C})
=\mathcal{N}\!\left(
\boldsymbol{\mu}_\theta(x,\tilde{\mathbf{a}}_{1:C}),\sigma^2\mathbf{I}
\right).
$$

reference 提供可行行为的起点，也把 VLA 多模态动作分布中这一次采样的模式传给单峰 Gaussian actor。这样，actor 无需仅凭状态 token 猜测当前应该跟随哪种动作模式。（来源：§IV-B，式 4）

**我的理解**：actor 直接输出完整 chunk，没有显式计算“reference 加残差”。它通过输入条件和训练损失形成局部编辑效果；能够偏离参考动作的程度，仍与 critic 的质量和正则权重有关。

### 2.5 BC regularizer：用 L2 软锚定策略改进

令动作块从当前 actor 中采样，actor loss 为：

$$
\mathcal{L}_{\pi}(\theta)=
\mathbb{E}\left[
-Q_\psi(x,\mathbf{a}_{1:C})
+\beta\big\lVert\mathbf{a}_{1:C}-\tilde{\mathbf{a}}_{1:C}\big\rVert_2^2
\right].
$$

第一项鼓励预测价值高的动作，第二项把输出锚定到 VLA reference。较大的 $\beta$ 加强模仿约束，较小的 $\beta$ 允许更多偏离，也更依赖 critic 在未充分覆盖动作上的估计。（来源：§IV-B，式 5）

这是一种 L2 软约束，没有严格的 KL trust-region 保证。图 7 中去掉 BC regularizer 带来的性能下降最大，支持在有限数据下约束 140 维动作块探索的重要性。

**待验证问题**：论文未给出 $\beta$ 以及动作各维的单位和归一化细节。若平移、旋转与夹爪通道尺度不同，L2 正则会隐式赋予它们不同权重，复现时需要核对这一点。

### 2.6 Reference dropout：保留独立生成动作的能力

conditioning 与 BC regularizer 都鼓励 actor 复制 reference，尤其是在 critic 尚未学会有效区分动作价值时。作者在 50% 的训练样本上将 **actor 输入中的 reference** 置零，推理时始终提供 reference。（来源：§IV-B；附录 B）

置零的是输入条件，BC 项仍以原来的参考动作为目标。若发生人工接管，这个 reference 则按照接管机制替换为人的动作；不能将 dropout 理解成要求 actor 模仿零动作。

**我的理解**：这种结构化输入 dropout 促使 actor 保持一条独立的动作生成路径，但零 reference 与真实的错误 reference 并不等价。对噪声、过期或错误 reference 的鲁棒性，仍需额外实验；论文没有单独的 w/o dropout 消融曲线。

### 2.7 人工接管如何进入 replay？

操作员可在接触复杂或需要纠正的情况下接管，用遥操作动作覆盖 actor 输出。写入 replay 时，执行动作与 reference 都替换为人的动作，使 intervention 同时成为局部行为目标。buffer 混合 VLA warmup、RL rollout 和人工纠正，episode 的成功与失败也由人标记。（来源：§V；算法 1）

**我的理解**：人同时承担纠错、奖励标注、关键阶段划分和策略切换等职责。论文报告的 robot-data efficiency 不能直接等同于 human-effort efficiency，因为它没有给出干预次数、接管时长和逐任务监督工时。

### 2.8 Critical-phase training 与自动 handoff

RLT 作为精密操作的局部策略嵌入完整任务。以 screw installation 为例，base VLA 负责抓起电动螺丝刀、移向螺丝等前序步骤；到对准螺丝头并旋入的关键阶段后，控制权由 base VLA 交给 RLT。

这里的 **handoff** 是启用专项 RL 策略；**human intervention** 则是在机器人执行过程中由人覆盖动作。论文明确描述的是 base VLA → RLT 的切换，没有建立一个通用的双向切换协议。

#### 先隔离困难阶段，集中练习精密操作

训练首先在受控的 critical-phase setting 中进行。操作员将机器人与物体复位到困难阶段即将开始的位置，每个 RL episode 只练习这一段，并在结束时标记成功或失败。这样，数据与奖励集中在 base VLA 最薄弱的操作上。

例如，zip tie fastening 从机器人已经分别抓住扎带两端的状态开始，RLT 直接学习弯曲扎带、把尾端穿过锁孔，不必每次重新完成发现、抓取和搬运。

阶段隔离还缩短了稀疏奖励的信用分配跨度。完整任务持续 30–120 s，在 50 Hz 下约有 1500–6000 个控制步；关键阶段通常为 5–20 s，约有 250–1000 步。再结合 $C=10$，局部策略面对的是约 25–100 个动作块的决策链。

**我的理解**：这里的效率既来自算法与表征，也来自人工任务分解。作者先把长程任务改造成一个可密集练习的短程问题，尚未解决任意长任务上的信用分配。

#### 再适应前序策略真实产生的到达状态

人工复位的状态可能比真实执行更规整。例如，base VLA 可能把螺丝刀抓得略偏，或让双臂以不同姿态抵达扎带操作阶段。这些状态构成了前序策略实际产生的到达状态分布。

为减小这种分布偏移，**screw 与 zip tie** 在受控关键阶段训练后，继续进入 full-task phase：机器人从 home position 出发，由 base VLA 完成前半段，再在困难阶段切换到 RLT。其他两个任务主要在关键阶段设置下评测。（来源：§V–VI）

这个 curriculum 先学习精密技能，再让它适应实际到达状态。full-task training 并不表示 RLT 接管全部任务：它优化的仍是自己控制的关键阶段，前序步骤继续由 base VLA 完成。

#### 切换边界如何获得？

训练中的 handoff 时刻由操作员判断。用 hierarchical RL 的语言理解，RLT 类似一个持续若干步的 option：有局部 policy 和终止条件，而何时进入该 option 的边界首先由人提供。

论文提出，可在在线训练结束后，利用这些切换与 intervention 标签对 VLA 再做短暂 SFT，使其预测何时交出控制权，从而自动启用 RLT。

:::{note}
**自动 handoff 的证据范围**：论文没有报告切换预测器的准确率、触发阈值、过早或过晚切换的代价，也没有独立比较人工与自动切换的最终成功率。因此，不能认定所有主实验结果都使用了自动 handoff。奖励标注、纠正和策略切换的完全自动化仍是后续方向。
:::

**我的理解**：这套设计适合存在清晰瓶颈阶段的装配任务。如果失败根源在更早的抓取或搬运阶段，或者任务没有明显可分割的关键阶段，局部 RLT 可能不足以修复整体行为。向长期自主学习扩展，还需要发现关键阶段、学习可靠的进入条件，并协调多个专项策略。

### 2.9 异步更新与高数据复用

机器人采集与 learner 异步执行。论文分别报告 UTD 为 5，以及每次 actor update 配两次 critic update；没有明确交代这两个比率的计数关系，不宜据此推算每秒梯度更新次数。（来源：§V；附录 B）

系统以 stride 2 构造重叠动作块：一秒 50 Hz 的连续数据约产生 25 条 replay samples。这个 stride 是回放样本的构造间隔，并不意味着每两个控制步调用一次 VLA。

**我的理解**：异步 learner 可以在机器人复位等间隙继续复用数据，但重叠样本也有较强相关性。双 Q、BC 锚定和冻结表征有助于约束学习过程，不能因此忽略高 UTD 下的 critic 过拟合。论文没有完整给出 replay、batch、优化器和实际更新吞吐率，无法仅凭这些比率复原训练系统。

## 三、关键公式与算法直觉

### 3.1 RL token 的读出与重建

将 VLA 最终层特征与一个 learned query 拼接，特殊位置的输出作为 RL token：

$$
\mathbf{z}_{\mathrm{rl}}=
g_\phi\!\left([\mathbf{z}_{1:M},\mathbf{e}_{\mathrm{rl}}]\right)_{M+1}.
$$

为避免混淆重建前缀，先记 $\bar{\mathbf{z}}_i=\operatorname{sg}(\mathbf{z}_i)$。decoder 与投影头 $h_\phi$ 根据 RL token 和目标前缀预测当前位置，再最小化重建误差：

$$
\begin{aligned}
\hat{\mathbf{z}}_i
&=h_\phi\!\left(d_\phi([\mathbf{z}_{\mathrm{rl}},\bar{\mathbf{z}}_{1:i-1}])\right)_i, \\
\mathcal{L}_{\mathrm{ro}}
&=\mathbb{E}_{\mathcal{D}}\!\left[
\sum_{i=1}^{M}\big\lVert\hat{\mathbf{z}}_i-\bar{\mathbf{z}}_i\big\rVert_2^2
\right].
\end{aligned}
$$

这里 $M$ 为参与重建的 token 数，$\mathbf{e}_{\mathrm{rl}}$ 为 learned query，$\mathbf{z}_{\mathrm{rl}}\in\mathbb{R}^{2048}$，$\operatorname{sg}$ 表示 stop-gradient。（来源：§IV-A，式 1–2）

**待验证问题**：如果 decoder 仅靠 teacher-forced 前缀就能预测大量后续特征，单 token 的信息承载可能弱于直觉预期。论文没有给出逐 token 重建误差或对前缀依赖程度的分析；一个 token 确实控制了下游 actor–critic 的输入成本，但其充分性仍需实验判断。

### 3.2 Chunk-level TD target

论文式 3 将一个动作块内的折扣奖励与下一状态的价值连接起来：

$$
\hat Q_t=
\sum_{k=0}^{C-1}\gamma^k r_{t+k}
+\gamma^C\mathbb{E}_{\mathbf{a}'\sim\pi_\theta}
\left[Q_{\psi'}(x_{t+C},\mathbf{a}')\right].
$$

其中 $\mathbf{a}'$ 是以 $x_{t+C}$ 及对应 reference 为条件采样的下一动作块，$\psi'$ 表示 target critic。附录 B 进一步使用双 Q 的最小值构造 target。

:::{note}
**终止状态的补充说明**：原论文式 3 与算法 1 省略了 terminal mask。按标准 episodic TD 语义，终止之后不应继续 bootstrap。若以 $d_{t,C}$ 表示该转移是否终止，bootstrap 项应乘以 $1-d_{t,C}$。这是对公式语义的解释性补足；论文没有说明 episode 边界处不足 $C$ 步的动作块如何截断、折扣和存储，不能据此断言其具体实现。
:::

稀疏奖励设置下，多数动作块内的 reward 为零，学习信号依靠 bootstrap 逐段传播。$C$ 太小会拉长决策链，并增加查询 VLA 的压力；$C$ 太大则延长开环执行时间，降低对接触变化的响应频率。

RLT 取 $C=10<H=50$，名义上每段动作覆盖 0.2 s。这个时长来自动作频率和块长度，不能直接当作已测得的端到端推理时延。

### 3.3 从伪代码看训练顺序

```{figure} images/RL-Token/rlt-algorithm.png
:alt: 原论文算法 1，列出任务适配、VLA warmup、人工接管、经验回放和 actor–critic 更新
:width: 85%
:align: center

**原论文算法 1：RLT 训练流程**。表征与任务适配先于在线 RL；采集与 off-policy 更新异步进行。
```

下面按执行逻辑整理流程，作为阅读辅助，不是可以直接运行的机器人代码：

```text
阶段 A：任务适配
  每任务收集 1–10 小时遥操作示范
  联合进行 VLA SFT 与 RL-token 重建训练（2000–10000 步）
  冻结 VLA 与 RL-token 模块

阶段 B：在线强化学习
  用 base VLA 采集 warmup 数据，初始化 replay
  异步运行采集端与 learner：
    采集端：
      读取相机和本体状态
      VLA 产生 H=50 的参考动作及内部特征
      encoder 输出一个 2048 维 RL token
      actor 根据状态和前 C=10 步 reference 产生动作块
      若人接管，以人的动作替换执行动作与 replay reference
      执行动作，标记终止结果，按 stride 2 构造重叠转移
    learner：
      从 replay 采样，用 chunk-level TD 更新双 Q
      通过价值目标与 BC 正则更新 actor
      50% 样本遮蔽 actor 的 reference 输入，保留 BC 目标
  screw 与 zip tie 从受控关键阶段继续适配到完整任务到达状态
  可追加 VLA SFT，学习从 base VLA 切换至 RLT 的时刻
```

## 四、数据来源与回放语义

| 数据源 | 规模 | 用途 | 人工参与与未披露项 |
| --- | --- | --- | --- |
| 任务遥操作示范 | 每任务 1–10 h | VLA SFT 与 RL-token 重建 | 需遥操作；未给逐任务轨迹数、划分和过滤细节 |
| VLA warmup | $N_{\mathrm{warm}}$ 未给出具体值 | 初始化 replay | 需终止奖励标记；未给逐任务规模 |
| Online RL | 约 400–1000 episodes；15 min–5 h 有效 robot data | Actor–critic 学习 | 涉及奖励、切换与可选接管；未给完整逐任务分项 |
| 人工 intervention | 未报告规模 | 纠错与局部模仿目标 | 未给次数、时长和 replay 占比 |

:::{important}
**区分数据时长与实验耗时**：15 min–5 h 指排除复位等开销后的有效机器人数据，前面还有 1–10 h 的任务示范采集。图 7 中“5 分钟数据”对应当时约 40 分钟的实验耗时，不是 5 分钟内完成训练，也不是整条学习曲线的总实验时间。
:::

从学习语义看，回放需要关联以下信息；这张表不是论文代码的实际序列化 schema：

| 信息 | 用途 |
| --- | --- |
| 当前 RL token 与本体状态 | 构成 $x_t$ |
| 实际执行的动作块 | 供 critic 评价该转移 |
| VLA 或人工 reference | Actor conditioning 与 BC 锚点 |
| 动作块内奖励或其折扣和 | 构造 TD target |
| 下一 RL state 与对应 reference | 产生下一动作块并 bootstrap |
| 终止与有效动作长度信息 | 处理 episode 边界；具体实现未披露 |

受控评测从关键阶段前的部分完成状态开始，带轻微初始随机化，每个 agent、每个任务评测 50 episodes。这个数量不能直接套用到所有 full-task 结果上。（来源：§VI-A）

**我的理解**：人工分段和部分复位改变了训练分布，使有限交互更容易得到与困难操作相关的奖励。screw 与 zip tie 后续加入完整任务产生的到达状态，正是为了减少人工 reset 与真实执行之间的偏移。

## 五、模型架构与训练配置

| 组件 | 结构 | 在线阶段状态 | 作用 |
| --- | --- | --- | --- |
| VLA | $\pi_{0.6}$：SigLIP 400M、Gemma 4B、860M action expert | 冻结 | 感知与语言先验、reference chunk |
| RL-token encoder / decoder | 轻量 Transformers；token 为 2048 维 | 任务适配后冻结 | 压缩与重建 VLA embeddings |
| 普通任务 actor / critic | 2-layer MLP，hidden dimension 256 | 从零训练 | Gaussian actor、双 Q critic |
| Screw actor / critic | 3-layer MLP，hidden dimension 512 | 从零训练 | 为更困难任务提供更大容量 |

表中 layer 沿用论文口径，不额外解释成隐藏层数。（来源：图 2；附录 B）

已披露配置包括：

- 任务适配与 token 训练为 2000–10000 gradient steps。
- Actor 使用固定小标准差，reference 输入 dropout 为 50%；推理始终提供 reference。
- Target 使用两个 Q 函数的最小值。
- $C=10$，replay stride 为 2，UTD 为 5，critic 与 actor 更新次数比为 2:1。
- 训练采用 off-policy replay，采集端与 learner 异步运行。

论文未完整披露优化器、学习率、batch size、$\gamma$、$\beta$、$\sigma$、target 更新系数、replay capacity、动作归一化、数据增强、训练硬件、随机种子和 checkpoint 选择细节。不能将这些留白替换为某个常见算法的默认配置。

## 六、推理与机器人部署

每次决策先从三相机观测与本体状态出发，由冻结 VLA 产生 reference 和内部特征；token encoder 压缩特征后，actor 结合 RL state 与 reference 生成 10 步动作块。执行后再读取新观测，继续闭环控制。

原生 VLA chunk 为 50 步，RLT 使用前 10 步 reference。按 50 Hz 计算，动作块覆盖 0.2 s；论文没有报告 VLA/token forward、actor、通信的实测延迟，也没有完整描述 learner 与控制端的算力分配，因此该跨度不等于已验证的实时调度保证。（来源：§III；附录 B）

官方介绍强调小网络可高频更新，但论文中的 UTD 与更新比率不足以复现具体的每秒更新吞吐率。这里把系统描述与已给出配置的实验数据区分开。

关键阶段之外仍由 base VLA 控制；进入关键阶段后启用 RLT。自动切换可通过后续 VLA SFT 实现，但其独立指标与对最终成功率的贡献没有单独报告。

## 七、实验设计与结果分析

### 7.1 四个任务考察什么？

| 任务 | 主要困难 | 评测范围 |
| --- | --- | --- |
| M3 screw installation | 亚毫米对齐；约 10 cm 的工具杠杆放大腕部角度误差 | 关键阶段与完整任务 |
| Zip tie fastening | 双臂操作柔性物体，将尾端穿入狭窄锁孔 | 关键阶段与完整任务 |
| Ethernet insertion | 连接器位置与角度对齐，接触后的稳定插入 | 关键阶段；另有 baseline 与消融比较 |
| Charger insertion | 插脚可见性有限，要求厘米级对齐 | 关键阶段 |

四项任务的几何容差不同，不能全部概括成“亚毫米控制”。（来源：§VI-A；图 3）

指标包括二值成功率，以及 throughput——每 10 分钟成功完成的次数。后者同时受到成功概率和 episode 长度影响，更接近任务产出速度，但单看 throughput 无法区分提升来自更快还是更稳。

### 7.2 主要结果及其证据范围

```{figure} images/RL-Token/rlt-results.png
:alt: RLT 与基础策略的吞吐率和成功率柱状图，虚线框区分完整任务评测
:width: 100%
:align: center

**原论文图 4–5：吞吐率与成功率**。虚线框内为完整任务，其余为关键阶段评测；吞吐率单位为每 10 分钟成功次数。
```

| 结论 | 论文证据 | 结果与适用范围 |
| --- | --- | --- |
| 提高困难操作的可靠性 | 图 5 | Screw 关键阶段成功率由 20% 到 65%；完整任务 screw 与 zip tie 分别约增加 40、60 个百分点，后两项为图中近似读数 |
| 改善关键阶段速度 | 图 4、9 | Charger 与 Ethernet 报告约 3 倍提速；Ethernet episode 长度中位数由 base 的 228 降至 RLT 的 66 个控制步 |
| 超过该组遥操作示范的速度 | 图 9 | Ethernet 中 teleop 中位数为 146 步，RLT 为 66 步；一半 RLT trials 快于全部所比较的 demos，仅限这组关键阶段评测 |
| 少量有效交互后就能获益 | 图 7 | Ethernet 关键阶段采集约 5 min 数据后已优于比较策略；到该检查点的实验耗时约 40 min |
| 出现不同于示范的改进动作 | §VI-C | 作者观察到更流畅的插入、失败后施压并轻微摇动以利用柔顺性；属于定性分析，未统计出现频率 |

成功率、吞吐率、平均长度与中位数是不同统计量，不应把这些数值都解释成同一种“3 倍提升”。受控评测给出了每 agent、每任务 50 episodes，但随机种子数量和误差棒定义没有完整交代。

### 7.3 Baseline：比较只在 Ethernet 上展开

| 方法 | 更新对象或动作接口 | 与 RLT 的差别 | 比较设置与结果 |
| --- | --- | --- | --- |
| HIL-SERL | ResNet 加小型 actor–critic，单步动作 | 无 VLA token 与动作块先验 | 使用 20 个示范 episodes 及人工纠正；在论文的 Ethernet 设置下未学会任务 |
| PLD | Cal-QL 预训练 critic 加单步残差 | 需要残差尺度，决策链较长 | 先用 50 条 base rollouts 预训练 critic；表现较差 |
| DSRL | 选择 flow / diffusion 的噪声 latent | 在 VLA 可生成的行为模式中优化 | 实现中将 $(1,32)$ latent 沿首维重复 50 次；成功率接近 RLT，吞吐率更低 |
| DAgger | 用示范与相同 intervention 集合混合 SFT VLA | 通过模仿纠正更新，没有直接 reward improvement | 成功率较高，速度改进少于 RLT |

:::{note}
**比较边界**：图 6 的四种 baseline 只在 Ethernet 上比较，不能据此推出 RLT 在四个任务上全面领先。HIL-SERL 原设置为 10 Hz 且带 action bounding box，论文使用 50 Hz 且无该 bounding box；作者认为这些差异使任务更困难，但没有独立隔离各因素的影响。因此，失败结果不能直接解释为算法本质上的劣势。
:::

RECAP 与 GR-RL 属于相关工作，未作为这组实验中的 baseline 运行。前者讨论全模型的迭代离线 RL，后者结合过滤式 BC 与 latent noise 在线优化；RLT 的关注点则是冻结 VLA 后，用小型策略直接优化任务关键阶段的动作块。（来源：§II、§VI-C；附录 C）

### 7.4 消融能说明什么？

```{figure} images/RL-Token/rlt-learning-curves.png
:alt: Ethernet 任务中完整 RLT、不同 baseline 和移除关键模块后的学习曲线
:width: 100%
:align: center

**原论文图 7–8：学习曲线与消融**。横轴为有效机器人数据时长；移除 BC regularizer 的影响尤其明显。5 min 数据对应约 40 min 实验耗时。
```

- **w/o RL token**：换成 frozen ImageNet ResNet-10，吞吐率下降约 50%。这支持 VLA 表征有用，但没有排除更强视觉 encoder 或简单 VLA pooling 的替代解释。
- **w/o Chunk**：同时将 $C$ 改为 1，并因 50 Hz 查询 VLA 不可行而使用 ResNet-10。该变体难以可靠达到 base policy 水平，但存在两个变量一起改变的混杂。
- **w/o BC Regularizer**：令 $\beta=0$，性能下降最大，支持局部探索约束的重要性。
- **w/o Pass-Through**：移除 actor 的 reference 输入，最终可追上，但早期学习更慢、失败更多。这主要支持 reference 改善样本效率，论文没有独立量化硬件安全收益。
- **Reference dropout**：正文解释机制，附录给出 50% 的比例，但没有单独的去除 dropout 曲线，其净贡献尚未被隔离验证。

## 八、我的总结

RLT 最有价值的设计是 VLA 与在线 RL 之间的接口：重建式 RL token 提供紧凑状态，reference chunk 提供动作先验，小型 actor–critic 用 chunk-level TD 和 BC 锚定进行局部改进。冻结大模型后，在线更新成本集中在轻量网络上。

实验显示，这套组合能够改善四个操作任务的关键阶段，并在 screw 与 zip tie 的完整任务中带来成功率收益。结果的边界也很明确：主要基于一个 VLA，baseline 与消融集中于 Ethernet，效率依赖人工分段、复位、奖励与纠正；自动切换、完整配置和跨任务泛化保持仍缺少充分证据。

对我而言，这篇论文提供了一条具体的思路：已有 VLA 能把机器人带到困难操作附近时，可以把本机在线学习集中到这一小段，让状态表征、动作先验与高频更新各自承担清楚的职责。
