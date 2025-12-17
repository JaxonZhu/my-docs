# GR-RL: Going Dexterous and Precise for Long-Horizon Robotic Manipulation

![](docs/source/images/GR-RL/GR-RL-0.png)

**Abstract**

现 VLA 普遍假设高质量的人类演示数据集对 VLAs 性能至关重要 $\Longrightarrow$ 精细 / 灵巧任务上，人类演示数据具备噪声且可能存在 suboptimal $\Longrightarrow$ 在此类任务中 human demos 难以提供优质数据，重要性降低

GR-RL $\Longrightarrow$ 一个能将通用 VLA *generalist* 变成专精于长程灵巧操作的专家 *specialist*

1. 【过滤】GR-RL 训练一个以 "视觉+语言" 为条件的**任务进度预测器**，用于将演示轨迹中<font color=red>**对最终任务完成进度具有正向贡献的状态转移**</font>过滤保留下来 $\Longrightarrow$ <font color=green>Offline RL + 稀疏奖励产生的 $Q$ 值</font>可以作为鲁棒的进度预测函数
2. 【增广】GR-RL 引入形态学对称性增强技术 *morphological symmetry augmentation* ，显著提升了 GR-RL 的泛化能力和性能表现。
3. 【强化】GR-RL 通过学习**潜在空间噪声预测器**来实现 Online RL ，使 VLA policy 在部署时能协调高精度控制行为。

<font color=green>**首个基于学习的自主系鞋带系统**</font>，通过将鞋带穿入多个眼孔实现自动系带，成功率达 $83.3\%$ 。该任务需要具备<u>长时程推理能力</u>、<u>毫米级精度</u>以及<u>柔顺的软体交互特性</u>。

---

**1 Introduction**

通用性 $\neq$ 可靠性，当前 VLA policy 在实际部署中仍存在两个根本性缺陷：

1. 精准操控能力 —— 对可变形物体的<font color=green>**毫米级控制**</font>仍未解决；
2. 长时域鲁棒性 —— <font color=green>**误差会随操作步骤累积**</font>，当与高精度灵巧操作结合时，问题会进一步恶化。

<font color=orange>双臂系鞋带并非字节团队独辟，这个任务也有做了两三年了</font> $\longrightarrow$ 传统方法通过运动规划，采用<u>预定义的动作基元</u>和<u>设计模式</u>来解决系鞋带问题 $\Longrightarrow$ 对未见鞋 / 鞋带的泛化能力、故障恢复能力以及其他灵巧技能仍是一个开放性问题 $\longrightarrow$ 学习类：行为克隆的简单扩展 $\Longrightarrow$ 系鞋带技能的**次优性**和**局限性**。

> 我倒是挺好奇像这种灵巧性比较强的任务是如何 motion planning 的，展示一些早期论文：
>
> - **Haining Luo** and **Yiannis Demiris**. Bi-manual robot shoe lacing. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 8772–8775, 2023. doi: 10.1109/IROS55552.2023.10341934.
> - **Haining Luo** and **Yiannis Demiris**. Benchmarking and simulating bimanual robot shoe lacing. IEEE Robotics and Automation Letters, 2024.

---

使用普遍的 “录数据 —— 微调 —— 部署” 范式用于系鞋带任务会导致一些问题：

- 在录数据阶段，在需要极高精确度和灵巧操作的场景下，人类演示者会放慢动作、犹豫不决<font color=orange>（说白了人会手抖）</font>，并向 policy 系统引入噪音干扰的次优演示。
- 在训推过程中，VLA 通过滑动窗口预测固定长度的 action chunk 来模仿人类演示。然而，为了实现平滑的推理与控制，通常会**对预测轨迹进行平滑处理**。这些系统级优化方法对于基于学习的策略实现平滑执行至关重要，但不可避免地会导致<u>模型训练与推理之间的不匹配</u><font color=orange>（进而会在推理过程中引入累计误差 $\Longrightarrow$ 传统 BC 难以克服这种累计误差）</font>。

---

GR-RL 采用 Offline RL 对成功与失败的轨迹均进行 critic 模型训练。在给定稀疏奖励条件下，在回合结束时预测值自然反映任务进展，因此进一步利用该值筛选出对进展有积极贡献的状态转移，同时剔除其余状态转移。采用分布性 critics 方法，并观察到其在离线稀疏奖励场景下表现出更为稳健的性能。

GR-RL 从离线预训练 checkpoint 作为 base model 出发，通过 Online RL 进一步探索并修正 base policy 的失效模式 $\Longrightarrow$ 通过**引导去噪过程向高回报区域学习**来实现这一目标。

GR-RL 设计了一种简单而有效的方法，通过<font color=green>镜像机器人动作和观察结果</font>，并采用翻转文本描述来增强机器人动作。该方案显著提高了 VLA 的整体成功率和泛化能力。

> Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Nagabandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. arXiv preprint arXiv:2506.15799, 2025.

**2 The GR-RL Model**

![](docs/source/images/GR-RL/GR-RL-1.png)

Mixture-of-Transformer 结构 = Action $\pi_{\theta}$ 模型 + Crtitc 模型，一共 $5B$ 参数

**policy** $\pi_{\theta}$ ：$a_{t:t+k} = \mathbf{a}_{t} = \pi_{\theta}(l, \mathbf{o}_{t}, \mathbf{s}_{t})$

GR-RL 使用 Qwen2.5-VL-3B-Instruct 作为 VLM backbone，仅使用 VLM 层后半部分的 kv cache 进行快速推理，并通过 DiT 预测 action chunk 并通过 flow-matching 目标训练。

**Critic**：$Q_\phi(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)$ 具体而言，采用 Q-chunking 算法，为每个 action chunk 预测一组 $Q$ 值，并运用分布型强化学习进行训练。与基于无界回归的策略评估不同，分布型 critics 将价值视为**具有上下限的离散分布**，这种设计能自然捕捉现实轨迹中的不确定性。在稀疏奖励场景下，分布型 critics 展现出比非分布型更强的鲁棒性。通过将上限设为 1 、下限设为 0 ，训练出的 critics 能自然反映任务的推进状态。

> **如何理解分布型强化学习？这里以 DSAC 为例进行介绍：**
>
> DSAC 是一种 off-policy actor-critic 算法，它的设计初衷是结合两个方向：一是 Soft Actor-Critic (SAC) 的最大熵 RL 框架，二是 **distributional RL** ，即<font color=red>不仅估计期望 return，还估计 return 的分布 value distribution</font> 。通过**估计 return 的整个分布**而**不仅仅是期望 $Q$ 值**，DSAC 能<u>更好地反映未来的不确定性</u>；同时<u>可以扩展到 risk-sensitive 风险敏感的策略</u> —— 即 policy 不一定只优化期望 return，还可以基于分布的某种统计特性比如 variance, percentile, tail risk 等等选择动作。
>
> **==> 定义随机 return 分布**
>
> - 对于 policy $\pi$ ，定义累计回报为随机变量 $Z_\pi(s, a)$ ，而不是单一标量 $Q(s,a)$ 。
>
>   具体地，可写为：
> 
>   $$
>   Z_\pi(s_t, a_t) \;=\; r_t + \gamma (r_{t+1} + \gamma (r_{t+2} + \dots)) \quad \text{(或加上 entropy bonus term)}
>   $$
>
> - DSAC 将**试图学习这个随机变量的分布**，而<u>不是仅学习其期望值</u>。
>
> **==> 定义值分布 value-distribution 网络**
>
> - Value-distribution 网络以 $s$ 和 action $a$ 为输入，输出对 $Z_\pi(s,a)$ 的一个**连续分布 (continuous value distribution)** 的参数化表示<font color=red>（结合论文正文提到，这个分布还可以是有界限的）</font>。具体来说，<font color=green>DSAC 用 Gaussian 分布来近似这个 return 分布</font>。也就是说，critic 网络输出这个高斯的均值 mean 和标准差 std / 方差 variance —— 或者类似参数。这样，当环境 stochastic 或 reward noisy／随机时，这个分布能反映不同可能 return 的不确定性。
>
> **==> Distributional Bellman 目标**
>
> - 类似传统 RL 中 Bellman 更新，但对的是 return 的分布。也就是说，目标是让当前 critic 输出的分布 $Z(s,a)$ 满足 distributional Bellman operator 即：
> 
>   $$
>   Z(s,a) \stackrel{D}{\approx} r + \gamma Z(s', a')
>   $$
> 
>   这里 $\stackrel{D}{\approx}$ 表示 <font color=red>“在分布意义上近似 / 对齐 (in distribution)”</font> 。 critic 的训练目标就是把当前分布向这个 target 分布靠近。由于采用了连续 Gaussian 分布的 parameterization, 因为 return randomness 会引入很大方差, 更新时需要注意 gradient 的稳定性。
>
> **==> Actor (Policy) 网络 + Maximum Entropy Objective**
>
> - 与 SAC 类似，policy 是 stochastic 的, 通常输出对动作分布的参数，比如 Gaussian 的 mean & std，并且 policy 的目标不是简单 maximize expected return，而是 maximize **(return + entropy)** —— 即保持一定 exploration / 随机性 (entropy) 的 soft maximization 框架。 
> - Critic 提供的是回报分布 (而不是单纯期望)，actor 可以基于这个分布来做更新。如果是 “风险中立 (risk-neutral)” 设定，<u>actor 可能只是 maximize distribution 的期望 value</u>；也可以<u>选择风险敏感 (risk-sensitive) 设置</u> (e.g. maximize某个 percentile／variance‐aware objective) 。

**3 Training Recipe**

**3.1 Data Filtering with a Learned Task Progress Evaluator**

直接模仿人类演示的所有数据会不必要地引入具有噪声的动作到训练过程中，导致策略性能 sub-optimal 欠佳。同时，标注这些 sub-optimal 片段并非易事，甚至可能引入<u>更多主观且含噪的人类先验知识</u>。

为识别并过滤 sub-optimal 动作 $\Longrightarrow$ <font color=green>**采用 Offline RL 训练任务进度模型**</font> $\Longrightarrow$ 使用 TD3+BC 训练 critic

训练 critic 的奖励函数过程如下，使用该奖励函数对成功的轨迹进行标注：

$$
\begin{equation}
    r(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_{t}) =
    \begin{cases}
        \gamma^{T-t}\mathbb{I}(\tau), & t > T-k, \\
        0, & t \leq T-k,
    \end{cases}
\end{equation}
$$

- 对于成功的轨迹：最后一步奖励标记为 1 ，从后向前以折扣因子 $\gamma$ 按步数指数递减 $k$ 步，其他步奖励都是 0

- 对于失败的轨迹：整条轨迹每一步都是 0

- 对成功的轨迹进行类似 Hindsight **后视经验回放 HER** 的增广：

  最关键的就是这句: "Suppose frames $m_i, 0\leq i< M$ are marked as retry keyframes in a successful trajectory $\tau_{0:T}$, we can augment $M$ failed trajectories $\tau_{0:m_i}, 0\leq i<M$ in addition to the original successful one."

  如图所示：取一条成功轨迹的中间部分作为 retry keyframes $m_i, 0\leq i< M$ ，以这些 retry keyframes 为终点的轨迹可能是<font color=red>**不完整 / 不全局成功 / 可能局部成功**</font>的轨迹，根据上面的奖励函数标定规则<font color=green>**都设置成全 0 奖励数值**</font>。因此有 $M$ 个 retry keyframes 就能衍生出 $M$ 条失败的 episodes 。

![](docs/source/images/GR-RL/GR-RL-2.png)

有了这些成功和失败的轨迹，通过在这些数据上进行时序差异 temporal difference 学习，<font color=green>critic 模型 $Q_{\phi}$ 可作为鲁棒任务进度评估器</font>。

在获得任务进度模型后，让 $Q_{\phi}$ 做前向推理来计算其类别级分布的均值，作为数据集中所有转换的进度 $\rho$ 。

$$
\begin{equation}
    \rho_t =\mathtt{mean}(Q_{\phi}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)).
\end{equation}
$$

若在时间步 $t$ 的序列 $\rho_{t:t+k}$ 中存在超过特定阈值 $\delta$ 的数值下降，则将样本 $(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)$ 定义为次优 sub-optimal 样本，并将所有次优 sub-optimal 样本从数据集中剔除：使用过滤过的高优质数据用于策略克隆。

**3.2 Imitation Learning with Data Augmentation**

- 图像观测：将所有图像进行水平翻转 + 将左腕图像与右腕图像互换。
- 本体感觉状态： $s_t$ 和动作 $a_t$ 中的变换均通过世界坐标系中的镜像对称性转换，再转换回局部腕部坐标系。
- 翻转语言指令中的空间描述。

**3.3 Online Steering for Policy Deployment Alignment**

Online RL 在长程 / 精确性任务中表现困难 $\Longrightarrow$ 真实世界中极大探索空间 $\Longrightarrow$ 在潜空间进行结构化探索

$$
\begin{equation}
    \mathcal{L}(\pi_{\theta^{\prime}}) = \mathbb{E}_{(\mathbf{o}_t, l, \mathbf{s}_t)\sim \mathcal{D}} \left[-Q_{\phi^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{\epsilon}_t) + c \max( \frac{1}{2}\Vert \mathbf{\epsilon}_t\Vert^2 - \beta, 0)\right], \mathbf{\epsilon}_t\sim \pi_{\theta^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t),
\end{equation}
$$

$$
\begin{equation}
    \mathcal{L}(Q_{\phi^{\prime}}) = \mathtt{cross\_entropy}\left( Q_{\phi^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{\epsilon}_t),  Q_{\phi}(\mathbf{o}_t, l, \mathbf{s}_t, \pi_{\theta}(\mathbf{o}_t, l, \mathbf{s}_t|\mathbf{\epsilon}_t))\right), \\
    \mathbf{\epsilon}_t\sim
    \begin{cases}
        \mathcal{N}(\mathbf{0}, \mathbf{1}) & \text{w.p. } 0.5, \\
        \pi_{\theta^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t) & \textrm{otherwise}.
        \end{cases}
\end{equation}
$$

- 构建一个噪声预测器 $\pi_{\theta}^{\prime}$ 用于产生动作块的 diffusion / flow 去噪过程的初始 noise 

  与策略 $\pi_{\theta}$ 共享 VLM backbone

  从公式来看，根据 $(\mathbf{o}_t, l, \mathbf{s}_t)$ 三元组噪声预测器 $\pi_{\theta}^{\prime}$ 预测一个**噪声分布**，并从中**采样**噪声。

- 噪声预测器 $\pi_{\theta}^{\prime}$ 学习目标：

  （1）最大化 $(\mathbf{o}_t, l, \mathbf{s}_t)$ 和 $\mathbf{\epsilon}_t$ 的联合 $Q$ 值；（2）为避免从噪声中生成任何脱离离线训练数据集分布的动作，当噪声预测器 $\pi_{\theta}^{\prime}$ 的输出偏离原始正态分布超过特定阈值 $\beta$ 时，对其施加惩罚。

- 蒸馏价值函数 $Q_{\phi^{\prime}}$ ：

  在噪声空间 $Q_{\phi^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{\epsilon}_t)$ 上推导出 $Q$ 函数，以避免在策略优化过程中通过 flow 模型进行反向传播。

  $Q_{\phi^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{\epsilon}_t)$ $\Longrightarrow$ $Q_{\phi}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{a}_t)$ ，而原始 $Q_{\phi}$ 在上文部分使用 TD3+BC 方式训练出来了

  蒸馏价值函数 $Q_{\phi^{\prime}}$ 训练目标 $\Longrightarrow$ 蒸馏损失 $\Longrightarrow$ 从 $\pi_{\theta}$ 预测出的 $\mathbf{a}_t$ 产生的 $Q_{\phi}$ 数值尽量与 $Q_{\phi^{\prime}}(\mathbf{o}_t, l, \mathbf{s}_t, \mathbf{\epsilon}_t)$ 一致 $\Longrightarrow$ 分布型 RL 下 $Q_{\phi^{\prime}}$ 和 $Q_{\phi}$ 的分布尽量一致 $\Longrightarrow$ 交叉熵损失函数

  <font color=green>Tricks: 优化收敛过程，以半概率从噪声预测器 $\pi_{\theta}^{\prime}$ 预测分布中采样 initial noise ，以半概率从标准正态分布采样噪声 $\Longrightarrow$ 本质上还是优化探索过程，避免探索坍缩到比较小的区域中。</font>

<font color=orange>Comments: 这里实现了如何把极高维度的 action space 进行修改，使 online RL 能在结构化空间中进行探索 $\Longrightarrow$ 本质出发点：进行探索的空间未必就是动作空间，也可能是其他参数空间 $\Longrightarrow$ 选择 diffusion / flow 的噪声预测分布的参数作为探索。</font>

**[3.4] 数据流转过程**

为实现 sample-efficient 的 offline 到 online policy 迁移 $\Longrightarrow$ 采用离线策略缓冲区 $B_{\text{offline}}$ 和在线策略缓冲区 $B_{\text{online}}$ $\Longrightarrow$ 从这两个缓冲区中均匀抽取样本批次。

在训练开始前，通过使用离线训练 checkpoints 在线抓数据来预热离线策略缓冲区 $B_{\text{offline}}$ 

特意避免将远程操作轨迹混入缓冲区，以防止策略在不匹配的动力学环境中训练。

在线策略缓冲区 $B_{\text{online}}$ 仅存储最近两个 checkpoints 生成的轨迹，而过时数据则被推入离线缓冲区 $B_{\text{offline}}$ 。

**4 Robot & System**

这部分主要描述字节机器人 **ByteMini-v2** 设计以及改进点：增大力矩负载 / 调整底盘面积和转动自由度 / 外观

**5 Experiments**

**Task Description** 在模型推理过程中，整合了一个轨迹优化模块，该模块对动作的突变性和时间连续性施加约束，以优化预测的动作片段。采用二元稀疏奖励设置，即仅当鞋带穿过正确眼孔并完全放置于桌面时，才会获得 1 的正向奖励。

**Main Results** 

![](docs/source/images/GR-RL/GR-RL-3.png)

![](docs/source/images/GR-RL/GR-RL-4.png)

从超级长程任务的每个子进程成功率掉点图来看，掉点最严重的其实是 pick up the shoelace 抓起鞋带任务，而像 “把鞋带传进鞋眼 Thread into the eyelet” 其实没有多少掉点。直观来看 “把鞋带传进鞋眼” 确实更精细点...

**Ablation on the Progress Evaluator**

![](docs/source/images/GR-RL/GR-RL-5-1.png)

![](docs/source/images/GR-RL/GR-RL-5-2.png)

**7 Limitations & Conclusions**

轻量级噪声预测器能力优先 / 大型潜在碳素空间中信用分配的复杂性