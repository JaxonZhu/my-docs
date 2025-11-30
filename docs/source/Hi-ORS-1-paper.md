# Hi-ORS 人在环 + 拒绝采样 + 奖励加权过程监督

Human-in-the-loop Online Rejection Sampling for Robotic Manipulation

**Abstract**

RL ====> 不精确的价值估计 + 中间步骤的稀疏性监督

IL ====> 离线特性导致效果总是子最优

提出：Human-in-the-loop Online Rejection Sampling ====> 使用拒绝采样达到训练稳定性和鲁棒性

- 在 online 微调中过滤到负奖励样本
- 奖励加权的监督训练目标提供密集的中间步骤监督
- 异步的训推框架支持频繁的人在环内纠正 ====> 作为 “在纠正中学习” 的显式引导

评估：3 个真实任务 + 2 个具身实体，在 $\pi_{0}$ 作为 base model 

效果：在富触任务上训练 1.5 小时，表现超过 IL 和其他 RL 方法，这些方法都在性能和效率上表现出了性能边界

同时使用该方法微调的 VLA 展现出强大的 test-time 可扩展性

**I. INTRODUCTION**

作为纯离线 “利用” 的方法，模仿学习 IL 可能因**复合错误**导致灾难性故障：实际执行时的故障可能使系统<font color=red>进入离线数据集未包含的状态</font>，导致整个事件失败。

RL 方法通常需要<u>针对特定环境进行超参数调优</u>和<u>自由探索</u> $\longrightarrow$ VLA 中难以实现，且在**现实世界数据收集受限的情况**下成本过高 $\longrightarrow$ 如何实现机器人操作任务中 VLA 的稳定、灵活的 online post-training ？

**====> 作者认为，将 RL 参与 VLA 后训练存在不稳定性因素源于两个方面。**

- 不准确的价值估计

  强化学习使用神经网络近似动作价值函数，该函数容易被高估，尤其是在高维动作空间，例如 action chunk 中。

  > 神经网络逼近 $Q(s,a)$ 时具有估计误差。由于 Q-learning 更新中包含对动作空间取最大值的操作，该误差在 high-dimensional continuous action space 中被系统性放大，从而导致 Q-value bias towards overestimation，进而影响策略优化的稳定性和性能。

- 低效的监督

  VLA 通常受益于<font color=blue>在最终动作预测前的</font>**利用中间过程计算结果**，例如扩散步骤和流匹配分布变换步骤，但 RL 通常<font color=red>仅监督最终动作</font>，导致学习信号稀疏。

**====> 基于上述原因，作者提出 Hi-ORS ：**

- 这是一种简单高效的训练后方法，通过精准的<u>基于结果的价值评估</u>和<u>奖励加权拒绝采样目标</u>，有效稳定 online 学习过程。
- 核心是一种具有强**理论保证**的<u>拒绝采样目标</u>，该方法已在 LLMs 研究领域得到广泛应用
- Hi-ORS 不采用高方差值函数的学习方法，而是采用**基于结果的筛选机制**：系统会剔除**负向奖励**的 rollouts ，仅保留经 “黄金” golden 奖励模型验证的优质 episodes 。
- 为实现对中间推理步骤的密集监督，采用了一种简单通用的监督学习损失函数，该函数可同时训练最终动作预测和中间表征。

**====> Hi-ORS 系统能自然整合人工干预机制，有效指导 VLA 以掌握错误恢复行为，从而在测试阶段表现良好。**

Hi-ORS 系统在数据采集过程中能无缝集成灵活的人工干预，包括<u>远程修正、针对性重置以及在轨迹中途注入的简短校正段</u>。这些干预措施为错误恢复提供了明确指导，并通过失误与恢复行为丰富了缓冲区，而这类行为在离线数据集中极为罕见。

**====> 最后 Hi-ORS 被验证出来比常规方法好。**

经过微调的 VLA 具备强大的测试时扩展能力，能够反复执行复杂的错误恢复操作，从而提升测试性能。

**II. RELATED WORK**

**A. Imitation Learning for Robotic Manipulation**

模仿学习 $\longrightarrow$ 大规模 robotics 数据集 $\longrightarrow$ 预训练 VLA $\longrightarrow$ 在线对齐 / 后训练以满足动态变化的环境 $\longrightarrow$ 人在环内模仿学习 <font color=red>human-in-the-loop imitation learning: collects interventions during on-policy rollouts to correct compounding errors and to expand coverage to failure states, 在在线策展时收集干预数据，以便纠正复合误差并扩展智能体对失败经历的覆盖面</font> $\longrightarrow$ RaC: 仍然严重依赖人力，缺乏自我提升机制 $\longrightarrow$ 作者的方法维护了一个监督目标同时保持自改进性。

**B. Reinforcement Learning for Robotic Manipulation**

引入自我提升机制 $\longrightarrow$ 引入 RL $\longrightarrow$ 训练不稳定 $\longrightarrow$ 解决方案：

- 时序差分 / 混合目标 $\longrightarrow$ 只在仿真器上训练
- PA-RL $\longrightarrow$ 监督训练目标来稳定在线训练 $\longrightarrow$ 动作优化策略**高度依赖精准的价值评估** $\longrightarrow$ 人机协同训练存在冲突。

方法通过基于结果的拒绝策略，避免了不稳定的价值驱动策略更新

**C. Rejection Sampling**

拒绝采样 $\longrightarrow$ 一种通过过滤候选方案来从目标分布中抽取样本的经典方法 $\longrightarrow$ LLMs 中是指从**多个候选方案**中选取**前 $k$ 个** / **通过验证**的样本 $\longrightarrow$ STaR: 在迭代过程中，基于原始预训练模型自动生成的响应，这些响应需满足验证器的要求 $\longrightarrow$ 本文通过在线 VLA 部署中实施奖励级拒绝机制

**III. Hi-ORS**

**A. Preliminaries**

构建 MDP $(S, A, p, \rho, r, \gamma)$ 

$S$ 状态空间 $A=R^d$ 表示 $d$ 维度的动作空间 $p$ 状态转移概率位置且具有随机性 $\rho$ 初始状态分布 $r$ 表示奖励函数

定义整条轨迹出现的概率 $p^{\pi_\theta}(\tau) = \rho_0\left(s_0\right) \prod_{t=0}^{T-1} P\left(s_{t+1} \mid s_t, a_t\right) \pi\left(a_t \mid s_t\right)$

目标：$R^{\pi_\theta} = E_{\tau \sim p^{\pi_\theta}(\cdot)}[\sum_{t=0}^T \gamma^t r(s_t, a_t)]$

经典的策略梯度表达式：

$$
\begin{align}
\nabla_\theta \! L^{\text{PG}}(\theta) \!=\! - \!\mathbb{E}_{\tau \sim p^{\pi_\theta}(\cdot)} [Q_\phi(s,a) \nabla_\theta \log \pi_\theta(a_t|s_t)],
\end{align}
$$

其中 $Q_\phi(s,a)$ 作为 “动作-价值” 函数的近似值。

该公式揭示了两个主要的不稳定来源：

1. 估值不准确。$Q_\phi(s,a)$ 在动作空间维度较高时（例如动作分块场景）尤为突出；

   高维空间意味着：神经网络估计 $Q(s,a)$ 有更多误差方向 $\longrightarrow$ 有更多 candidate actions 会因为噪声而 “碰巧” $Q$ 比较高 $\longrightarrow$ $\max$ 操作就更容易 “捡到” 那个错误高估的 action $\longrightarrow$ 所以高维动作空间会放大这个 over estimation bias

2. 监督效率低下。$\log \pi_\theta(a_t|s_t)$ 仅关注最终动作，却忽略了最终动作预测前的中间计算过程 —— 这在当前的 VLAs 中具有重要意义（例如扩散策略的去噪步骤或自回归策略的 token 级生成）。

   > 例如 $\pi_0$ 采用动作分块技术预测多步骤动作序列，并运用流匹配实现连续动作生成。动作分块技术会随着预测时间范围的扩展而呈指数级增长动作空间，这使得 $Q_\phi(s,a)$ 在进行精确价值估计时面临显著挑战。

   > 使用 RL 进行训练 flow 时，需要通过时间反向传播（back-propagation through time, BPTT）进行迭代去噪，这会显著增加策略更新过程中的方差和计算开销。

   ====> <font color=blue>这句话的理解是：无论是基于扩散过程还是流匹配的 VLA ，使用 RL 只是对扩散 / 流的最终**结果**进行监督，因此忽略了扩散 / 流**过程**的监督，而不能理解成监督了 $s_t$, $a_t$ 而忽视了前面 $t-1$ 时刻的估计。</font>

<img src="C:\Users\aw\AppData\Roaming\Typora\typora-user-images\image-20251104233651891.png" style="zoom:50%;" />

**B. Rejection Sampling for Robotic Manipulation**

**1) Evaluation Phase**

给定一个轨迹的累计奖励 $R(\tau) = \sum_{t=0}^{T} r_t$ 使用一个指标函数来定义接受度标准：

$$
\begin{equation}
\mathcal{I}_m(\tau) = \mathbb{1}_{R(\tau) \geq m}
\end{equation}
$$

其中 $m$ 是奖励阈值，这个阈值会<font color=red>随着训练迭代过程而提高</font>。该过滤机制作为拒绝采样策略，将奖励阈值 $m$ 以下的轨迹排除，仅保留高绩效轨迹用于策略更新。

**2) Improvement Phase**

采用基于奖励加权的监督学习目标来更新策略，该方法通过模拟成功行为来实现

$$
\begin{align}
\nabla_\theta L^{\text{Hi-ORS}}(\theta) \!=\! - \!\mathbb{E}_{\tau \sim p^{\pi_\theta}(\cdot)} [I_m(\tau)  \nabla_\theta \log \pi_\theta(a_t|s_t)],
\end{align}
$$

====> 对于基于 flow-matching 的 VLA :

$$
\begin{align}
L^{\text{Hi-ORS}}(\theta)
\!=\!\!\!\! \mathop{E}\limits_{\substack{\tau \sim p^{\pi_\theta}(\cdot)\\ x^0 \sim N(0, I)\\ u \sim \mathrm{Unif}([0,1])}}
\!\!\!\!\Big[\underbrace{\color{mypink}I_m(\tau)}_{\color{black}\texttt{For I1}} \!
\underbrace{\color{myblue}\| v_\theta(u, s_t, x^u) \!-\! (x^1\! -\! x^0) \|_2^2}_{\color{black}\texttt{For I2}} \color{black} \Big],
\end{align}
$$

第一项是用于稳定值估计的指示函数，第二项是流匹配损失，它通过在去噪时间序列 $u$ 上提供密集监督，有效解决了系统不稳定的两个关键问题。

为实现持续优化，可采用**递增阈值**方案：在 $N$ 次训练迭代中逐步设置阈值 $m_1≤m_2≤\cdots≤m_N$ 。这种递增阈值的筛选机制，能**生成质量递增**但**规模递减**的数据子集。通过在更高质量数据子集上连续微调策略 $\{π_{\theta_{k}}\}_{k≥1}$，可确保策略性能的单调提升。

实际应用中，评估与改进阶段采用异步运行机制，通过独立的策略副本分别执行探索与训练，从而实现高效的离策略学习。该设计既能有效应对 VLA 推理带来的计算开销，又确保了学习过程的稳定性。此外，系统支持根据可用计算资源动态调整更新数据 update-to-data (UTD) 比例。

**C. Varied Frequency for Human Corrections**

在需要处理高维视觉数据和连续动作空间的复杂操作任务中，纯自主探索在现实场景中成本过高。

为解决这一难题，引入了具有双重关键作用的策略性人工干预：其一是通过引导策略聚焦状态空间中的潜力区域，实现高效探索；其二是通过示范错误恢复方法，向机器人展示如何从难以自主发现的故障模式中快速复原。

在自主部署过程中，Hi-ORS 系统支持操作员通过相对末端执行器控制或绝对关节控制，在任意时间点进行干预。单个轨迹中可能多次出现干预操作，形成混合型 “自主-人工” 干预场景。关键在于，仅保留符合 $\mathcal{I}_m(\tau)$ 筛选标准的正向奖励干预场景。这一机制确保了次优的人工修正不会污染训练数据。

Hi-ORS 采用基于控制权限的自适应交互频率：

$$
\begin{align}
f_{t} = \begin{cases}
f^{\text{high}}, & t \in \text{human intervention period;} \\
f^{\text{low}}, & t \in \text{autonomous control period,}
\end{cases}
\end{align}
$$

在人工干预时，会以更高的频率记录状态转换，以便捕捉更精细的纠正行为；而在自主执行时，则采用较低的频率，以确保策略执行的一致性，避免出现动作突兀或回退行为。

**D. Asynchronous Infrastructure**

在给定 $G$ 个GPU的情况下，预留一个 GPU 用于在线推理，其余 $G−1$ 个GPU用于学习。

**执行节点**向**学习节点**传输数据流，多个学习器通过 agentlace 协议进行模型更新。通过 ZeRO-2 框架对学习器进行编排，实现高容量 VLA 的大规模分布式训练。这种异步执行器学习器设计将训练吞吐量提升约 2 倍，即使机械臂停止工作也能持续学习 —— 这在长期实际运行中是常见场景。

为确保训练稳定性，当相对变换范数低于阈值时会过滤无效操作，同时剔除极短的训练片段以避免动作分块错误。总延迟由三部分组成：模型推理延迟（约 160 毫秒）、通信延迟（约 400 毫秒）和顺序执行时间（约 900 毫秒）。单次迭代训练耗时约 1.5 秒，因此自然 UTD 比约为 1 。在动作分块模式下，典型推理步骤约 20 步，导致每集训练耗时约 20 秒。

**IV. EXPERIMENTS**

*Raise-Hand* 任务：Paxini Tora One 机器人被要求将左臂抬至目标姿势，其动作空间包含末端执行器的绝对位置和左臂夹具的张开程度，通过 Meta Quest 3进行人工干预。

*Pack-Detergent* 任务：Paxini Tora One 机器人被要求从传送带上取洗衣粉，然后放入纸箱中。

*Insert-Moisturizer* 任务：Dobot X-Trainer 机械臂需取用细长保湿霜并将其置入底座。动作空间包含绝对关节角度和夹爪张开度。主要机械臂通过关节映射进行干预。所有任务的观测空间由手腕顶部和左侧摄像头的图像、本体感觉以及任务指令组成。

**baseline**: behavior cloning / HIL-SERL / Q-Chunking

**base model**: $\pi_0$

由于 behavior cloning 、HIL-SERL 和 Q-Chunking 均需离线数据支持，为其他方法收集了初始人类示范数据。

**A. Real-world Experiments**

<img src="C:\Users\aw\Desktop\工作学习\论文合集-VLA\Hi-ORS-2.png" style="zoom:75%;" />

HIL-SERL: 不稳定性伴随振荡回归随着训练数据量的增加

Q-Chunking: 通过引入蒸馏损失来稳定性能，但会阻碍性能的进一步提升。

**B. Learning Dynamics**

为验证这一结论，评估了 Hi-ORS 算法与行为克隆策略在不同测试预算下的最终性能表现。

Hi-ORS 算法在 test-time scaling 方面表现突出。这种持续优化的趋势表明，该策略能有效利用额外重试机会来修复中间错误，而非简单重复失败操作。此外，图中数据还显示，增加测试时间计算量带来的边际效益逐渐递减。相比之下，行为克隆策略的扩展性效果明显不足，这说明其在测试阶段进行针对性修复的能力存在局限性。

**C. Spatial Generalization**

在本小节中，采用课程数据收集策略评估 Hi-ORS 的空间泛化能力。首先通过人工干预将物体<u>初始定位在棋盘格点位</u>，收集相关数据，并在测试场景中将<u>物体置于棋盘格中间区域</u>进行 Hi-ORS 测试。随后采用不同训练和测试场景进行类似实验。

<img src="C:\Users\aw\Desktop\工作学习\论文合集-VLA\Hi-ORS-3.png" style="zoom:50%;" />

即使在物体远离机器人重置起始位置的极端情况下，Hi-ORS 仍展现出强大的空间泛化能力。这种泛化特性得益于 Hi-ORS 的在线数据收集特性，能够快速调整操作策略。

**D. Error Recovery**

<img src="C:\Users\aw\Desktop\工作学习\论文合集-VLA\Hi-ORS-4.png" style="zoom:75%;" />

**E. Ablation Studies**

**a) Choice of learning scheduler**

最初假设采用<font color=red>循环调度器</font>可能缩短训练时间 —— 因为高学习率能更快适应新数据，而低学习率则有助于收敛。

但消融实验表明，循环调度器对训练时间和成功率的影响微乎其微。

基于奥卡姆剃刀原则，在 Hi-ORS 的最终版本中去除了该机制。

> 循环调度器是指把 learning rate 在训练过程中做周期性上下波动的学习率调度策略，而不是固定单调递减。

> Occam’s Razor（奥卡姆剃刀）如果同一件事存在多个解释 / 多种方法，但效果差不多或都能成立——应该选择那最简单的那个方案。因为 ablation 发现 cyclical LR 对结果影响很小、没有显著增益 —— 但是带来了额外复杂性，所以基于 Occam’s Razor，他们选择 **删除** 这个设计，保留一个更简单、效果相同或更好的版本。

**b) Choice of reward model**

用学习得到的奖励分类器替代人工标注的奖励，成功概率反而降低，且训练时间没有节省。

这主要是因为奖励模型在人机交互的错误恢复过程中，可能会预测出**假阳性奖励**，导致训练过程混乱。

> Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning.

**c) Importance of human correction**

当取消人工干预时，发现模型成功率明显下降，因为无法有效执行错误恢复机制来重试评估任务。

取消人工干预还会延长训练时间，因为模型在缺乏明确指导的情况下，可能需要耗费极长时间才能获取正向奖励样本。这一结果验证了人工干预在 Hi-ORS 中的重要性。

**d) Choice of data filters**

移除 **no-op 动作过滤器**会拖慢训练进度，成功率跌至 $20\%$ 。该发现与文献[5]结论一致

移除**短 episode 过滤器**后，成功率降至 $60\%$ ，说明剔除无信息量的运行轨迹能提升学习稳定性。

**e) Varied execution frequency**

该动态频率策略在人类干预阶段采用高频模式以获取更多数据点，而在模型执行阶段则切换为低频模式以避免回溯操作。

若将动态频率固定为较低频（如 5 步）或较高频（如 25 步），都会导致性能下降，这验证了该策略的有效性。

**V. CONCLUSION**
