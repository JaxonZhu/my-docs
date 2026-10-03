# LWD 论文解读：在机群部署中持续强化学习

论文原题：Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies

- 作者：Yi Wang、Xinchen Li、Pengwei Xie 等，通讯作者 Jianlan Luo
- 机构：Shanghai Innovation Institute、AGIBOT Finch、Columbia University
- 发表：arXiv，2026 年 5 月 1 日首次提交
- 论文：[arXiv:2605.00416](https://arxiv.org/abs/2605.00416)
- 原文：[项目提供的论文 PDF](https://finch-static.agibot.com/LWD/lwd-paper.pdf)
- 项目：[LWD 官方项目页](https://finch.agibot.com/research/lwd)

Learning While Deploying（LWD）把部署产生的成功、失败和人工纠正汇入共享 replay，持续更新一个通用 VLA 策略，再将新策略下发给机器人机群。它以 DIVL 学习价值，以 QAM 改进 flow-based 动作生成，在 16 台双臂机器人、8 个任务上验证这套离线到在线的训练闭环。

我关注的重点是：部署经验究竟提供什么学习信号，分布型价值模型如何引导策略，以及异步数据采集怎样与多机训练保持一致。

:::{note}
**阅读范围**：正文依据所提供的 PDF 笔记整理，公式、配置与实验口径对照项目提供的 18 页论文核对。arXiv 后续有版本更新，此处不将项目 PDF 等同于最新版本，也未进行训练代码复核。论文结论、我的理解和待验证问题分别表述。
:::

:::{important}
**先看指标口径**：主结果的平均分为 0.95，但八个任务使用两类评分：四个补货任务是二元成功率，四个长任务是子步骤平均得分。0.95 不能直接解释为所有完整任务都有 95% 的成功率。
:::

## 一、为什么要在部署中继续学习？

### 1.1 固定离线数据难以覆盖持续变化的环境

预训练 VLA 从大规模数据获得通用操作能力，但真实部署还会带来新物体、新布局、语言变化、长尾失败和人工纠正。固定演示数据集难以提前包含所有这些情况。

作者希望把部署本身变成持续的数据来源：不同机器人探索不同场景，经验汇入同一训练过程，共享策略改进后再影响后续采集。这里的机群不仅增加并行执行数量，也扩大了单个策略能接触的状态分布。（来源：§I–II）

### 1.2 学习信号与更新节奏是两个不同问题

原笔记比较了离线 RL、在线 RL 和交互式模仿学习。理解它们时，需要分开看“从什么信号学习”和“什么时候更新”。

| 路线 | 部署经验提供的主要信号 | 更新方式与局限 |
| --- | --- | --- |
| 迭代离线 RL，如论文中的 RECAP baseline | 任务结果、价值与 advantage 标签 | 收集后集中训练，再重新部署；新经验需要经过这一轮流程才能影响策略 |
| 交互式模仿学习，如 HG-DAgger | 人的纠正动作，以及选定的 rollout 动作 | 提供直接的动作监督，但没有由终局奖励进行 Bellman 更新的机制 |
| On-policy 在线 RL | 当前策略的交互结果 | 可直接优化回报，但真机交互昂贵，数据复用受到策略更新方式限制 |
| LWD | 成功、失败、自主动作和人工纠正组成的 replay | 用 off-policy RL 持续复用离线与在线数据，采集与学习异步推进 |

**我的理解**：RECAP 与 HG-DAgger 的区别不只是更新快慢。奖励描述“这次行为最终带来了什么结果”，动作标签描述“在这个状态应该怎样做”。LWD 希望既使用前一种结果信号，又缩短新经验进入下一次策略更新的周期。

### 1.3 失败轨迹怎样提供信息？

以泡茶时打翻容器为例，模仿学习可以在有人纠正时学习对应动作；价值学习还可以利用这次失败的终局结果，调整相关状态与动作的价值估计。

但稀疏终点奖励不会自动标出“前六步都好，第七步才出错”。一条失败轨迹本身的回报可能全为零。要区分可恢复的中间状态与不可恢复的动作，仍需要数据覆盖、成功或恢复经验，以及函数近似和 bootstrap 的有效泛化。

因此，“利用失败数据”不等于自动知道失败根因，也不意味着失败轨迹的每一段都会获得正价值。所谓轨迹拼接，依赖不同经验之间存在可以学习的状态联系。

### 1.4 两个算法难点

首先，机群 replay 混合任务、场景、策略版本和人工干预，价值学习需要从异构数据中提取有用信号。其次，flow policy 通过多步去噪生成动作，直接把 critic 梯度反传穿过整个生成过程，可能带来较大的计算开销和不稳定性。

LWD 分别用 **DIVL**（Distributional Implicit Value Learning，分布型隐式价值学习）和 **QAM**（Q-learning via Adjoint Matching，伴随匹配策略提取）处理这两个部分。（来源：§IV）

## 二、整体流程与模块接口

LWD 先用示范适配预训练的 $\pi_{0.5}$，得到行为克隆参考策略；随后在固定离线 buffer 上进行 RL，再将策略部署到机器人机群，从不断增长的在线 replay 中继续学习。

```{figure} images/LWD/lwd-overview.png
:alt: LWD 的离线到在线训练流程，以及 DIVL 价值模型和 QAM 策略提取的网络结构
:width: 100%
:align: center

**原论文图 2：训练流程与模型结构**。策略网络负责动作生成，独立的 value/critic 网络提供价值信号；在线经验与离线数据共同支持持续更新。
```

| 模块 | 输入 | 输出或职责 |
| --- | --- | --- |
| VLA policy | 相机观测、任务指令与机器人状态 | 长度 $H=30$ 的动作块 |
| 分布型 value $V_\psi$ | 状态 $s$ 的 readout 表示 | Replay 动作对应的标量 Q 值分布 |
| 双 critic $Q_\phi$ | 状态表示与动作块 | 标量动作价值，取较小估计以抑制高估 |
| QAM | 固定参考 flow、critic 的动作梯度 | 当前策略 flow 的局部回归目标 |
| 共享 replay | 离线数据、在线自主 rollout 与人工纠正 | Learner 使用的混合 minibatch |
| 机群数据系统 | 完整 episodes、模型 checkpoint | 版本化数据视图与策略分发 |

这里有两个容易混淆的概念：DIVL 的 **distributional** 指价值分布建模；机群基础设施的 **distributed** 指跨机器人、跨主机运行。前者是学习方法，后者是系统组织方式。

论文训练一个跨八个任务共享的策略，而非给每台机器人分别训练一个独立 specialist。Policy 与 value/critic 的 backbone 也不是同一个网络，只有策略 checkpoint 会下发到机器人进行推理。（来源：§IV-D）

## 三、DIVL 怎样从分布中提取价值？

### 3.1 先理解 IQL 的隐式价值选择

IQL 通过 replay 中的动作价值学习一个偏向高价值动作的状态值，不显式执行全动作空间的 $\max_a Q(s,a)$。其 expectile 回归可以写成：

$$
\mathcal L_V(\psi)=
\mathbb E_{(s,a)\sim\mathcal D}
\left[L_2^\tau\bigl(Q_{\bar\phi}(s,a)-V_\psi(s)\bigr)\right],
$$

$$
L_2^\tau(u)=|\tau-\mathbf 1(u<0)|u^2.
$$

当 $\tau>0.5$ 时，$Q-V>0$ 的误差受到更大的权重，状态值因而偏向数据中的较高 Q 值。这里得到的是 **expectile**，不能直接称为同参数的 quantile。（来源：§III-B）

避免显式最大化，有助于减少 bootstrap 目标依赖数据外动作的问题；它不是对整个训练过程“绝不外推”的保证。

### 3.2 DIVL 建模的究竟是什么分布？

DIVL 的分布型 value 接收状态，拟合这个状态下 replay 动作对应的 critic 值分布。用随机变量表示，其核心定义是：

$$
A\sim\mathcal D(\cdot\mid s),\qquad
V_\psi(s)\approx\operatorname{Law}\bigl(Q_\phi(s,A)\bigr).
$$

也就是说，同一个状态下，不同数据动作可能被 critic 赋予不同价值。DIVL 保留这些值的分布，再从中提取统计量；**Q 本身仍是标量 critic**。（来源：§IV-A，式 11）

这与“直接学习固定状态—动作对的完整随机回报分布”并不完全相同。论文用异构回报作为动机，但具体模型定义应以状态条件下的 replay action-values 为准。

**我的理解**：标量期望本身并不假设回报确定，也不要求同方差。DIVL 的区别在于保留更丰富的条件分布信息，并允许训练过程选择不同分位数；不能把所有标量价值学习都简化为错误的平均。

### 3.3 C51 风格投影与交叉熵

模型使用固定 support 上的 $C=201$ 个等距 atoms，范围为 $[-0.1,1.1]$。对于样本 $(s,a)$，先由 EMA critic 给出标量监督 $q=Q_{\bar\phi}(s,a)$，将其裁剪到 support，再线性分配到相邻两个 atoms，得到目标分布 $m$。

若 $v_j\le q\le v_{j+1}$，则：

$$
\begin{aligned}
m_j&=\frac{v_{j+1}-q}{v_{j+1}-v_j},\\
m_{j+1}&=\frac{q-v_j}{v_{j+1}-v_j}.
\end{aligned}
$$

其他位置的目标质量为零，随后最小化交叉熵：

$$
\mathcal L_V(\psi)=
-\mathbb E_{(s,a)\sim\mathcal D}
\left[\sum_{c=1}^{C}m_c\log p_{\psi,c}(s)\right].
$$

例如，**仅为说明插值**，若相邻 atoms 是 0.72 与 0.74，目标为 0.73，两者各获 0.5 的概率质量。LWD 的实际等距间隔为 $1.2/200=0.006$，并不是这个示意例子中的 0.02。（来源：附录 A1）

单条训练样本投影的是一个标量；模型跨 replay 样本拟合后，才形成状态条件下的价值分布。

### 3.4 分位数、众数与均值分别回答什么？

令 $F_\psi(v\mid s)$ 为累计分布，DIVL 取：

$$
\operatorname{Quant}_\tau(V_\psi(s))
=\inf\{v:F_\psi(v\mid s)\ge\tau\}.
$$

**众数**选择概率质量最大的 atom；**均值**对所有值加权平均；**分位数**按累计概率定位。下面是一个说明统计量差别的例子，不是论文中的实测分布：

| 价值 | 概率 | 累计概率 |
| --- | --- | --- |
| 0 | 0.3 | 0.3 |
| 0.5 | 0.4 | 0.7 |
| 1.0 | 0.3 | 1.0 |

这里均值与众数都是 0.5，0.5 分位数为 0.5，0.9 分位数为 1.0。较高分位数可以选中右侧的高价值区域，但它既不是最大值，也不是“成功概率为 90%”的保证，更不是固定取出最好的 10% 动作。

对于概率集中在同一个 atom 的离散分布，分位数还可能长时间不变。因此，数据变好不必然体现为分位数数值持续上升；估计覆盖度与可靠性同样重要。

### 3.5 分布拟合与 expectile 有什么理论关系？

论文将两类统计量放在一个非对称损失族中：

$$
\rho_{\tau,p}(u)=|\tau-\mathbf 1(u<0)|\,|u|^p.
$$

$p=2$ 对应 expectile，$p=1$ 对应 quantile。命题 1 说明：在分布拟合准确、离散化足够细等理想条件下，先拟合分布再提取相应统计量，与直接优化**同一个非对称损失**得到相同的最优标量价值估计。

这不是说同一 $\tau$ 下 expectile 与 quantile 彼此相等，也不是对有限数据、201 个 atoms 和实际训练过程的等价性保证。（来源：§IV-A；附录 A2）

### 3.6 用熵调整 bootstrap 的乐观程度

DIVL 使用归一化熵描述预测分布的分散程度：

$$
\mathcal H(s)=
-\frac{1}{\log C}\sum_{c=1}^{C}p_{\psi,c}(s)\log p_{\psi,c}(s).
$$

然后设置：

$$
\tau(s)=\operatorname{clip}
\bigl(\tau_{\mathrm{base}}-\alpha\mathcal H(s),
\tau_{\min},\tau_{\max}\bigr).
$$

熵高时降低 $\tau$，取较保守的分位数；熵低时保留较乐观的目标。离线阶段 $\tau_{\mathrm{base}}=0.6$，在线阶段为 0.9，$\alpha=0.3$。计算 TD target 时，$\tau$ 被视为 stop-gradient。（来源：式 17–18；附录 B2）

**我的理解**：这是利用分布熵调节乐观度的启发式规则。熵同时可能反映动作差异、任务随机性与拟合情况，不是经过校准的 epistemic uncertainty。低熵也可能是模型过度自信，不能直接等同于“数据充分、估计正确”。

### 3.7 分位数如何进入策略改进闭环？

设动作块长为 $H$，块内折扣奖励为：

$$
\bar r_t=\sum_{i=0}^{H-1}\gamma^i r_{t+i}.
$$

未终止时，单个 chunk 的 TD target 为：

$$
y_Q=\bar r_t+\gamma^H
\operatorname{Quant}_{\tau(s_{t+H})}\bigl(V_\psi(s_{t+H})\bigr).
$$

critic 最小化 $\mathbb E[(Q_\phi(s_t,\mathbf a_t)-y_Q)^2]$。于是，下一状态的分位数影响当前动作的 Q 值；QAM 再使用 $\nabla_a Q_\phi$ 改进策略；新策略采集的数据反过来更新 replay。（来源：式 14–15）

这条链路解释了分位数如何参与学习，而非仅作为展示用的统计量。但“更高分位数 → 更好的动作 → 更高成功率”仍依赖 critic 梯度质量，不是单调改进定理。Reference 生成的动作及策略更新也可能离开数据充分覆盖的区域。

## 四、QAM 怎样更新 flow-based 策略？

### 4.1 Flow 策略通过积分生成动作块

Flow matching 以 Gaussian 噪声 $a^0$ 和数据动作 $a^1$ 为端点，使用插值：

$$
a^w=(1-w)a^0+wa^1,\qquad w\in[0,1],
$$

训练条件向量场 $f_\theta(s,a^w,w)$。推理时沿向量场积分，从噪声得到完整动作块。（来源：§III-C）

如果直接最大化生成动作的 critic 值，就需要通过生成过程计算 $\nabla_\theta Q_\phi(s,a^1_\theta)$。多步积分中的 Jacobian 连乘可能放大或衰减梯度，保留中间计算图也可能增加内存成本。

**我的理解**：这说明直接反传可能昂贵或不稳定，不能理解成数学上不可用。开销还与求解器、缓存和重计算有关；论文没有证明每个去噪步都需要完整重跑视觉语言 backbone，也没有给出“30 倍全模型激活”的实测结论。

### 4.2 参考策略与 KL 正则化的改进目标

QAM 固定行为克隆参考策略 $\pi_\beta$，以 critic 定义改进目标：

$$
\pi^*(a\mid s)\propto
\pi_\beta(a\mid s)\exp\!\left(\frac{Q_\phi(s,a)}{\lambda}\right).
$$

参考策略提供先验，指数项偏向高 Q 动作；$\lambda$ 控制价值偏好与参考约束之间的权衡。LWD 使用 $\lambda=2$。参考 flow $f_\beta$ 固定为离线 RL 开始前的 BC checkpoint，当前 flow $f_\theta$ 则在离线与在线阶段继续优化。（来源：式 8；§IV-B）

### 4.3 将终点价值梯度转成逐步监督

首先沿固定参考 flow 生成轨迹，在去噪终点计算：

$$
\tilde g_1=-\nabla_a\left[Q_\phi(s,a^1)/\lambda\right].
$$

再沿参考轨迹求解伴随动力学，得到中间时刻的 $\tilde g_w$。记 $f_\delta=f_\theta-f_\beta$，LWD 论文给出的目标为：

$$
\mathcal L_{\mathrm{QAM}}(\theta)=
\mathbb E\!\left[
\int_0^1
\left\lVert
\frac{2f_\delta(s,a^w,w)}{\sigma_w}
+\sigma_w\tilde g_w
\right\rVert_2^2\,dw
\right],
$$

$$
\sigma_w=\sqrt{2(1-w)w}.
$$

以上记号与表达式沿用 LWD 式 9–10。固定局部目标时，回归希望 $f_\delta$ 接近 $-\sigma_w^2\tilde g_w/2$，从而将终点 critic 梯度转化为生成过程中的局部更新方向。

:::{note}
**两处计算边界**：伴随计算仍涉及参考向量场对动作状态的导数，通常以 Jacobian-vector 或 vector-Jacobian 运算实现；冻结参数不等于不需要这些导数。另外，损失同时包含 $\sigma_w$ 与 $1/\sigma_w$，不能只因 $\sigma_w$ 在端点趋零就断言端点不重要。端点离散化与数值处理需要结合实际实现，本文不补造代码细节。
:::

### 4.4 这套方法保留了什么，又依赖什么？

QAM 将整段生成过程的优化转成沿参考轨迹的局部回归，保留 flow 模型表达多模态动作的能力。它并不意味着动作生成路径完全不变：更新后的 $f_\theta$ 会产生不同于参考策略的动作分布。

**待验证问题**：如果参考策略在某类状态下很差，其生成轨迹是否足以提供可改进的起点？同一个温度是否适合所有任务？LWD 没有单独比较 QAM 与其他策略提取机制，也没有把原 QAM 论文中的 flow 步数或温度敏感性结果重新做成 LWD 消融。

## 五、数据与离线到在线训练

### 5.1 离线数据包含成功与失败

| 数据来源 | 时长 | 占比 | 含义 |
| --- | --- | --- | --- |
| Demonstrations | 336.6 h | 51.6% | 人类专家的成功示范 |
| Successful rollouts | 88.8 h | 13.6% | 历史策略的成功执行 |
| Failed rollouts | 39.2 h | 6.0% | 历史策略的失败执行 |
| Play data | 187.9 h | 28.8% | 人工引导探索失败模式与边缘情况，按未成功数据处理 |
| 合计 | 652.5 h | 100% | 用于离线 RL 的固定 buffer |

```{figure} images/LWD/lwd-data.png
:alt: 652.5 小时离线数据按任务和示范、成功执行、失败执行及 play 来源划分
:width: 100%
:align: center

**原论文图 7：离线数据组成**。长任务占 81.2% 的数据时长；失败 rollout 与 play 合计约占三分之一。
```

四个补货任务合计 122.7 h，四个长任务合计 529.8 h。时长差异也与 episode 更长有关，不能直接等同于训练时每个任务的采样权重。（来源：附录 B1，表 IV）

**我的理解**：失败数据不等于应当删除的低质量数据。它可能提供失败边界、恢复状态和负结果信息；但“分布建模”也不是自动数据清洗或轨迹优先级机制。线上线下 1:1 的采样比，只平衡两个数据源，并不解决所有任务不平衡。

### 5.2 从 episode 到 chunk transition

观测包含相机与机器人状态，语言指令指定任务。连续动作按 $H=30$ 组织为动作块，执行后记录下一状态。抽象训练样本为：

$$
(s_t,\mathbf a_t,\bar r_t,s_{t+H}),\qquad
\mathbf a_t=[a_t,\ldots,a_{t+H-1}].
$$

成功终止时原始 reward 为 1，其他时刻为 0，再在 chunk 内计算折扣和。实际存储还需要处理终止与有效长度，但这里的元组是学习语义，不是代码中的完整序列化格式。

人工接管时，实际执行的纠正动作作为常规在线 transition 存入 replay。接管本身不获得一个额外的即时正奖励，仍按照 episode 的终局成功或失败分配 reward。（来源：§III-A；§IV-C）

### 5.3 三个训练阶段的更新范围

| 阶段 | 数据 | 目标与更新范围 |
| --- | --- | --- |
| BC 初始化 | 成功示范 | 适配预训练 $\pi_{0.5}$，得到所有后训练方法共用的 reference checkpoint |
| Offline RL | 652.5 h 固定 replay | DIVL 与 QAM；policy、value/critic 全量微调 |
| Online RL | Offline 与 online replay 约 1:1 | Policy 冻结 VLM backbone，仅更新 action expert；value/critic 继续全量微调 |

参考 flow $f_\beta$ 始终固定，不能与不断发布到机器人的当前策略 $f_\theta$ 混为一谈。在线阶段冻结视觉语言参数，有助于控制更新成本，但没有因此证明通用能力完全不变。（来源：§IV-D；附录 B3）

### 5.4 离线多步 TD，在线单步 chunk TD

离线阶段，补货任务使用 $n=1$，长任务使用 $n=10$ 个 chunk 的 target：

$$
y_Q=
\sum_{i=0}^{n-1}\gamma^{iH}\bar r_{t+iH}
+\gamma^{nH}
\operatorname{Quant}_{\tau(s_{t+nH})}
\bigl(V_\psi(s_{t+nH})\bigr).
$$

若在窗口内终止，就截断回报并去掉 bootstrap 项。这里的“10-step”是 **10 个动作块**，不是 10 个底层控制步。（来源：式 19）

多步 target 有助于加快固定离线 buffer 内稀疏奖励的传播。在线阶段全部改用 $n=1$；作者认为，多步片段可能跨越策略动作与人工干预，使其不再对应单一策略的执行，加上已有离线初始化，单步 chunk 更新更合适。

这是论文对观察结果的解释，不是“off-policy RL 禁止多步 TD”的一般结论。共享训练目标旨在缓解 offline-to-online 的失配，也不保证 critic 在新状态上始终校准良好。

## 六、模型架构与训练配置

### 6.1 Policy 与 value/critic 使用独立 backbone

| 模块 | 结构 | 作用 |
| --- | --- | --- |
| Policy VLM | PaliGemma：Gemma-2B 与 SigLIP | 编码视觉、语言与任务情境 |
| Policy action expert | Gemma-300M，flow-based action head | 生成动作块 |
| Value/critic backbone | Gemma3-270M-IT 与 SigLIP-So400M | 以 readout token 提取状态表示 $z_t$ |
| Value head | Categorical head，201 atoms | 预测状态条件价值分布 |
| Critic heads | 动作块经 learned temporal attention pooling 编码，再与 $z_t$ 拼接，输入两个 scalar heads | 预测动作价值，采用 clipped double-Q |

Value 与 critic 共享自己的 backbone；它们与 policy 不共享 backbone。价值网络的视觉投影层和预测 heads 从零初始化，预训练语言与视觉模块则作为初始化来源。（来源：§IV-D）

论文没有完整给出本体状态的编码维数、注意力 mask 和缓存实现。不能仅凭 Gemma 的名称推断整个多模态系统使用怎样的因果注意力或 KV-cache。

### 6.2 已披露的训练超参数

| 项目 | 配置 |
| --- | --- |
| Policy optimizer | AdamW，基础学习率 $2\times10^{-5}$，cosine decay |
| Value/critic optimizer | Adam，基础学习率 $5\times10^{-4}$，cosine decay |
| 折扣因子 | $\gamma=0.9999$ |
| Action horizon | $H=30$ |
| QAM temperature | $\lambda=2$ |
| Target EMA rate | 0.005 |
| DIVL support | 201 atoms，$[-0.1,1.1]$ |
| $\tau_{\mathrm{base}}$ | Offline 0.6；online 0.9 |
| 熵敏感系数 | $\alpha=0.3$ |
| Online replay | Offline : online 约 1:1 |
| 策略发布频率 | 每 50 个 learner steps |
| 在线预算 | 每方法 4 h wall-clock，机群合计约 60 robot-hours 数据 |

按定义计算，$\gamma^{30}\approx0.997$，$\gamma^{1000}\approx0.905$。这些是折扣系数的数值解释，不是论文新增的实验结果。

论文没有完整给出 batch size 的具体数值、训练硬件、随机种子、动作归一化、$\tau_{\min}/\tau_{\max}$、实际去噪步数与推理延迟。原笔记引用的其他 QAM 实验设置，不能直接填成 LWD 的配置。（来源：附录 B2）

## 七、机群系统怎样闭合数据与模型回路？

### 7.1 机器人端执行什么？

实验使用 16 台 Agibot G1 双臂机器人，每台有两条 7-DoF 手臂、平行夹爪，以及头部与双腕的三个 RGB 相机。四台负责补货任务，四个长任务各分配三台。

机器人运行 30 Hz 的关节位置控制，策略每次生成 30 步动作并在执行后重规划，名义上覆盖 1 s。论文未完整写明动作张量维数、绝对或增量表示、归一化和后处理，不能仅根据关节数补出全部动作通道。

这个 1 s 是动作跨度，不是已测得的推理延迟。机群采集与 learner 异步，不等于单台机器人已经实现推理与执行的无缝重叠；论文也没有量化 chunk 执行期间的响应延迟。（来源：§III-A；§V-A）

### 7.2 Episode 上传与版本化 replay

```{figure} images/LWD/lwd-system.png
:alt: Robot edge clients 经对象存储、消息队列和 Coordinator 向多主机 learner 提供版本化 replay，再发布模型给机群
:width: 100%
:align: center

**原论文图 10：分布式训练数据系统**。异步机器人采集通过版本化 snapshot 接入多主机训练，更新后的策略在 episode 边界加载。
```

完整数据流包括：

1. **Edge client 汇总 episode**：在 episode 边界上传 payload 到对象存储，保存元数据并发布消息事件。
2. **Coordinator 提交 snapshot**：消费通知，形成单调递增的版本号，定义 learner 可见的数据视图。
3. **DRB Reader 读取数据**：每个训练节点使用预取进程下载 payload；多主机 SPMD JAX learner 在训练步前通过 barrier 对齐 snapshot。
4. **Learner 更新模型**：从混合 replay 采样，优化 value、critic 与 policy。
5. **向机群发布策略**：通过 pub-sub 分发 checkpoint，机器人在 episode 边界加载新策略。

版本一致并不意味着所有主机拿到相同 minibatch，而是它们基于同一个已提交的数据视图训练。每 50 步发布一次，也不等于所有机器人即时切换；网络传输和 episode 边界都会影响实际生效时间。（来源：附录 D）

### 7.3 可靠性与延迟的证据

| 指标 | 测量结果 | 口径 |
| --- | --- | --- |
| Episode → learner 可采样 | P50 41 s；P99 148 s | 数据进入学习侧的端到端延迟 |
| Model 发布 → actor 加载 | P50 38 s；P99 55 s | 策略分发路径的端到端延迟 |
| 完整数据管线 | 稳态摄入的 1604 个 episodes 均完成管线 | 一次 8 h、16 actors 的系统运行 |

这些系统指标来自同一次 8 小时 profiling run，不能与主实验的 4 小时在线训练预算混用。1604 也是系统管线的 episode 数，不是主结果的性能评测样本数。

系统描述为 at-least-once delivery，借助原子对象上传、持久化消息、失败重试和原子 snapshot 提交保证数据进入管线；这不等于证明 exactly-once 消费。延迟对网络带宽与链路竞争敏感，也不是动作推理时延。（来源：附录 D；表 VI）

## 八、实验结果应该怎样阅读？

### 8.1 八个任务与两类评分

```{figure} images/LWD/lwd-tasks.png
:alt: 四个长任务为调酒、泡功夫茶、榨果汁和装鞋盒，另有四种商店补货任务
:width: 100%
:align: center

**原论文图 3：八个真机任务**。长任务覆盖多阶段工具与容器操作，补货任务强调语言指令、物体与货架布局变化。
```

四个补货任务分别为 Restocking、Correction、Freezer、Open-Cooler，涉及平面货架补货、错放物品纠正、冷柜开门补货和带纸箱处理的开放冷柜补货。

四个长任务为 Gongfu Tea、Fruit Juice、Cocktail、Shoebox，通常持续 3–5 分钟，包含 5–8 个标注子步骤。策略接收“Make Tea”这类高层指令，并没有把人工标注的评分子步骤逐一作为执行指令输入。（来源：§V-A）

| 指标 | 定义 | 解读边界 |
| --- | --- | --- |
| 补货成功率 | 在时限内按正确指令完成任务，二元评分 | 衡量该组任务完整成功 |
| 长任务 step-wise score | 子步骤成功为 1；轻微瑕疵或一次重试后成功为 0.5；多次尝试后失败为 0；取平均 | 衡量步骤完成质量，不能换算为整条 episode 的失败概率 |
| Cycle time | 统计成功与失败尝试，失败在任务 timeout 处截断 | 同时受执行速度、重试与终止规则影响 |

分数由训练过的人类评估员按统一 rubric 给出。论文没有完整列出各任务评测 episodes 数量、随机种子和评分者一致性，因此应把表中结果视为该套协议下的报告值。

### 8.2 主结果与数据预算

| 方法 | Restocking | Correction | Freezer | Open-Cooler | Gongfu Tea | Fruit Juice | Cocktail | Shoebox | 平均分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SFT | 0.70 | 0.88 | 0.83 | 0.95 | 0.64 | 0.66 | 0.70 | 0.70 | 0.76 |
| RECAP | 0.95 | 0.96 | 0.94 | 0.95 | 0.84 | 0.82 | 0.71 | 0.70 | 0.85 |
| HG-DAgger | 1.00 | 0.92 | 0.92 | 1.00 | 0.60 | 0.66 | 0.76 | 0.90 | 0.85 |
| LWD Offline | 1.00 | 1.00 | 0.92 | 0.95 | 0.72 | 0.74 | 0.83 | 0.86 | 0.88 |
| LWD Online | 1.00 | 1.00 | 0.97 | 0.98 | 0.89 | 0.90 | 0.93 | 0.92 | 0.95 |

以上为原论文表 I。LWD Online 取得最高总平均分和四个长任务的最高分，但并非每个任务都严格最好，例如 Open-Cooler 的 HG-DAgger 为 1.00，LWD 为 0.98。

相对于 SFT，平均分由 0.76 升至 0.95，增加 0.19；长任务平均分由约 0.68 升至 0.91。Gongfu Tea 的 0.89 是步骤评分，不能解释成“每十次泡茶仍有一次失败”。

```{figure} images/LWD/lwd-results.png
:alt: LWD 与 SFT 在各任务的分数比较，以及四个长任务的执行耗时比较
:width: 100%
:align: center

**原论文图 5：分数与执行耗时**。LWD 提高任务得分，同时相对 SFT reference 平均缩短 23.75 s 的长任务 cycle time。
```

### 8.3 Baseline 是否使用了相同数据？

| 方法 | 数据与训练方式 | 比较时需要注意 |
| --- | --- | --- |
| SFT | 336.6 h 专家示范，flow matching | 数据量与来源少于 LWD，不能将全部差距归因于 RL 目标 |
| RECAP | 同一 reference 起点；两轮自主 rollout，每轮约 60 robot-hours，加上示范训练 advantage-conditioned policy | 论文将其适配到八任务设置；不是“只会单任务”的对照 |
| HG-DAgger | 同一 reference；约 60 robot-hours 在线 buffer 与离线示范 | 附录描述自主 rollout 和人工纠正汇入 buffer，主文强调成功 rollout；筛选细节未完全展开 |
| LWD Offline | 652.5 h 的示范、成功/失败 rollout 与 play | 已使用更丰富的离线来源 |
| LWD Online | LWD Offline 初始化，再加入约 60 robot-hours 在线数据 | 同时改变训练机制、初始化与使用的数据来源 |

后训练方法使用相同 policy optimizer 与学习率 schedule，在线比较给出相同时间预算；这些控制提高了可比性，但数据组成并未完全匹配。结果支持整套 LWD 配方有效，不能单靠这张表证明差距完全来自算法而与数据无关。（来源：§V-A；附录 C1）

### 8.4 DIVL 与自适应分位数的消融

| 价值学习方式 | 补货 Offline | 补货 Online | 长任务 Offline | 长任务 Online |
| --- | ---: | ---: | ---: | ---: |
| Scalar expectile regression | 0.96 | 0.97 | 0.72 | 0.78 |
| DIVL | 0.97 | 0.99 | 0.79 | 0.91 |

原论文表 II 保持其他组件不变，比较 DIVL 与标量 expectile 回归。长任务 online 得分由 0.78 到 0.91，绝对增加 0.13，约为 16.7% 的相对提升；两个口径不能混用。Expectile 本来就偏向较高价值，不宜把这个 baseline 简化为普通均值回归。

附录表 V 还给出了**离线和在线的完整逐任务消融结果**。DIVL 在总平均与长任务上更好，但并非每个单项都胜出，例如离线 Freezer 与 Open-Cooler 的 expectile 分别为 1.00，DIVL 为 0.92 与 0.95。

自适应 $\tau$ 的离线消融将常数设为 adaptive run 中观测到的平均值 0.52，其他组件保持不变。平均分由常数版本的 0.84 升至自适应版本的 0.88；部分任务仍由常数版本更高，所以证据支持平均收益，不能写成逐任务一致提升。（来源：表 III）

LWD 没有单独隔离 QAM 的贡献、离线多步 TD 的贡献，也没有比较 4、8、16 台机器人形成机群规模曲线。16 台的运行结果展示了这套规模下的可行性，尚不足以建立 scaling law。

### 8.5 价值函数是否学会了任务进度？

```{figure} images/LWD/lwd-value-progress.png
:alt: 成功和失败的 Gongfu Tea episode 中，价值分位数随任务步骤变化的对比
:width: 100%
:align: center

**原论文图 6：价值随执行过程变化**。成功轨迹总体向更高价值推进，失败轨迹在停止取得进展后保持较低价值；这是代表性案例的定性诊断。
```

图 6 展示分位数曲线；附录图 9 进一步展示分布形状，成功例子的 mode 约从 0.4 移到 1.0，失败例子约从 0.5 移到 0.6 后停滞。两张图中的 quantile 与 mode 是不同统计量。

**我的理解**：这些例子与价值学习捕捉任务进展的解释一致，但不能据此认定获得了通用、准确的进度检测器。论文没有单独报告进度预测误差、失败定位准确率或跨任务校准。

## 九、与相关工作及现有笔记的联系

LWD 把几个已有方向组合进同一个部署循环：IQL 提供数据内非对称价值选择的思路，QAM 提供 flow policy 的价值梯度提取机制，机群基础设施则支持在线经验进入共享训练。

它与 [SOP 论文解读](SOP-paper.md) 的联系尤其直接：SOP 关注机群部署与异步更新的系统底座，LWD 在这种部署方式上具体实现离线到在线 RL。两者关注点相互衔接，不能把系统能力与学习算法混成同一个贡献。（来源：§II-C）

RECAP 和 HG-DAgger 是论文实际运行的对照；SERL/HIL-SERL、DSRL、RL100 等属于相关工作。LWD 的实验没有覆盖所有这些方法，也没有证明对所有 specialist RL 或 latent steering 路线的普遍优势。

## 十、我的总结与仍需验证的问题

LWD 的核心是把部署结果接入训练：DIVL 在状态条件下保留 replay 动作价值的分布，以自适应分位数构造 bootstrap；QAM 将 critic 的动作梯度变成 flow 模型的局部监督；版本化 replay 与策略分发把这一过程扩展到共享同一策略的机器人机群。

论文在八任务上给出清楚的总体收益，长任务改进更明显。但这仍是一个平台、一组任务和有限时间预算下的结果。新任务与新本体泛化、长期更新后的遗忘、人工干预成本，以及更大机群的更新稳定性，都需要进一步证据。

作者也明确指出，当前框架没有显式建模执行安全，长任务仍依赖简短高层指令，后续需要更好的任务分解、恢复与更新机制。原笔记中关于更短动作块、更强感知和减少干预的建议可以作为研究假设，不能当成已被实验证实的失败原因。

对我而言，这篇论文最值得保留的认识是：部署数据的价值不仅在于增加成功示范，还在于用任务结果组织成功、失败和纠正经验；而要让这种学习持续发生，算法目标与数据系统必须一起设计。
