# Genesis 实践记录：双视角 Flow BC 的数据引擎与自主抓取评测

这篇记录整理我在 Genesis 中完成的一次视觉 Behavior Cloning（BC，行为克隆）实验：用仿真数据引擎自动采集抓取演示，训练一个 4.39M 参数的双视角 Flow BC policy，再在本地 MacBook Air 上完成 100 回合自主评测。**同一个 `best.pt` 在这次标准仿真分布下成功完成 96 个回合，评测不调用专家恢复控制器，也没有人工干预。**

之前，这个 pick-and-lift 任务上的视觉 BC policy 在无恢复评测中大约只有 60% 的成功率。这是当时的数据分布、视觉表示和动作建模方式共同得到的真实结果。对于一个只要求机械臂抓住方块并稳定抬升 8 cm 的简化任务而言，60% 意味着策略仍会频繁错过抓取、在接近阶段产生不稳定轨迹，或者无法连续完成抓取与抬升。

因此，这一阶段我重新检查了整条 BC pipeline：数据是否覆盖了足够多的初始状态和成功路径，两个差异明显的相机是否应该共享视觉参数，以及确定性动作回归是否适合包含多种专家运动方式的数据集。

```{figure} https://pic1.imgdb.cn/i/034aRZe7KQckrdD8qrS4wy.png
:target: https://pic1.imgdb.cn/i/034aRZe7KQckrdD8qrS4wy.png
:alt: Genesis 实验总览，连接仿真数据引擎、双视角 Flow BC 与 96 次成功的自主评测
:width: 100%
:align: center

**图 1：从仿真演示到自主抓取。** 左侧是空间、外观与轨迹分布的设计，中间是双视角条件动作生成，右侧展示标准仿真评测。96/100 是学习策略的评测结果，99.12% 是采集阶段特权专家的成功率。
```

## 任务设置与策略输入

任务运行在 Genesis 中，控制对象是 Panda 机械臂和一个边长 4 cm 的方块。目标是抓取方块并稳定抬升，暂不包含目标位置放置、释放或回到初始位姿。

| 项目 | 本次配置 |
| --- | --- |
| 成功条件 | 方块相对初始位置抬升至少 8 cm，并连续保持 5 个 control steps |
| 策略控制频率 | 50 Hz，每步对应 20 ms 仿真时间 |
| 物理仿真频率 | 500 Hz，每个 control step 包含 10 个 physics substeps |
| 视觉输入 | 固定外部相机与腕部相机的两路 RGB 图像 |
| 状态输入 | 当前 proprioception，以及短期状态与动作历史 |

采集演示时，privileged recovery controller 可以读取物体位置、接触状态和仿真受力等特权信息。**这些信息用于生成 demonstration 和判断采集结果，不作为 BC policy 的输入。** 自主评测时，策略只能使用上述视觉和机器人自身状态。

## 仿真数据引擎与演示分布

### 用反馈控制器生成演示

数据采集由 privileged recovery controller 自动完成 pick-and-lift。控制器根据反馈，在全局接近、局部预抓取、最终下降、接触确认、probe lift、正式抬升和失败恢复之间切换。

Cartesian reference 由带速度、加速度和 jerk 限制的 servo 在线生成。如果出现 missed grasp、weak grasp 或 slip，控制器会调整局部目标并重新规划，避免继续播放已经失效的开环轨迹。

这套数据引擎对我的意义，是能用先验控制算法自动产生大量演示，减少逐条使用 SpaceMouse 遥操作采集轨迹的工作量。后续训练得到的 BC policy 则需要从有限观测中学习完成任务。

### 显式设计位置、颜色与运动方式

这次采集的重点是 demonstration distribution。方块初始位置在中心点周围 X、Y 各 ±5 cm 的范围内变化，并使用 Latin hypercube sampling（LHS，拉丁超立方采样）提高二维工作区的覆盖均匀性。外观包含 8 种颜色，每种严格采集 128 个回合。

专家运动包含 6 种 trajectory profiles：staged、funnel、spline、pause-and-verify、center-seeking 和 slip-recovery。它们提供不同但有效的任务完成方式。

```{note}
LHS 是一种分层采样方法。以采集 1,024 个初始位置为例，分别把 X、Y 的取值范围划分成 1,024 个等概率区间，让每个维度的每个区间恰好被采样一次，再随机组合两个维度。它约束的是各维度的覆盖，不意味着枚举所有二维网格组合。
```

颜色和 trajectory profile 分别保持边际均衡，再通过独立 permutation 分配给各个回合，以减少“某种颜色总是对应某种运动路径”的伪相关。颜色 ID、profile ID、物体真值位置和接触阶段会写入 dataset metadata，供审计与分层分析使用，但不会进入 policy observation。

```{figure} https://pic1.imgdb.cn/i/034aRZh4yyYRdLMowwS3Gi.png
:target: https://pic1.imgdb.cn/i/034aRZh4yyYRdLMowwS3Gi.png
:alt: 仿真数据引擎的位置采样、八种颜色、六种专家轨迹与四进程采集流程
:width: 100%
:align: center

**图 2：数据分布与采集流程。** 每个 worker 使用独立的 Genesis scene，按预先分配的位置、颜色、轨迹与 seed 生成演示，最后汇总 episode shards 和 manifest。
```

### 采集规模与训练集划分

采集使用 4 个 spawn workers。每个 worker 创建独立的 scalar Genesis scene，进程之间不共享可变仿真状态。回合数据独立保存，通过临时文件和原子替换落盘，主进程最后汇总生成 manifest。

| 统计项 | 数值 |
| --- | ---: |
| 采集回合数 | 1,024 |
| 原始 transitions | 286,444 |
| 图像与状态数据量 | 约 2.46 GiB |
| 采集耗时 | 约 2.02 小时 |
| 特权专家成功回合 | 1,015 / 1,024 |
| Demonstration success rate | 99.12% |
| 筛选后进入训练与验证划分的样本 | 260,301 |

训练阶段按照 episode success 和本次任务的初始化方式选择样本。筛选后得到的 260,301 个 transition-level samples 是 train/validation split 的总样本池。切分时优先按 visual group 划分，避免把同一外观组分到训练与验证两侧，也避免直接随机拆散所有 transitions。

这里的“泛化”主要指增加训练任务分布内部的位置、外观与成功路径覆盖，并减少数据中的伪相关。后面的 100 回合评测使用标准外观和新的 simulator seeds，主要检验初始位置与仿真随机性下的分布内泛化；它不能直接说明未知物体、全新背景或 sim-to-real 的表现。

## 双视角视觉表示与轻量网络

固定相机提供机械臂、工作区与方块之间的全局关系，腕部相机更敏感于局部接近、遮挡和抓取阶段的视觉变化。共享 CNN backbone 可以减少参数量，但这两个视角的成像分布差异很大；从零开始训练时，同一套低层卷积参数需要同时适配两种视觉空间。

因此，我最终使用了**结构相同、参数相互独立的两个 Lightweight CNN backbones**。每个 backbone 的 channel width 依次为 48、64、96、128 和 160，由 depthwise-separable residual blocks 构成。每路 128 × 128 图像经过 CNN 后，adaptive pooling 得到 4 × 4 feature grid，再投影为 16 个 256 维 visual tokens；两路相机共产生 32 个 visual tokens。

状态侧使用当前 8 维 proprioception、四步历史状态和四步历史动作，共形成 56 维 context，再由两层 MLP 编码成一个 state token。模型把 1 个 CLS token、1 个 state token 和 32 个 visual tokens 组成 34-token sequence，输入 2 层、4 个 attention heads、token dimension 为 256 的 Transformer Encoder。CLS 输出最后映射成 384 维 observation condition。

```{figure} https://pic1.imgdb.cn/i/034aRZlHq0i6t6QdM8tkuV.png
:target: https://pic1.imgdb.cn/i/034aRZlHq0i6t6QdM8tkuV.png
:alt: 两个独立 CNN 将双相机图像编码为视觉 tokens，与状态历史融合后条件化动作流网络
:width: 100%
:align: center

**图 3：双视角 Flow BC 网络。** 两个相机分别学习低层视觉表征，由轻量 Transformer 融合空间 tokens 与状态历史，再生成未来动作块。
```

动作流网络接收 8 × 4 的 noisy action chunk、64 维 Fourier time embedding 和 observation condition；网络宽度为 384，包含 4 个受条件调制的 residual blocks，输出整个动作块的速度场。

这套结构没有使用预训练 ResNet 或大型视觉 Transformer，完整 policy 包含 4.3928M 个可训练参数。我的设计意图是让两个相机保留各自的视觉特征，再通过 token fusion 建立跨视角关联；相比只拼接两个全局 CNN feature，4 × 4 网格也保留了更多空间结构。

## 从动作回归到 Conditional Flow Matching

### 为什么对未来动作块建模

之前的 BC 使用 deterministic regression 预测动作。当演示只有一种标准路径时，这通常是合理的；但当前数据集刻意包含六种 trajectory profiles，相似观测附近可能有多种正确的后续路径。例如，机械臂可以先垂直撤回再横向对齐，也可以沿 funnel trajectory 连续接近。

如果直接做单点 mean-squared error 回归，模型可能学到多个 mode 的平均动作。这个平均值在数值上接近 demonstration，却不一定属于任何一条可执行的专家轨迹。为此，我将 policy head 改成 conditional flow matching，让模型学习**给定当前观测时，未来动作块的条件分布**。这是选择生成式动作头的动机，实际闭环收益仍需要通过 rollout 检验。

### 动作块、插值路径与训练目标

以 $k$ 表示环境控制步，从同一个回合提取未来 8 步专家动作：

$$
x_1 = a_{k:k+7}, \qquad x_1 \in \mathbb{R}^{8\times4}.
$$

动作块不会跨越回合边界。回合尾部不足 8 步时，使用最后一个动作完成 tensor padding，同时生成 valid mask；padding 部分不参与 loss。

以 $\tau$ 表示 Flow Matching 的插值时间，区分它与环境控制步。采样同形状的 Gaussian noise 与插值时间：

$$
x_0 \sim \mathcal{N}(0,I), \qquad \tau \sim \mathcal{U}(0,1).
$$

在噪声与专家动作块之间构造线性路径：

$$
x_\tau = (1-\tau)x_0 + \tau x_1.
$$

对应的目标速度为：

$$
u_\tau = \frac{\mathrm{d}x_\tau}{\mathrm{d}\tau} = x_1-x_0.
$$

视觉与状态 encoder 产生 observation condition $c=f(o)$。Flow network 学习：

$$
v_\theta(x_\tau,\tau,c) \approx u_\tau.
$$

训练目标是在有效动作位置上计算的 masked mean-squared error。整个过程仍属于有监督的 Behavior Cloning：每个 gradient step 只在随机的 $x_\tau$ 与 $\tau$ 上回归条件速度场，不需要运行 ODE solver，也不需要计算 action likelihood。

### 训练配置与 checkpoint 选择

| 项目 | 本次配置 |
| --- | --- |
| Gradient steps | 30,000 |
| Batch size | 128 |
| Optimizer | AdamW |
| Initial learning rate | `1e-4` |
| Warmup | 500 steps |
| Learning rate schedule | Cosine decay，衰减到 `1e-5` |
| EMA decay | 0.999 |
| Validation 间隔 | 每 250 steps |
| `best.pt` 保存依据 | EMA validation loss |
| 30k step 时 validation loss | 0.0161 |

较低的 Flow Matching validation loss，表示模型在 held-out samples 上能更准确地预测目标速度场。**它本身不能直接推出更高的闭环成功率。** 策略能否真正完成抓取，仍要通过 simulator rollout 验证。

## 动作块采样与滚动闭环执行

### 预测 8 步，执行 4 步

部署时，策略从 Gaussian action chunk 出发，在当前 observation condition 下求解：

$$
\frac{\mathrm{d}x}{\mathrm{d}\tau}=v_\theta(x,\tau,c),
\qquad \tau:0\rightarrow1.
$$

当前配置使用 4-step Heun integrator。每次采样生成长度为 8 的 future action chunk，环境只执行前 4 步，然后重新读取两路相机和 proprioception，再生成下一段动作。

| 参数 | 配置及含义 |
| --- | --- |
| Action horizon | 8 个控制步，预测未来 160 ms |
| Execution horizon | 4 个控制步，执行前 80 ms 后重新观察 |
| Sampling steps | 4 个 Heun 积分步，覆盖插值时间 $\tau\in[0,1]$ |

表中的 160 ms 和 80 ms 是按 50 Hz 控制频率换算的**仿真时间跨度**，不代表模型实际计算耗时。Heun 的积分步数也不等于机器人控制步数。

8 步联合生成用于提高局部轨迹一致性，执行 4 步后重新规划则保留视觉闭环纠偏能力。同一个回合内复用同一份 initial noise latent，目的是减少连续 replanning 时局部运动风格的频繁切换；每个新回合都会重置 latent。

### Heun 的预测与校正

Heun 是一种二阶 ODE 数值积分方法，每个积分步包含两次速度场预测。先用当前速度给出 Euler 预测：

$$
\tilde{x}_{\tau+\Delta\tau}
= x_\tau + \Delta\tau\,v_\theta(x_\tau,\tau,c).
$$

然后在预测终点再次估计速度，使用两者的平均值完成校正：

$$
\begin{aligned}
x_{\tau+\Delta\tau}
&= x_\tau + \frac{\Delta\tau}{2}\Bigl[
  v_\theta(x_\tau,\tau,c) \\
&\qquad + v_\theta(\tilde{x}_{\tau+\Delta\tau},\tau+\Delta\tau,c)
\Bigr].
\end{aligned}
$$

相比每步只使用当前点速度的 Euler 方法，Heun 同时利用当前点与预测终点的速度，代价是每个积分步调用速度场网络两次。这里的 4-step Heun 对应 8 次速度场网络求值；这不意味着要重复编码图像 8 次，观测条件 $c$ 在同一次动作采样中保持固定。

当前策略每次只执行未来 80 ms 的动作，之后便重新观察。它更接近 receding-horizon control 下的生成式动作块策略，还不是完整的长时域轨迹规划。

## MacBook Air 上的 100 回合自主评测

训练完成后，我先在 40 个独立 seeds 上评测 `best.pt`，成功 38 次，即 95%。随后，我在本地 MacBook Air 上把评测扩大到 100 个回合，使用从 50000 开始的独立 seeds，并通过两个 spawn workers 并行执行。同一个 `best.pt` 最终完成了 96 次任务。

| 评测轮次 | 成功回合 | 成功率 |
| --- | ---: | ---: |
| 第一轮 | 38 / 40 | 95% |
| 扩大评测 | 96 / 100 | 96% |

扩大评测的 seed 范围为 50000–50099。**评测不调用 privileged recovery controller，没有专家恢复接管，也没有 human intervention。** 每个 worker 分别初始化 policy 和 Genesis environment，每个回合开始前重置 action queue、状态历史与 episode latent。两个 worker 用于提高评测吞吐，不共享单回合状态，也不改变环境动力学。

```{figure} https://pic1.imgdb.cn/i/034aRZnSwggcsbdTh8MZ2t.png
:target: https://pic1.imgdb.cn/i/034aRZnSwggcsbdTh8MZ2t.png
:alt: 标准仿真评测中重置、接近、抓取抬升及稳定抬升的双视角画面，100 回合成功 96 次
:width: 100%
:align: center

**图 4：自主评测的关键阶段。** 示例画面对应 reset、approach、grasp and lift、stable lift。图中的平均回合长度为 318.62 个控制步，另附这批评测记录的 bootstrap 95% 区间 [92%, 99%]；这些统计仅对应当前评测设置。
```

```{figure} https://pic1.imgdb.cn/i/034aRZqT2BCdnGOxzJMZoi.png
:target: https://pic1.imgdb.cn/i/034aRZqT2BCdnGOxzJMZoi.png
:alt: 一个成功回合的十二组连续抽帧，每组同时显示固定相机与腕部相机画面
:width: 100%
:align: center

**图 5：一个成功回合的连续抽帧。** 从初始状态到接近、抓取与稳定抬升，每组画面同时展示固定相机和腕部相机。单个成功示例用于说明动作过程，整体成功率由 100 个回合的统计给出。
```

## 这次实验能够说明什么

这次实验没有提出新的 Flow Matching algorithm。它验证的是：一条经过演示分布设计的 simulation-to-BC pipeline，可以用 1,024 个仿真回合采集的数据，训练出一个 4.39M 参数的双视角生成式 policy，并在这次 100 回合的标准仿真评测中达到 96% autonomous success。

相较于之前大约 60% 的结果，我让几个环节彼此匹配：数据引擎提供空间、颜色与轨迹风格覆盖；独立 CNN 保留两个相机各自的视觉特征；Flow Matching 对多种合理的未来动作块建模；8 步预测与 4 步执行在局部动作一致性和闭环纠偏之间取得平衡。对我而言，能自动生成 1,024 条由先验控制算法驱动的轨迹，也是这条流程很实际的价值。

这些改动是一起发生的。目前的结果不足以分别量化数据分布、独立 CNN、Flow Matching、Heun 或 latent 复用各自贡献了多少提升；上面的分析是设计动机与整体实验观察，不能当作逐项消融结论。

**96% 是当前标准仿真分布下的成功率。** 本次评测的随机性主要来自 simulator seed 与方块初始位置，尚未验证未知物体几何、显著背景变化、相机外参偏移、强光照变化或动力学误差下是否有同等鲁棒性，也没有得到真实机器人上的成功率。

当前任务只要求抓取并稳定抬升，不包含 target placement、release 或 return-home。我更愿意把它看作后续 BC-to-HIL-RL pipeline 的一个稳定起点，下一步仍需要扩大任务范围，并单独检查失败模式和泛化能力。
