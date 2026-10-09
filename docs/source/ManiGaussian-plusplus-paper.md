# ManiGaussian++ 论文解读：分层高斯模型与双臂协作

论文原题：ManiGaussian++: General Robotic Bimanual Manipulation with Hierarchical Gaussian World Model

- 作者：Tengbo Yu、Guanxing Lu、Zaijia Yang（共同一作），Haoyuan Deng、Season Si Chen、Jiwen Lu、Wenbo Ding、Guoqiang Hu、Yansong Tang、Ziwei Wang
- 团队：清华大学、海南大学、南洋理工大学（按所附 arXiv v1 的作者单位）
- 发表：**IROS 2025**，[会议官方程序](https://ras.papercept.net/conferences/conferences/IROS25/program/IROS25_ContentListWeb_2.html)收录于 Grasping & Manipulation 3，编号 WeCT20.3
- 论文：[arXiv:2506.19842v1](https://arxiv.org/abs/2506.19842v1)，2025 年 6 月 24 日
- 代码：[yangzaijia/ManiGaussian_Bimanual](https://github.com/yangzaijia/ManiGaussian_Bimanual)

[ManiGaussian](ManiGaussian-paper.md) 用动作后的三维场景重建辅助学习单臂操作表征。ManiGaussian++ 将问题推进到双臂：**当一只手稳定物体，另一只手执行操作时，怎样表示两只手及物体之间的相互影响？**

作者增加了两个设计：给 Gaussian 加入任务相关实例信息，区分场景中需要交互的部分；再用 leader–follower 世界模型，先预测稳定臂的影响，再在此基础上预测操作臂的影响。十项双臂仿真任务平均成功率达到 **35.6%**，相对 PerAct² 的 15.4% 提高 **20.2 个百分点**。

这篇已经被 IROS 2025 接收。下文的方法与数字采用所附 arXiv v1；其中真机正文和图 4 的平均成功率为 **62.22%**，摘要与结论写作 60%，阅读时需要保留这一口径差异。

```{figure} https://pic1.imgdb.cn/i/034cpZD4Ibm9eSBaZZGfPd.png
:target: https://pic1.imgdb.cn/i/034cpZD4Ibm9eSBaZZGfPd.png
:alt: 黄色方块交接任务中，PerAct² 未能完成双臂协作，ManiGaussian++ 通过双臂 Gaussian 表征完成交接
:width: 100%
:align: center

**原论文图 1：双臂任务需要理解彼此的动作影响**。一只手的目标位置取决于另一只手及被操作物体的状态；图中展示的是一个具体任务案例。
```

## 一、相比 ManiGaussian，新增了什么？

把单臂策略的动作输出扩展到两只手，可以得到双臂基线，但未必能学好两只手的配合。论文中的 ManiGaussian 基线就采用这种适配方式，十任务平均成功率为 18.8%。

| 比较项 | ManiGaussian | ManiGaussian++ |
| --- | --- | --- |
| 操作对象 | 单臂多任务操作 | 双臂多任务协作 |
| Gaussian 附加信息 | 预训练视觉模型提供的语义特征 | 任务相关的实例 logits，以 GroundedSAM 掩码监督 |
| 动态预测 | 根据动作预测一次场景形变 | 按稳定臂、操作臂分层预测形变 |
| 世界模型的用途 | 通过未来重建改善共享表征 | 在共享表征中学习双臂与物体的交互 |
| 本文实验 | 前作主要为单臂 RLBench | 十项双臂仿真任务与九项真机任务 |

这里的重点是训练监督如何组织。Gaussian 世界模型帮助策略学到与双臂交互有关的视觉表征，动作分支仍然从当前观测和语言指令预测下一步动作。论文没有引入基于未来渲染的在线候选动作搜索或 MPC；[公开实现](https://github.com/yangzaijia/ManiGaussian_Bimanual/blob/main/agents/manigaussian_bc2/qattention_manigaussian_bc_agent.py)也将渲染辅助损失放在训练开关下。

## 二、输入、表征与动作输出

### 2.1 从 RGB-D 构建共享三维表征

观测包括 RGB、深度，以及夹爪状态和当前时间等低维信息。图像结合相机标定投影到三维体素，再经过稀疏 3D 卷积得到体积表征 $v_t$。这套表征同时提供给 Gaussian 回归器和动作策略。

动作分支使用基于 PerceiverIO 的多模态 Transformer，结合语言指令输出左右两臂动作。Gaussian 分支则负责重建当前场景、预测下一时刻场景，并将辅助损失反传到共享表征。（论文 §III-A、§III-E）

```{figure} https://pic1.imgdb.cn/i/034cpZIogXfFdqevTxmgl3.png
:target: https://pic1.imgdb.cn/i/034cpZIogXfFdqevTxmgl3.png
:alt: ManiGaussian++ 从 RGB-D 和本体信息编码体素表征，通过任务 Gaussian 与 leader-follower 形变学习双臂交互，策略分支输出双臂动作
:width: 100%
:align: center

**原论文图 2：任务相关 Gaussian 与分层世界模型**。上方分支提供当前重建、实例监督和未来预测损失；下方策略分支学习双臂动作。
```

### 2.2 每只手预测一个末端目标

每臂的动作包含四部分：

- **平移**：在 $100^3$ 体素网格上分类，选择末端目标位置。
- **旋转**：三个轴分别按 $5^\circ$ 离散，每轴 72 个类别。
- **夹爪**：预测开合状态。
- **碰撞设置**：预测供运动规划器使用的碰撞避让开关。

这些输出定义下一步末端目标，由运动规划器执行；它们并非电机力矩，也不代表高频连续控制命令。两臂动作都由行为克隆监督，但世界模型中的条件依赖专门考虑了两臂之间的相互影响。

## 三、让 Gaussian 带上任务相关实例信息

### 3.1 从几何基元到任务相关表示

第 $i$ 个 Gaussian 除了位置、颜色、旋转、尺度和不透明度，还带有三维实例 logit 向量。省略时间下标，可写为：

$$
\theta_i=(\mu_i,c_i,r_i,s_i,\sigma_i,l_i).
$$

其中 $l_i\in\mathbb{R}^{3}$ 是未归一化的分数。论文用它区分机械臂和目标物体等任务相关实例，使 Gaussian 表征包含“哪些部分与这次操作有关”的信息。（论文 §III-C）

实例 logits 与颜色一样，通过 alpha blending 投影到像素。省略时间下标，将像素 $p$ 上第 $i$ 个 Gaussian 的合成权重写为 $w_i(p)$，则：

$$
\begin{aligned}
w_i(p)&=\alpha_i(p)\prod_{j<i}\bigl(1-\alpha_j(p)\bigr),\\
L(p)&=\sum_i w_i(p)\,l_i,\\
B(p)&=\operatorname{softmax}\bigl(L(p)\bigr).
\end{aligned}
$$

$B(p)$ 才是实例类别概率，用于后续交叉熵监督。前作的语义特征蒸馏在这里变成了更直接的任务实例监督。

### 3.2 掩码来自 GroundedSAM 伪标签

作者从人类语言指令中提取关键词，提示 GroundedSAM 生成机械臂与目标物体的实例掩码。这些掩码作为任务损失的监督目标。

因此，这部分依赖已有视觉模型产生的伪标签，效果受目标检测、分割和任务关键词影响。论文没有完整说明三维 logits 的所有类别映射，也没有清楚给出跨任务的角色自动分配及交换规则，不能仅凭实例监督就断言模型已经自主解决了全部双臂角色分配问题。

## 四、先稳定、再操作的分层世界模型

### 4.1 两层预测建立动作之间的条件依赖

作者将双臂角色记为 stabilizing arm（稳定臂）与 acting arm（操作臂）。例如，一只手固定或支撑目标，另一只手完成交接、推动等动作。角色描述的是任务中的作用，不应直接等同于固定的左臂或右臂。

Leader 根据稳定臂动作预测中间场景，Follower 再结合该中间场景与双臂动作预测最终未来场景。为便于阅读，以下统一用 $s/a$ 表示两种角色，以 $\widetilde\theta_{t+1}$ 表示中间 Gaussian 状态：

$$
\begin{aligned}
v_t&=f_\phi(o_t),\\
\theta_t&=g_\phi(v_t),\\
\widetilde\theta_{t+1}
&=q_{s,\phi}(\theta_t,a_{s,t},v_t),\\
\theta_{t+1}
&=q_{a,\phi}(\widetilde\theta_{t+1},a_{s,t},a_{a,t},v_t).
\end{aligned}
$$

最后将 $\theta_{t+1}$ 渲染为未来图像，与示范中真实执行双臂动作后的观测比较。这是对论文 §III-D 的流程重写；核心依赖是 Follower 在 Leader 预测的基础上继续建模。

**这种先后顺序属于世界模型内部的计算。** 它不要求机器人执行时必须先停住一只手、等另一只手完成后再动，也不意味着策略输出被拆成两个完全独立的控制器。

### 4.2 形变仍以位置与旋转增量表达

论文保持 Gaussian 的颜色、尺度、不透明度与实例 logits 不变，将两种角色的影响合成为位置和朝向参数的增量：

$$
\begin{aligned}
\mu_{i,t+1}&=\mu_{i,t}+\Delta\mu_{s,i,t}+\Delta\mu_{a,i,t},\\
r_{i,t+1}&=r_{i,t}+\Delta r_{s,i,t}+\Delta r_{a,i,t}.
\end{aligned}
$$

这里沿用论文的参数更新记法，不能将第二式理解为直接相加旋转矩阵。虽然正文使用了刚体运动和物理后果的语言，实际给出的机制仍是学习 Gaussian 形变，没有显式求解接触力、质量或摩擦约束。

我的理解是，分层结构给表征学习加入了一个与双臂协作有关的假设：第二只手造成的变化应依赖第一只手已经造成的变化。这个结构比一次性回归整个场景的未来更贴近任务中的相互依赖，但其物理一致性仍需要实验验证。

### 4.3 四个训练目标共同塑造表征

| 损失 | 监督对象 | 作用 |
| --- | --- | --- |
| $\mathcal L_{\mathrm{BC}}$ | 左右两臂示范动作 | 学习任务动作 |
| $\mathcal L_{\mathrm{Recon}}$ | 当前时刻的多视角 RGB | 保留当前场景几何与外观 |
| $\mathcal L_{\mathrm{Task}}$ | GroundedSAM 实例掩码 | 保留任务相关实例信息 |
| $\mathcal L_{\mathrm{Pred}}$ | 动作后的未来 RGB | 学习动作引起的场景变化 |

当前重建和未来预测采用像素均方误差，任务实例采用交叉熵，双臂行为克隆采用动作分类交叉熵。总目标为：

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{BC}}\\
&+\lambda_{\mathrm{Recon}}\mathcal L_{\mathrm{Recon}}\\
&+\lambda_{\mathrm{Task}}\mathcal L_{\mathrm{Task}}\\
&+\lambda_{\mathrm{Pred}}\mathcal L_{\mathrm{Pred}}.
\end{aligned}
$$

未来图像是训练时的监督目标，部署时并不预先可见。所附八页版本没有给出这些权重的完整数值，复现时需要结合代码配置，不能直接沿用前作的超参数。

```{figure} https://pic1.imgdb.cn/i/034cpZM0BIkEGhwYKzpYoc.png
:target: https://pic1.imgdb.cn/i/034cpZM0BIkEGhwYKzpYoc.png
:alt: 当前和未来场景的新视角 Gaussian 重建，对照真实图像并显示若干示例的 PSNR
:width: 100%
:align: center

**原论文图 3：当前与未来的新视角重建**。图注明确说明，为展示重建效果关闭了行为克隆损失；这些 PSNR 是所示案例的结果，不能当作完整策略的平均重建指标。
```

## 五、仿真结果：收益集中在哪些任务？

### 5.1 十项双臂任务的完整比较

作者在 RLBench² 的十项双臂任务上评估，每项使用 **100 条 Oracle 示范**。仿真观测来自六个 **256×256 RGB-D** 相机；多视角图像还用于 Gaussian 重建监督。评估表说明使用 25 个 episodes，每个 episode 最多允许 25 步。（论文 §IV-A、§IV-B）

| 任务 | PerAct² | ManiGaussian | ManiGaussian++ |
| --- | --- | --- | --- |
| Pick laptop | 12% | 8% | **12%** |
| Straighten rope | 24% | 28% | **40%** |
| Lift tray | 1% | 4% | **8%** |
| Push box | 6% | 24% | **48%** |
| Handover easy | **41%** | 36% | 40% |
| Put in fridge | 3% | 4% | **28%** |
| Press buttons | 47% | 36% | **48%** |
| Handover item | 11% | 12% | **20%** |
| Sweep to dustpan | 0% | 24% | **92%** |
| Take out tray | 9% | 12% | **16%** |
| **平均** | **15.4%** | **18.8%** | **35.6%** |

数据来自原论文表 I。完整方法相对 PerAct² 提高 **20.2 个百分点**，相对适配后的 ManiGaussian 提高 **16.8 个百分点**。论文摘要所说的 “20.2% improvement” 对应前一个绝对差值。

提升最明显的是 Sweep to dustpan：相对 ManiGaussian 从 24% 上升到 92%，适合观察两只手如何围绕同一个操作目标配合。另一方面，Lift tray 仍只有 8%，Take out tray 为 16%；Handover easy 还比 PerAct² 低 1 个百分点。因此，平均优势不代表所有任务均已解决。

这里的 35.6% 与前作单臂任务的 44.8% 来自不同任务和数据设置，不能据此判断新版本性能下降。论文也没有提供完整的随机种子与置信区间信息，小幅差异不宜过度解读。

### 5.2 消融只覆盖三项代表任务

原论文表 II 逐步加入 Gaussian 重建、任务实例监督和分层预测，展示三项任务的结果：

| 配置 | Sweep | Handover item | Push box | 三任务平均 |
| --- | --- | --- | --- | --- |
| PerAct² 基线 | 0% | 11% | 6% | 5.67% |
| 加入 Gaussian 表征 | 24% | 12% | 24% | 20.00% |
| 再加入任务实例监督 | 32% | 16% | 32% | 26.67% |
| 再加入分层世界模型 | **92%** | **20%** | **48%** | **60.00%** |

沿这条消融路径，任务实例监督的增量为 **6.67 个百分点**，分层模型的增量为 **33.33 个百分点**。正文对任务实例的“约 21%”描述使用了最初 5.67% 的基线，属于累计差值，不能当作它在已有 Gaussian 上的独立增量。

消融支持两种设计在所选任务上的收益，但分层模型的平均提升很大程度来自 Sweep，且表中没有独立的“仅加入分层、去掉实例监督”配置。另外，表 II 标题写有 12 tasks，而主实验为十项、消融实际列出三项；这里按表内数据解释，不扩展为十二任务结论。

## 六、真机：双 UR5e 上的九任务评估

### 6.1 数据与硬件条件

真机系统使用两台 **UR5e**，配备 **Robotiq 2F-85** 夹爪，通过两只 Xbox 控制器采集示范。两台 RealSense 相机采集 **640×480、30 Hz** RGB-D，训练使用多视角监督，推理只使用一个相机视角。（论文 §IV-A）

作者用一个语言条件策略学习九项任务，真机训练不使用仿真预训练或 sim-to-real 迁移。正文写有 30 条人类示范和 10 个评估 episodes，但没有明确 30 条是否按每项任务计算，因此这里不进一步推算总示范量。评估硬件为 RTX 4080。

```{figure} https://pic1.imgdb.cn/i/034cpZQoZ7V68ibnFotZls.png
:target: https://pic1.imgdb.cn/i/034cpZQoZ7V68ibnFotZls.png
:alt: 双 UR5e、RealSense 相机与遥操作系统，以及打乒乓球和折叠衣物的双臂任务示例
:width: 100%
:align: center

**原论文图 5：真实机器人设置与任务案例**。作者展示了打乒乓球、折叠衣物等操作；相机的 30 Hz 是采集频率，论文没有据此报告策略控制频率。
```

### 6.2 平均 62.22%，困难任务仍有明显空间

```{figure} https://pic1.imgdb.cn/i/034cpZOI0vx4akETzcN9bh.png
:target: https://pic1.imgdb.cn/i/034cpZOI0vx4akETzcN9bh.png
:alt: 九项真机任务上 PerAct²、ManiGaussian 与 ManiGaussian++ 的成功率柱状图，平均分别为 31.11%、45.56%、62.22%
:width: 100%
:align: center

**原论文图 4：九项真实双臂任务的成功率**。正文与图中均值一致，ManiGaussian++ 为 62.22%；摘要和结论写作 60%。
```

| 方法 | 九任务平均成功率 |
| --- | --- |
| PerAct² | 31.11% |
| ManiGaussian | 45.56% |
| ManiGaussian++ | **62.22%** |

从图 4 可见，LiftBox 与 PingPong 达到 100%，FoldClothes 达到 80%；HandoverBowl、PressHandsan、RelocateBrush 则均为 20%。完整方法的真机平均成功率比 ManiGaussian 高约 **16.66 个百分点**，但任务之间仍有较大差异。

这些结果补充了前作缺少的真机证据，也覆盖了衣物等非刚性物体操作。不过，成功案例本身不足以证明 Gaussian 动态模型准确恢复了布料物理；论文也未提供独立的未见任务、未见机器人或长期分布变化测试。

## 七、我的理解与适用边界

ManiGaussian++ 最有价值的变化，是让世界模型的结构对应双臂任务中的依赖关系：不仅学习整个场景会怎样变化，还尝试区分不同执行者分别造成什么变化。任务实例监督提供空间上的区分，分层预测提供动作之间的条件关系，两者共同作用于策略使用的表征。

如果把它看作一种小数据机器人模仿学习方案，实验支持其在所选双臂任务上的收益；如果把它看作通用物理模型或通用角色规划器，当前证据还不充分。实际使用时尤其需要考虑：

- **监督成本**：深度、相机标定、多视角重建和 GroundedSAM 伪标签都是训练条件的一部分。
- **角色定义**：论文未充分交代稳定臂与操作臂在不同任务、不同阶段的具体分配和切换规则。
- **预测范围**：主要通过下一时刻重建训练表征，没有验证长期滚动预测、接触力或形变的物理准确性。
- **控制与泛化**：动作依赖离散目标和运动规划器，真机实验尚不足以覆盖更广泛的机器人、任务与环境变化。

与前作连起来阅读，可以看到一条明确的研究路线：从“用未来场景监督操作表征”，推进到“按执行者的关系组织未来预测”。双臂条件下，怎样划分并组合动作影响，可能比单纯增加重建容量更值得研究。

## 参考资料

- [IROS 2025 官方会议程序](https://ras.papercept.net/conferences/conferences/IROS25/program/IROS25_ContentListWeb_2.html)：WeCT20.3，确认会议收录。
- [ManiGaussian++ arXiv v1](https://arxiv.org/abs/2506.19842v1)：§III 为方法，表 I 为十任务仿真结果，表 II 为三任务消融，§IV-E 与图 4、5 为真机实验。
- [官方代码仓库](https://github.com/yangzaijia/ManiGaussian_Bimanual)：已发布仿真实验代码；原 April-Yz 地址当前重定向至此。
- [ManiGaussian 论文解读](ManiGaussian-paper.md)：前作的动态 Gaussian、语义监督与单臂实验。
