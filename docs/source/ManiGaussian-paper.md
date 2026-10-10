# ManiGaussian 论文解读：动态高斯重建与操作表征

论文原题：ManiGaussian: Dynamic Gaussian Splatting for Multi-task Robotic Manipulation

- 作者：Guanxing Lu、Shiyi Zhang、Ziwei Wang、Changliu Liu、Jiwen Lu、Yansong Tang
- 团队：清华大学、南洋理工大学、卡内基梅隆大学（按所附 v2 的作者单位）
- 发表：**ECCV 2024**，[ECVA 官方论文条目](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5194_ECCV_2024_paper.php)
- 论文：[arXiv:2403.08321v2](https://arxiv.org/abs/2403.08321v2)，2024 年 7 月 18 日修订
- 项目与演示：[ManiGaussian](https://guanxinglu.github.io/ManiGaussian/)
- 代码：[GuanxingLu/ManiGaussian](https://github.com/GuanxingLu/ManiGaussian)

ManiGaussian 关注一个表征学习问题：**如果策略不仅要识别当前物体，还必须预测执行动作后场景如何变化，能否更好地理解推动、拖拽和堆叠中的物体交互？** 作者用带语义的 3D Gaussian 表示场景，学习动作引起的位置与旋转变化，再通过未来视角重建监督共享表征。

在十项 RLBench 仿真任务上，完整模型平均成功率为 **44.8%**，GNFactor 为 **31.7%**，相差 **13.1 个百分点**。这项工作已正式收入 ECCV 2024；下文的方法和实验数字采用所附 arXiv v2。

我最关注的是世界模型在这里的用途：未来重建帮助训练策略表征，部署时仍然根据当前观测直接预测下一关键帧动作。它没有在执行时对大量候选动作渲染未来，再搜索最优路径。

```{figure} https://pic1.imgdb.cn/i/034cp7citwQzurLZCuyhLT.png
:target: https://pic1.imgdb.cn/i/034cp7citwQzurLZCuyhLT.png
:alt: 堆叠两个玫红色方块的案例，GNFactor 误操作固定绿色底座，ManiGaussian 完成堆叠
:width: 100%
:align: center

**原论文图 1：看清物体，还需要理解物体之间的交互**。图中对照展示了具体成功与失败轨迹；整体效果仍需结合后文的任务成功率判断。
```

## 一、输入是单视角，学习目标包括多视角与未来场景

### 1.1 从 RGB-D 到共享体素表征

策略在每个决策步读取三类输入：正面相机的一帧 **128×128 RGB 图像及其对齐深度图**、描述任务目标的语言指令，以及 **4 维夹爪状态与时间向量**。按官方预处理代码，这 4 维具体为夹爪开合状态、左右两指的关节位置，以及按回合长度归一化的当前决策步；时间值从回合开始的 +1 逐渐减小至末尾的 −1。这里按代码明确其组成，避免将论文 §3.1 中较笼统的状态描述理解为“末端三维坐标加时间”。[状态预处理实现](https://github.com/GuanxingLu/ManiGaussian/blob/main/helpers/utils.py)

**反投影先把二维像素变成三维点。** 对每个具有有效深度的像素，利用相机内参将“像素位置 + 深度”还原为相机坐标系中的三维位置，再通过相机外参变换到场景坐标系，并附上对应像素的 RGB 颜色。随后，体素化将预设工作空间沿三个轴各划分为 100 份，把这些彩色点归入 $100\times100\times100$ 个空间格子。同一格内的点坐标与颜色分别取平均；格子的实际尺寸由工作空间范围决定，与输入图像的 128×128 分辨率是两个不同概念。

每个体素的 **10 维输入特征**由以下四部分拼接而成：

| 分量 | 维数 | 具体含义 |
| --- | ---: | --- |
| RGB | 3 | 落入该格子的观测点的平均颜色 |
| 三维坐标 | 3 | 这些点在场景坐标系中的平均位置 |
| 网格索引 | 3 | 格子沿三个轴的离散索引，分别除以 100 归一化 |
| 占用标记 | 1 | 有观测点落入时为 1，否则为 0 |

其中，点坐标描述“观测到的表面在哪里”，网格索引描述“当前格子在哪里”。占用标记只表示该格子是否接收到观测点；单视角看不到的遮挡区域也可能为 0，因此不能把 0 一概当作物理上的空闲空间。这些特征的聚合与拼接方式可见[官方体素化实现](https://github.com/GuanxingLu/ManiGaussian/blob/main/voxel/voxel_grid.py)。

**浅层 3D U-Net 再把这些原始属性编码为可学习的空间特征。** 网络通过三维卷积、下采样、上采样与跳跃连接汇聚邻域信息，并恢复到原来的体素分辨率。按附录 B，输入形状为 $100^3\times10$，输出为 $100^3\times128$：**每个空间格子对应一个 128 维特征向量**，仍保留三维位置关系；128 表示特征通道数，不表示网格变成了 128³，也不是将整个场景压缩成一个 128 维向量。（[论文 §3.2、附录 B](https://arxiv.org/html/2403.08321v2)）

**语言与夹爪状态在动作分支中参与融合，不包含在上述 10 维体素输入中。** 官方配置将编码后的体素特征按 $5\times5\times5$ 的块、以步长 5 聚合为 $20^3=8000$ 个视觉 token；4 维状态经全连接层映射后，复制到各空间块并沿特征维拼接。语言指令由 CLIP 文本编码器生成 token 特征，经线性映射后与视觉 token 沿序列维拼接，再加入位置编码，交给 PerceiverIO 建立任务语言、场景位置与机器人状态之间的关联。[融合实现](https://github.com/GuanxingLu/ManiGaussian/blob/main/agents/manigaussian_bc/perceiver_lang_io.py)、[官方配置](https://github.com/GuanxingLu/ManiGaussian/blob/main/conf/method/ManiGaussian_BC.yaml)

同一套特征支持动作预测与 Gaussian 重建两个方向：动作分支学习下一步应该把末端移动到哪里，重建分支要求特征保留场景几何、语义和动作后果。

**单视角指的是策略的观测输入。** 训练时，作者用 **20 个已标定相机视角**提供重建监督，与 GNFactor 保持一致。因此，这不是只凭单张 RGB 图像就能完成全部训练的方案，深度和多视角监督均是重要条件。

### 1.2 输出是关键帧动作，由运动规划器执行

动作由四部分组成：目标平移位置、目标旋转、夹爪开合和碰撞设置。平移在 $100^3$ 体素上分类；旋转按每轴 $5^\circ$ 离散，即每轴 72 个类别；夹爪与碰撞设置为二元决策。

示范通过夹爪状态改变、速度接近零等规则抽取关键帧。模型预测下一关键帧末端目标，再由预定义运动规划器（例如 RRT-Connect）执行。因此，论文中的一步策略决策并非一个底层电机控制周期。（附录 A）

```{figure} https://pic1.imgdb.cn/i/034cp7hverjCwDTDkKbqpu.png
:target: https://pic1.imgdb.cn/i/034cp7hverjCwDTDkKbqpu.png
:alt: ManiGaussian 将单视角 RGB-D 体素化，用共享表征预测动作和 Gaussian 参数，再根据动作预测形变并重建未来场景
:width: 100%
:align: center

**原论文图 2：策略与 Gaussian 世界模型的联合训练**。未来场景一致性约束通过共享表征影响动作学习；实现中的执行路径无需渲染未来图像。
```

## 二、动态 Gaussian 表示了什么？

### 2.1 给三维高斯附上语义特征

普通 Gaussian Splatting 用三维 Gaussian 基元表示场景，包含位置、颜色、旋转、尺度和不透明度。ManiGaussian 增加语义特征，将第 $i$ 个基元写为：

$$
\theta_{i,t}=(\mu_{i,t},c_i,r_{i,t},s_i,\sigma_i,f_i).
$$

其中 $\mu$ 和 $r$ 描述位置与旋转，$c$、$s$、$\sigma$ 分别描述颜色、尺度和不透明度，$f$ 是由预训练视觉特征提供监督的语义表示。附录配置使用 **16,384 个 Gaussian 点**。

这些基元可以投影到给定相机视角，并通过可微的 alpha blending 合成颜色或语义特征图。因此，监督不必直接提供每个 Gaussian 的真实参数，而可以比较渲染结果与真实观测。（论文 §3.3）

### 2.2 未来变化通过位置和旋转增量表示

作者采用刚体场景假设，在传播过程中保持颜色、尺度、不透明度和语义特征不变，只预测位置与旋转参数的变化：

$$
\begin{aligned}
\mu_{i,t+1}&=\mu_{i,t}+\Delta\mu_{i,t},\\
r_{i,t+1}&=r_{i,t}+\Delta r_{i,t}.
\end{aligned}
$$

这是论文的参数更新记法；实现中的旋转采用四元数表示，不能把上式解释为对旋转矩阵直接做加法。形变网络学习的是 Gaussian 如何随动作变化，也没有显式求解接触力、摩擦或刚体动力学方程。

**我的理解**：相比只监督静态重建，这个目标要求特征保留“哪一部分可能随夹爪移动、动作会带动哪个物体”等信息。但预测了运动外观，不等于获得了经过物理定律验证的仿真器。

## 三、Gaussian 世界模型怎样提供监督？

### 3.1 编码、回归、形变与渲染

论文将世界模型拆为四个模块。用 $o_t$ 表示当前观测，$v_t$ 表示共享视觉特征，$w$ 表示监督相机的位姿，可将其写为：

$$
\begin{aligned}
v_t&=q_\phi(o_t),\\
\theta_t&=g_\phi(v_t),\\
\Delta\theta_t&=p_\phi(\theta_t,a_t),\\
\widehat C_{t+1}^{(w)}&=\mathcal R(\theta_{t+1},w).
\end{aligned}
$$

$q_\phi$ 编码当前场景，$g_\phi$ 的多个预测头回归 Gaussian 参数，$p_\phi$ 预测动作条件下的位置和旋转增量，$\mathcal R$ 渲染未来 RGB。上式沿用正文的简化表达；附录说明形变预测还利用当前视觉特征。（论文式 4，附录 B）

未来监督通常与示范中的下一个目标关键帧配对，时间跨度不固定，$t+1$ 不宜理解为固定帧率下的下一张视频图像。[关键帧配对实现](https://github.com/GuanxingLu/ManiGaussian/blob/main/agents/manigaussian_bc/launch_utils.py)

### 3.2 四类目标各自约束不同信息

| 损失 | 监督内容 | 对表征的要求 |
| --- | --- | --- |
| $\mathcal L_{\mathrm{Act}}$ | 示范中的动作类别 | 预测正确关键帧目标 |
| $\mathcal L_{\mathrm{Geo}}$ | 当前多视角 RGB 重建 | 保留支持三维重建的信息 |
| $\mathcal L_{\mathrm{Sem}}$ | 渲染语义与预训练视觉特征的一致性 | 保留任务相关的高层视觉信息 |
| $\mathcal L_{\mathrm{Dyna}}$ | 动作后的未来 RGB 重建 | 保留物体交互与运动信息 |

当前与未来重建采用平方误差：

$$
\begin{aligned}
\mathcal L_{\mathrm{Geo}}&=\lVert C_t-\widehat C_t\rVert_2^2,\\
\mathcal L_{\mathrm{Dyna}}&=
\lVert C_{t+1}-\widehat C_{t+1}(o_t,a_t)\rVert_2^2.
\end{aligned}
$$

虽然命名为 Geo，当前场景损失直接比较的是 **RGB 图像**，并非真实三维网格或深度的逐点误差。

语义监督来自 **Stable Diffusion 的视觉特征**，采用余弦相似性约束：

$$
\mathcal L_{\mathrm{Sem}}
=1-\cos(\widehat F_t,F_t^{\mathrm{teacher}}).
$$

这些特征只为训练提供语义监督，部署不需要运行完整 Stable Diffusion 图像生成流程。（论文 §3.4）

完整目标为：

$$
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{Act}}
+0.01\mathcal L_{\mathrm{Geo}}\\
&+0.0001\mathcal L_{\mathrm{Sem}}
+0.001\mathcal L_{\mathrm{Dyna}}.
\end{aligned}
$$

作者在前 3,000 次更新冻结形变预测器，先让当前表征和 Gaussian 回归稳定，再联合训练。三种辅助目标均服务于动作学习；它们的权重并不是越大越好。

### 3.3 部署路径不需要未来渲染

论文把整个系统概括为在 Gaussian 表征空间学习动作。更具体地看，官方实现中的 PerceiverIO 动作网络读取共享体素特征、本体状态和语言；Gaussian 分支通过训练损失塑造这些特征。未来重建使用示范中的专家动作作为条件，执行函数则明确关闭 `use_neural_rendering`。因此，世界模型在这里主要提供训练监督，没有执行时的候选动作搜索或 MPC。[训练与执行代码](https://github.com/GuanxingLu/ManiGaussian/blob/main/agents/manigaussian_bc/qattention_manigaussian_bc_agent.py)

这与“训练中预测未来，执行时保留动作策略”的思路一致。关键区别在于 ManiGaussian 选择显式三维 Gaussian 及其形变作为未来表示，并利用动作条件和多视角渲染来监督它。

## 四、十项 RLBench 任务的主结果

### 4.1 协议与比较范围

作者选取十项 RLBench 仿真任务，共 166 个变体，变化包括颜色、大小、数量、摆放与物体类别等。**166 是任务配置变体数，不是 166 项独立任务**；论文也没有将其全部定义为训练未见的物体或 OOD 测试。

每任务使用 20 条训练示范，测试最终 checkpoint，每任务评估 25 回合；在最多 25 次关键帧动作决策内完成语言目标即成功。比较方法使用相同版本的 PerceiverIO，并在两张 RTX 4090 上训练 100K 次更新，batch size 为 2，使用 LAMB 和初始学习率 $5\times10^{-4}$。（论文 §4.1）

### 4.2 平均提升明显，但长任务仍然困难

下表转录原论文表 1，单位为成功率百分比；“四相机”是 PerAct 的多视角输入版本。

| 任务 | PerAct | PerAct 四相机 | GNFactor | ManiGaussian |
| --- | --- | --- | --- | --- |
| close jar | 18.7 | 21.3 | 25.3 | **28.0** |
| open drawer | 54.7 | 44.0 | **76.0** | **76.0** |
| sweep to dustpan | 0.0 | 0.0 | 28.0 | **64.0** |
| meat off grill | 40.0 | **65.3** | 57.3 | 60.0 |
| turn tap | 38.7 | 46.7 | 50.7 | **56.0** |
| slide block | 18.7 | 16.0 | 20.0 | **24.0** |
| put in drawer | 2.7 | 6.7 | 0.0 | **16.0** |
| drag stick | 5.3 | 12.0 | 37.3 | **92.0** |
| push buttons | 18.7 | 9.3 | 18.7 | **20.0** |
| stack blocks | 6.7 | 5.3 | 4.0 | **12.0** |
| 十任务均值 | 20.4 | 22.7 | 31.7 | **44.8** |

44.8% 相比 31.7% 增加 **13.1 个百分点**，对应约 **41.3% 的相对提升**。摘要中的“13.1%”应按百分点理解。

较大的收益集中在工具使用：drag stick 提高 54.7 个百分点，sweep to dustpan 提高 36 个百分点。但模型并非所有任务都严格最优：open drawer 与 GNFactor 持平，meat off grill 低于四相机 PerAct；put in drawer 和 stack blocks 也仍只有 16% 与 12%。

这些数字反映论文当时、该训练评测协议下的比较，不代表当前所有机器人策略的排名。PDF 没有明确给出主表的训练种子数、标准差或置信区间，小幅差距不宜解读为稳定优势。

## 五、消融：几何、语义与动态监督如何配合？

### 5.1 动态监督在静态重建之上继续改善结果

原论文表 2 的消融从“只训练共享表征与动作预测”的版本开始。它的 23.6% 是消融基线，不能与主表 PerAct 的 20.4% 混为一行。

| 几何重建 Geo | 语义 Sem | 动态 Dyna | 十任务平均成功率 |
| --- | --- | --- | --- |
| 无 | 无 | 无 | 23.6% |
| 有 | 无 | 无 | 39.2% |
| 有 | 有 | 无 | 41.6% |
| 有 | 无 | 有 | 43.6% |
| 有 | 有 | 有 | **44.8%** |

只加入 Gaussian 几何重建，提高 15.6 个百分点。在此基础上，单独加入语义监督提高 2.4 个百分点，单独加入动态监督提高 4.4 个百分点；两者同时加入则比几何版本提高 5.6 个百分点。

若以已经包含语义的 41.6% 为起点，加入动态后的增量是 **3.2 个百分点**。比较模块贡献时需要保持相同起点，而不能把不同消融路径的增量直接相加。

### 5.2 辅助目标存在权衡

作者还将十个任务分为 Planning、Long、Tools、Motion、Screw、Occlusion 六组。完整方案并非每组最好：相对 Geo+Dyna，Planning 从 54% 降到 40%，Motion 从 64% 降到 56%；整体均值仍有所上升。

这六组包含的任务数不同，总平均按十项任务计算，不能直接平均六列组分数。附录的权重实验也显示，语义权重从 $10^{-4}$ 增至 $10^{-3}$ 时，Geo+Sem 版本从 41.6% 降到 37.6%，说明辅助重建与行为克隆需要平衡。

## 六、训练速度与可视化需要分别阅读

### 6.1 2.29 倍指训练速度

```{figure} https://pic1.imgdb.cn/i/034cp7jRXAGE3bcYZD65NM.png
:target: https://pic1.imgdb.cn/i/034cp7jRXAGE3bcYZD65NM.png
:alt: ManiGaussian 与 GNFactor 在训练耗时和平均任务成功率上的学习曲线
:width: 100%
:align: center

**原论文图 3：专门的训练效率比较**。图注说明，为公平比较移除了重建损失中的辅助项；灰色虚线为移动平均，因此不能把这条曲线直接当作完整配置的主结果。
```

作者每 10K 次更新评估一次，在这组设置中报告相对 GNFactor **2.29 倍训练加速，以及该设置下约 1.18 倍的平均成功率**。它比较的是训练耗时与该实验的成功率，并未报告机器人在线动作频率提高 2.29 倍。（论文 §4.3）

### 6.2 轨迹案例展示具体行为差异

```{figure} https://pic1.imgdb.cn/i/034cp7pGtSxICb2Qg0tZ9p.png
:target: https://pic1.imgdb.cn/i/034cp7pGtSxICb2Qg0tZ9p.png
:alt: 滑动方块与转动左侧水龙头的执行轨迹，比较 GNFactor 与 ManiGaussian 的行为差异
:width: 100%
:align: center

**原论文图 4：两个任务的执行案例**。上方展示滑块操作的纠正，下方展示目标水龙头选择与操作；这些是定性例子，不是额外的扰动鲁棒性测试。
```

在滑块案例中，ManiGaussian 回到物体附近重新推动；在水龙头案例中，它选择并操作了指令对应的左侧目标。作者将这种差异解释为更好的语义与动态理解，但单条轨迹不足以单独证明成功由哪个辅助损失造成。

### 6.3 重建图移除了动作损失

```{figure} https://pic1.imgdb.cn/i/034cp7rGUmpCoCzFjbHGwc.png
:target: https://pic1.imgdb.cn/i/034cp7rGUmpCoCzFjbHGwc.png
:alt: 从前视角观测重建当前新视角与未来新视角，展示 Gaussian 对夹爪和物体变化的预测
:width: 100%
:align: center

**原论文图 5：当前场景与未来场景的新视角重建**。作者为展示重建效果明确移除了 action loss；图中的 PSNR 不能直接当作完整联合训练策略的重建指标。
```

这个实验表明 Gaussian 表示可以渲染当前不可直接观察的视角，也能预测夹爪和受其影响物体的位置变化。它支持“可学习动作条件下的场景变化”这一设计，但并未建立图像重建质量与操作成功率之间的一一对应关系。

## 七、我的理解与适用边界

ManiGaussian 把静态三维重建推进到动作条件下的动态重建。我的理解是，真正面向控制的变化在于：共享表征不仅要解释“这个场景现在长什么样”，还要能够解释“这个动作之后哪些物体会移动”。显式 Gaussian 让这种变化可渲染、可视化，也使未来图像成为训练信号。

它与 [EgoWAM](EgoWAM-paper.md) 都关注未来监督如何塑造动作表征，但任务条件不同：ManiGaussian 使用机器人示范、深度和多视角监督学习三维场景形变；EgoWAM 则比较人机联合训练中不同未来表征的迁移收益。不能仅因两者都使用世界预测，就把数据来源、预测接口和实验结论合并。

当前证据的边界包括：

- **仿真范围**：正文和补充材料均为 RLBench，没有真机实验，也没有独立的未见物体或传感噪声测试。
- **数据条件**：推理虽用单个 RGB-D 视角，训练仍依赖标定后的多视角监督，作者在结论中明确承认这一限制。
- **动态表达**：刚体假设下只更新 Gaussian 位置与旋转，颜色、尺度、不透明度和语义保持不变；柔性物体、大外观变化与长期预测可靠性尚未得到验证。
- **控制范围**：策略依赖关键帧与预定义运动规划器；它并非直接输出高频力矩的低层控制系统，长任务成功率也仍偏低。

这篇论文提供了一个值得继续研究的训练目标：在有限机器人示范下，用动作引起的三维场景变化补充监督。已有结果支持其在所选任务上的表征收益，而更广泛的物理理解与真实部署能力仍需要额外证据。

## 参考资料

- [ECCV 2024 官方论文条目](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5194_ECCV_2024_paper.php)：会议论文、补充材料与出版方 DOI 入口。
- [ManiGaussian arXiv v2](https://arxiv.org/abs/2403.08321v2)：方法见 §3，结果见 §4，关键帧与模块细节见附录 A、B，损失权重见附录 C。
- [项目页面](https://guanxinglu.github.io/ManiGaussian/)与[官方代码](https://github.com/GuanxingLu/ManiGaussian)：演示及实现参考。
