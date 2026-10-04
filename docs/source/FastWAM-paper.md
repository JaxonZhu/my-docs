# Fast-WAM 论文与源码解读：视频监督与未来想象

论文原题：Fast-WAM: Do World Action Models Need Test-time Future Imagination?

- 作者：Tianyuan Yuan、Zibin Dong、Yicheng Liu、Hang Zhao
- 团队：清华大学交叉信息研究院（IIIS）、Galaxea AI
- 论文：[arXiv:2603.16666](https://arxiv.org/abs/2603.16666)，2026 年 3 月 17 日首次提交，3 月 23 日更新 v2
- 项目：[Fast-WAM 官方项目页](https://yuantianyuan01.github.io/FastWAM/)
- 代码：[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)
- 权重：[yuanty/fastwam](https://huggingface.co/yuanty/fastwam)

Fast-WAM 把 WAM 中经常同时出现的两件事拆开研究：**训练时学习预测未来视频，是否意味着推理时也必须生成未来视频？** 作者保留视频与动作的联合训练，通过注意力掩码让动作只读取当前观测。部署时，视频骨干只前向编码一次，动作专家继续迭代去噪，从而省去未来视频采样。

我最关注的是这组对照实验背后的工程含义：哪些计算真正改善了训练，哪些计算必须保留到部署？论文在 LIBERO 和 RoboTwin 上发现，移除视频联合训练带来的下降，大于不同未来想象方式之间的差距；但这个结果仍有任务、骨干和训练规模的适用范围。

:::{note}
**阅读范围**：方法与原始实验以 [论文 v2](https://arxiv.org/html/2603.16666v2) 为准；实现细节核对官方仓库 [7faa711](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)（2026 年 8 月 20 日提交）。后续代码增加了加速路径和 Optional IDM，文末单独说明。本篇属于论文阅读与静态源码分析，没有自行复现训练或成功率。
:::

## 一、WAM 的收益究竟来自哪里？

### 1.1 两种作用经常被绑定在一起

许多 WAM 将未来视觉建模与动作预测结合。这里至少包含两种不同的作用：

1. **训练时的视频监督**：未来观测提供比低维动作标签更丰富的学习信号，可能帮助骨干形成与运动和交互有关的表示。
2. **推理时的未来生成**：模型显式采样未来视觉，再根据这些未来预测动作，或者联合去噪视频与动作。

第二种作用需要反复处理大量视频 tokens，带来额外延迟。只比较一个 WAM 与一个 VLA，很难判断性能差异究竟来自视频预训练、联合训练目标，还是部署时的额外计算。

Fast-WAM 的切入点是：在尽量统一骨干、tokenization 和训练设置的框架中，分别改变训练监督和推理依赖，观察两者的作用。（论文 §1、§3.3）

```{figure} images/FastWAM/fastwam-paradigms.png
:alt: 联合生成、先视频后动作、Fast-WAM 三种范式，Fast-WAM 在推理时跳过未来视频生成
:width: 100%
:align: center

**原论文图 1：三种 WAM 范式**。Fast-WAM 的 single forward pass 指视频骨干对当前观测的单次编码，动作生成仍包含多步去噪。
```

### 1.2 从显式未来到当前观测的表示

令 $o$ 表示当前观测，$\ell$ 表示语言指令，$a_{1:H}$ 表示动作块。先想象再执行的思路可以写成：

$$
p(a_{1:H}\mid o,\ell)
=\int p(v_{1:T}\mid o,\ell)\,
p(a_{1:H}\mid o,\ell,v_{1:T})\,dv_{1:T}.
$$

其中 $v_{1:T}$ 是未来视觉观测。Fast-WAM 在部署时直接建模：

$$
p_\theta(a_{1:H}\mid o,\ell)
=p_\theta(a_{1:H}\mid z(o,\ell)),
$$

$z(o,\ell)$ 由视频骨干编码当前上下文得到，不需要先采样未来视频。实际实现还加入本体状态；以上写法沿用论文的简化记号。（论文式 1–4）

**我的理解**：训练目标能够影响模型参数和表示，却不一定要成为推理时的必选输出。Fast-WAM 要验证的是这种拆分能否保留控制能力，而不是宣称机器人已经不需要任何形式的未来推理。

## 二、MoT 如何共享视觉信息？

### 2.1 两个专家，共同参与注意力计算

Fast-WAM 使用 Wan2.2-5B 的视频 DiT，并增加约 1B 参数的 Action DiT。二者组成 Mixture-of-Transformer（MoT），论文将总模型规模记为 6B。

| 组件 | 结构与作用 |
| --- | --- |
| 视频 VAE | 将 RGB 图像或视频编码成 latent；使用 Wan 预训练权重 |
| T5 文本编码器 | 产生语言条件，通过 cross-attention 提供给两个专家 |
| Video DiT | 约 5B 参数；隐藏维度 3072，30 层 |
| Action DiT | 约 1B 参数；隐藏维度 1024，30 层 |
| MoT 注意力 | 按层组织两个专家的 Q、K、V，以掩码控制可见关系 |
| 本体状态投影 | 将状态投影到 4096 维，作为额外 context token |

源码中的两个专家拥有各自的投影、MLP 等参数。**共享注意力计算不等于共享整套 Transformer 权重。** Q、K、V 被投影到兼容的注意力空间后，在序列维拼接；注意力输出再拆回各专家，继续各自的残差、cross-attention 和 MLP。配置使用 24 个 attention heads，每个 head 128 维，因此不能直接用 action hidden dimension 除以 head 数推断其注意力维度。

源码入口是 [模型配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/model/fastwam.yaml) 与下文的 MoT 实现。

```{figure} images/FastWAM/fastwam-architecture.png
:alt: Fast-WAM 视频专家和动作专家通过共享注意力连接，语言条件通过交叉注意力注入
:width: 100%
:align: center

**原论文图 2(a)：模型结构**。训练时同时处理当前观测、未来视频与动作；推理时保留当前观测编码和动作专家。
```

### 2.2 三组 tokens 的可见关系

基础 Fast-WAM 将 tokens 分为当前观测 $O$、未来视频 $V$ 和动作 $A$。当前观测是无噪声的视觉锚点，未来视频与动作则参与去噪训练。

下表中，行表示发出 query 的 token，列表示可读取的 key/value：

| Query ＼ Key/Value | 当前观测 $O$ | 未来视频 $V$ | 动作 $A$ |
| --- | --- | --- | --- |
| 当前观测 $O$ | 可见 | 不可见 | 不可见 |
| 未来视频 $V$ | 可见 | 可见 | 不可见 |
| 动作 $A$ | 可见 | 不可见 | 可见 |

未来视频块内部与动作块内部都允许双向注意力；这不是逐帧或逐动作 token 的自回归解码。语言及本体状态通过另一个 cross-attention context 提供，不列入这张 self-attention 表。

```{figure} images/FastWAM/fastwam-attention-mask.png
:alt: 训练和推理注意力掩码，动作只读取当前观测和动作 tokens，未来视频在推理时移除
:width: 320px
:align: center

**原论文图 2(b)：训练与推理掩码**。当前帧不能读取未来帧，动作也不能读取未来视频，因此删去未来视频 tokens 不会破坏基础模型的前向条件依赖。
```

### 2.3 看不到未来，为什么还能受益于视频监督？

动作无法直接读取未来视频 tokens，不代表视频训练与动作训练完全独立。视频损失与动作损失都会影响视频骨干；当前帧与未来帧也使用同一套视频专家参数。视频目标改变这些参数后，当前帧的编码及其提供的 K/V 会随之改变，动作专家就可能受益。

这里需要区分**前向信息流**与**训练时的参数更新**。掩码阻止的是动作读取样本中的未来视觉信息，并没有阻止视频损失塑造共享的视频骨干。至于学到的表示是否已经具备完整物理因果知识，论文的成功率实验并不能直接证明。

## 三、视频与动作如何联合训练？

### 3.1 两种目标使用相同的 flow matching 形式

对目标 $y$，采样高斯噪声 $\epsilon$ 和流时间 $\tau\in(0,1)$：

$$
y_\tau=(1-\tau)y+\tau\epsilon,
\qquad \epsilon\sim\mathcal N(0,I).
$$

模型预测的目标速度为 $\epsilon-y$：

$$
\mathcal L_{\mathrm{FM}}(y)
=\mathbb E\left[
\left\|f_\theta(y_\tau,\tau,o,\ell)-(\epsilon-y)\right\|_2^2
\right].
$$

这里 $\tau=0$ 对应干净数据，$\tau=1$ 对应噪声；采样沿噪声到数据的方向积分。不要与采用相反时间方向的 flow matching 记号混用。（论文式 5–6）

动作目标取 $y=a_{1:H}$，视频目标取未来观测的 VAE latent $y=z_{1:T}$，联合损失为：

$$
\mathcal L
=\mathcal L_{\mathrm{act}}
+\lambda\mathcal L_{\mathrm{vid}}.
$$

视频监督作用在 latent 上，并非直接对 RGB 图像计算这个损失。实现保持第一帧干净，视频损失排除它，同时对视频和动作的 padding 使用有效性掩码。默认配置中两个目标的权重均为 1。（论文式 7–9；基础模型的训练实现）

### 3.2 初始化、冻结与训练设置

视频专家从 Wan2.2 初始化。动作专家的 backbone 则通过 Wan 权重插值和缩放初始化，动作输入映射与输出 head 等不直接照搬视频层。因此，不能将整个 Action DiT 描述成完全随机初始化。

当前实现冻结 VAE 和文本编码器，训练视频专家、动作专家及本体状态投影。默认可使用预计算文本 embedding，后续版本也支持在线文本编码。

论文报告的训练配置如下：

| 项目 | 论文设置 |
| --- | --- |
| 动作 horizon | 32 |
| LIBERO | 20,000 steps |
| RoboTwin / 真机毛巾折叠 | 各 30,000 steps |
| 优化器 | AdamW |
| 学习率 / weight decay | $10^{-4}$ / 0.01 |
| 学习率调度 | cosine annealing |
| 精度与梯度 | mixed precision，gradient clipping 1.0 |
| 推理去噪 | 10 步，CFG scale 1.0 |

“没有 embodied pretraining”指没有在额外机器人数据上先做预训练，**不代表没有预训练**：Wan 的大规模视频预训练仍然是重要起点。去掉 video co-training 的消融也保留这一初始化，只移除当前机器人训练阶段的视频目标。

## 四、从数据窗口到模型输入

### 4.1 数据规模与评估协议

| 数据集 / 场景 | 训练数据 | 评估方式 |
| --- | --- | --- |
| LIBERO | 4 个 suites，各 10 个任务、500 条 demonstrations，总计 2,000 条 | 40 个任务合计 2,000 次 trials |
| RoboTwin 2.0 | 2,500 条 clean 与 25,000 条 randomized demonstrations | clean 与 randomized 条件下，每任务 100 次 trials |
| 真机毛巾折叠 | Galaxea R1 Lite 上采集 60 小时遥操作数据 | 成功率与平均完成时间 |

RoboTwin 实验覆盖 50 个任务，附录逐项报告结果。clean 与 randomized 的总体均值接近，可以说明这组评估条件下表现稳定；它不能直接等同于对任意视觉域变化的泛化保证。（论文 §4.2、附录 A.1）

### 4.2 33、9 和 3 分别是什么？

这是读源码时最容易混淆的一组数字。当前数据配置使用 `num_frames=33`、`global_sample_stride=1`、`action_video_freq_ratio=4`：

$$
\underbrace{33}_{\text{原始时间窗口}}
\ \xrightarrow{\text{每 4 步采样图像}}\
\underbrace{9}_{\text{RGB 帧}}
\ \xrightarrow{\text{VAE 时序压缩}}\
\underbrace{3}_{\text{latent 时刻}}.
$$

动作取这一窗口中的 32 步；图像取对应的 9 个采样时刻。VAE 对 9 张 RGB 图进一步进行约 4 倍时间压缩，其因果编码长度为：

$$
T_{\mathrm{latent}}=1+\frac{T_{\mathrm{RGB}}-1}{4}
=1+\frac{9-1}{4}=3.
$$

所以，`action_video_freq_ratio=4` 与 VAE 的时间压缩率是**两次不同的处理**。不能把 9 张输入视频帧直接写成 9 个 latent 时刻，也不能仅用 `T//4` 表达保留首帧的时序长度。

### 4.3 多相机、动作与状态

下表对应核对时的官方配置。图像尺寸以 $H\times W$ 表示，latent 形状采用 $[B,C,T,H,W]$：

| 项目 | LIBERO | RoboTwin |
| --- | --- | --- |
| 相机 | 主视角 + 腕部，共 2 路 | 高位 + 左右腕部，共 3 路 |
| 单路 resize | $224\times224$ | $240\times320$ |
| 最终复合图像 | 水平拼接，$224\times448$ | 专用拼接并 resize，$384\times320$ |
| 训练 RGB 视频 | $[B,3,9,224,448]$ | $[B,3,9,384,320]$ |
| VAE latent | $[B,48,3,14,28]$ | $[B,48,3,24,20]$ |
| 动作块 | $[B,32,7]$ | $[B,32,14]$ |
| 本体状态维度 | 8 | 14 |
| 默认归一化 | min/max | z-score |

LIBERO 的配置将动作前 6 维标记为 delta、夹爪维标记为非 delta；RoboTwin 评估接口执行的是 14 维双臂关节位置与夹爪命令（qpos），不是 14 维末端位姿。动作单位、坐标与反归一化必须遵循各环境的数据和控制接口，不能跨任务直接复用统计文件。

训练样本虽然包含一段本体状态序列，基础模型实际只取窗口首个状态作为 context，未把未来状态提供给动作预测。

这些 latent 还要经过视频 DiT 的 patch embedding，**latent 网格元素数也不等于最终 attention token 数**。形状只是当前代码配置的推导，不应反推为所有版本、所有 checkpoint 的固定规格。

来源：[LIBERO 数据配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/data/libero_2cam.yaml)、[RoboTwin 数据配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/data/robotwin.yaml)。

## 五、推理为何能省掉视频去噪？

### 5.1 单次视频编码，多次动作去噪

基础 Fast-WAM 的一次预测可以概括为：

```text
当前多相机图像 + 语言 + 本体状态
  → VAE 编码当前图像
  → Video DiT 以干净首帧做一次 prefill
  → 缓存各层 video K/V
  → Action DiT 进行 10 步去噪，每一步读取同一份缓存
  → 输出 32 步 action chunk
```

由于首帧既不读取未来视频，也不读取动作，它的表示不会随着 action denoising 改变，因此可以复用缓存。省下的是未来视频的反复去噪，以及本可重复执行的视频上下文计算。

**视频骨干仍然在部署时运行，动作专家也仍然迭代运行。** Fast-WAM 并没有将整个 6B 策略变成一次前向，也不是只删除最终 VAE 的 RGB 解码步骤。

### 5.2 预测 horizon 与执行 horizon 要分开

32 是模型输出动作块的长度，不自动等于重规划周期。核对的公开评估配置中，LIBERO 执行动作前缀为 10 步，RoboTwin 为 24 步，再以新观测进行下一次预测。具体部署仍应以评估入口的参数及其覆盖值为准。

论文报告的 190 ms 是指定硬件上的推理延迟，不能由此直接推出底层控制频率、完整机器人系统延迟，或判定系统是否采用异步执行。动作步长、执行前缀、通信和控制器都影响实际闭环。

### 5.3 四个变体改变了什么？

| 变体 | 视频 co-training | 推理时生成未来视频 | 动作读取的视觉信息 |
| --- | --- | --- | --- |
| Fast-WAM | 有 | 无 | 当前观测 |
| Fast-WAM-Joint | 有 | 有，与动作联合去噪 | 当前观测与去噪中的未来视频 |
| Fast-WAM-IDM | 有 | 有，先视频后动作 | 当前观测与生成的未来表示 |
| 无 video co-training | 无 | 无 | 当前观测 |

论文的 IDM 训练对条件视频以 0.5 的概率进行噪声增强，缓解动作模型只见过真实未来、测试却读取生成未来的差异。

Joint 的实现细节也值得注意：当前源码放开的是 **action 对全部 video tokens 的读取**，video 对 action 的读取仍关闭。联合去噪描述的是采样过程，不能直接等同于双向跨模态注意力。

这些变体尽量共享实现框架，但原论文的 Joint、IDM 和基础模型是不同训练配置，并非同一 checkpoint 在推理时简单开关一个分支。IDM 还改变了条件视频的训练组织方式，因此不宜将它们称为排除了所有混杂因素的严格因果证明。

## 六、仿真结果支持怎样的结论？

### 6.1 先看框架内部的对照

以下成功率均为百分比，转录自论文表 1、表 2：

| 变体 | RoboTwin Clean | RoboTwin Rand. | RoboTwin 平均 | LIBERO 平均 |
| --- | --- | --- | --- | --- |
| Fast-WAM | 91.88 | 91.78 | 91.8 | 97.6 |
| Fast-WAM-Joint | 90.84 | 90.32 | 90.6 | 98.5 |
| Fast-WAM-IDM | 91.16 | 91.34 | 91.3 | 98.0 |
| 无 video co-training | 82.76 | 84.80 | 83.8 | 93.5 |

以论文表中四舍五入的均值计算，移除视频监督使 RoboTwin 下降 **8.0 个百分点**，LIBERO 下降 **4.1 个百分点**。相比之下，基础模型与两个想象变体的均值差距为 0.4–1.2 个百分点，而且优势方向并不一致。

这支持作者的主要观察：在这套骨干、数据与任务设置下，视频训练目标的作用大于显式未来生成的边际作用。它不意味着 Joint 或 IDM 在每一个任务上都没有价值。

### 6.2 LIBERO 的下降集中在哪里？

| 变体 | Spatial | Object | Goal | Long |
| --- | --- | --- | --- | --- |
| Fast-WAM | 98.2 | 100.0 | 97.0 | 95.2 |
| Fast-WAM-Joint | 99.6 | 99.4 | 98.2 | 96.8 |
| Fast-WAM-IDM | 98.8 | 97.8 | 97.8 | 97.6 |
| 无 video co-training | 89.2 | 99.2 | 95.4 | 90.0 |

Spatial 与 Long 属于 **LIBERO**。去掉视频监督后，这两个套件分别下降 9.0 和 5.2 个百分点。Joint 相比基础模型在 Spatial 上提高 1.4 个百分点，IDM 在 Long 上提高 2.4 个百分点，说明聚合均值会掩盖不同子集上的权衡。

套件名称并不是独立能力测量：仅凭 Spatial 或 Long 的结果，还不能断言模型具体改善了哪一种空间推理机制，也不能把小幅均值差距当成已验证的统计显著差异。

### 6.3 与其他方法比较时保留预训练口径

| 方法 | 额外 embodied pretraining | RoboTwin 平均 | LIBERO 平均 |
| --- | --- | --- | --- |
| $\pi_{0.5}$ | 有 | 79.8 | 96.9 |
| Motus | 有 | 87.8 | 97.7 |
| Motus from Wan2.2 | 无 | 77.3 | — |
| LingBot-VA | 有 | 92.2 | 98.5 |
| Fast-WAM | 无 | 91.8 | 97.6 |

Fast-WAM 在不使用额外具身预训练的情况下接近较强 WAM 基线，这是有价值的结果。但不同论文的骨干、预训练数据与训练预算并不完全一致，不能仅从这张跨方法表格隔离出架构贡献。论文另列 LingBot-VA from Wan2.2 的 80.6%，该行只有 clean 结果，未报告 randomized 分数，因而不在这里当作两种场景的共同均值。

### 6.4 平均分没有消除困难任务

论文附录中，基础 Fast-WAM 仍有成功率明显偏低的任务：

| RoboTwin 任务 | Clean | Rand. |
| --- | --- | --- |
| Hanging Mug | 58 | 62 |
| Open Microwave | 62 | 45 |
| Turn Switch | 61 | 59 |
| Move Stapler Pad | 77 | 64 |
| Place Can Basket | 71 | 69 |

这些结果说明较高总体均值不代表每个任务都可靠。不过，表格只给出了任务成功率，没有给出足够的失败轨迹或干预实验，无法据此把失败统一归因于缺少力反馈、接触建模不足或某一种空间能力缺陷。

## 七、真机实验与推理延迟

```{figure} images/FastWAM/fastwam-towel-folding.png
:alt: Galaxea R1 Lite 在真实场景中执行毛巾折叠
:width: 100%
:align: center

**原论文图 3：真机毛巾折叠任务**。作者使用 60 小时遥操作数据，评估可变形物体操作的成功率与完成时间。
```

毛巾折叠既关心最终能否完成，也关心是否反复尝试、调整才完成，因此作者同时比较 success rate 和 average completion time。

```{figure} images/FastWAM/fastwam-real-world-results.png
:alt: 毛巾折叠成功率与平均完成时间散点图，以及不同策略的推理延迟柱状图
:width: 100%
:align: center

**原论文图 4：真机效果与推理延迟**。左图越靠左上越好；右图的延迟在单张 NVIDIA RTX 5090D V2 32GB 上测得。完成时间与单次推理延迟是两个不同指标。
```

论文正文明确给出以下结论：

- 具身预训练的 $\pi_{0.5}$ 在这项真机实验中拥有最高成功率和最短完成时间。
- Fast-WAM 家族中，IDM 的成功率更高，基础模型的完成时间更短。
- 去掉 video co-training 后，成功率降到 10%，完成时间也明显变差。
- 基础模型推理延迟为 190 ms，IDM 为 810 ms，约相差 4.3 倍；图 4 还明确标注 Joint 为 580 ms。

其余散点值可结合原图阅读，这里不将目测估计写成精确实验数字。也不能用“10 步视频去噪 + 10 步动作去噪”推导延迟恰好翻倍：每一步的 token 数量、模块开销和缓存复用都不相同。

**我的理解**：低延迟的价值不仅是缩短一次模型调用，还在于减少动作决策滞后。不过，这篇论文只给出一种真机任务，尚不足以回答高速动态目标、多类接触任务或长程多任务场景中的全部部署问题。

## 八、源码阅读路径与后续版本

### 8.1 从配置到推理，优先核对哪些接口？

公开仓库主要提供 LIBERO 与 RoboTwin 的训练、评估实现；真机毛巾实验的完整数据和部署流程不能由此推定已经公开。源码分析应把数据形状、注意力方向和评估执行前缀连接起来，以下链接固定到本篇核对的版本：

| 阅读入口 | 重点 |
| --- | --- |
| [基础模型配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/model/fastwam.yaml) | 专家维度、scheduler、损失权重与首帧掩码 |
| [数据集采样与相机拼接](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/datasets/lerobot/robot_video_dataset.py#L50-L71) | 33 步窗口、图像子采样、相机布局与动作长度 |
| [基础模型实现](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py#L397-L418) | 掩码、首状态条件、损失有效性处理与推理循环 |
| [MoT 与 K/V cache](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L474-L526) | 按层 prefill，动作去噪时复用视频 K/V |
| [Joint 掩码实现](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam_joint.py#L29-L49) | action 读取全部 video，video 仍不读取 action |
| [动作骨干预处理](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/scripts/preprocess_action_dit_backbone.py) | Wan 权重如何插值为动作专家初始化 |
| [LIBERO 评估配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/sim_libero.yaml) | 动作执行前缀、采样参数与评估任务 |
| [RoboTwin 评估配置](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/configs/sim_robotwin.yaml) | 双臂评估参数和重规划配置 |

同一模型名并不保证相同实验设置。权重应与对应的 dataset statistics、图像配置、动作归一化和采样 scheduler 配套使用；论文中的训练 steps 也不能直接当作后续配置文件的当前默认值。

### 8.2 后续加速结果与论文延迟分开看

官方仓库后续报告端到端推理加速，包括文本编码和 VAE 编码：H20 从 470 ms 降至 210 ms，RTX 4090 从 190 ms 降至 110 ms。LIBERO 默认启用编译后的 action inference 路径。这里的硬件和实现与论文中的 RTX 5090D V2 测量不同，不能把两个“190 ms”混成同一次实验。

后续版本还支持 LeRobot 3.0 数据格式，保留 2.1 支持；动作 scheduler shift 默认改为 1.0，原始发布 checkpoint 的评估说明仍要求使用 5.0 来复现原设定。这些是发布实现的变化，不是论文表格的新测量值。（[固定版本 README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)）

### 8.3 Optional IDM：同一权重的两种推理模式

后续 Optional IDM checkpoint 支持：

- `idm`：先生成未来视频，再生成动作。
- `first_frame`：跳过未来想象，只使用当前观测。

官方报告该 checkpoint 在完整 LIBERO 评估中的平均成功率分别为 **98.55%** 和 **97.75%**，每个任务评估 50 episodes，动作 scheduler shift 为 1.0。这是同一后续模型的模式切换，与原论文中分别训练的 Fast-WAM-IDM 98.0% 和 Fast-WAM 97.6% 不是同一组结果。（[发布权重说明](https://huggingface.co/yuanty/fastwam)）

**我的理解**：同一权重切换推理模式，有助于进一步研究“哪些输入值得付出未来生成的成本”。但能切换不代表已经实现自动选择；如何预测何时想象有益，仍是另一个问题。

## 九、我的理解与仍然开放的问题

### 9.1 与 GigaWorld-Policy 的共同点和区别

Fast-WAM 与 [GigaWorld-Policy](GigaWorld-Policy-paper.md) 都强调训练视频、部署时直接生成动作，但内部连接不同：Fast-WAM 使用独立的视频与动作专家，以 mixed attention 连接；GigaWorld-Policy 的论文设计采用共享 DiT，并允许未来视觉读取动作条件。

Fast-WAM 基础配置中的视频分支不读取动作 tokens，因此不能把它的未来视频目标直接描述为显式的动作条件动力学模型。两种设计都可能从视觉监督中获益，但应具体追踪哪组 token 读取哪组信息，而不是仅凭“WAM”名称推断结构。

### 9.2 这组结果改变了怎样的设计判断？

对我而言，最有用的结论是：为策略加入辅助世界建模时，应同时设计一个可独立运行的动作路径，并测量辅助目标是否真的改善策略。不能只因为训练时生成了更丰富的输出，就默认部署时也应支付同样的计算成本。

同时，当前结果还留下几个值得追问的问题：

1. **更长时间尺度**：当任务需要在多个动作块之外比较候选未来时，显式想象的价值是否会增大？单块生成的对照不能覆盖所有规划方式。
2. **数据与骨干规模**：换用其他视频模型，或增加具身预训练数据后，视频监督与推理想象之间的收益比例是否相同？
3. **层间交互**：30 层都进行专家间注意力是否必要？更少的交互层能否保持性能并进一步降低开销？论文没有给出对应消融。
4. **真实执行成本**：应把成功率、完成时间、推理延迟和闭环重规划一起评估，而不是仅追求较低的单次调用耗时。
5. **失败机制**：困难任务需要结合执行轨迹、接触过程与观测误差分析，单靠每任务成功率无法确定根因。

这些是我基于论文提出的问题，不是作者已经验证的结论。Fast-WAM 提供的证据支持在所测设置下减少不必要的未来视频采样，同时保留视频监督；它没有给出所有 WAM、所有任务都应放弃未来想象的普遍答案。

## 参考资料

- [Fast-WAM 论文 v2](https://arxiv.org/html/2603.16666v2)：方法、实验与图表。
- [官方项目页](https://yuantianyuan01.github.io/FastWAM/)：方法概览与真机展示。
- [官方代码固定版本](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)：数据、模型及评估实现。
- [发布权重与模型说明](https://huggingface.co/yuanty/fastwam)：checkpoint、配套统计量与 Optional IDM 结果。

本页图 1–4 均截取自原论文 v2，版权归原作者，原论文以 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 发布；中文图注与分析由本笔记整理。
