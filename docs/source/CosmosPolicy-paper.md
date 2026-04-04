# Cosmos Policy: Fine-Tuning Video Models For Visuomotor Control And Planning

![](../_images/CosmosPolicy/CosmosPolicy-1.png)

**ABSTRACT & 1 INTRODUCTION**

【大前提】

video generation models $\Longrightarrow$ 通过数百万视频学习 temporal causality 时间因果关系 / implicit physics 隐性物理规律 / motion patterns 运动模式 $\Longrightarrow$ 复杂物理交互 / 动态时变场景 能力 up

【启发】

利用视频生成模型的 spatiotemporal priors 时空先验 for robotics policy

【现有问题】

- <font color=red>训练多 post-training（例如：视频微调后进行动作模块训练）</font>
- <font color=red>设计动作生成的新架构组件（例如：分离 action 扩散器 / 逆动力学模型）</font>
- <font color=blue>尝试训练 video-action-model 但是自身设计架构导致无法利用预训练 video model 的时空先验能力</font>

【解决方案】

<font color=green>将大型预训练视频模型 Cosmos-Predict2-2B 通过在**目标具身实体**上收集的机器人演示数据进行**单阶段 post-training** ，无需架构修改，即可适配为有效的机器人策略。</font>

- 作者提出 <font color=red>"预训练 video model 的学习算法本身能非常适合与其他模态并行表征动作"</font> 假设 $\Longrightarrow$ 直接在视频生成模型的 <u>latent diffusion 过程中</u>生成 <u>latent frames 的机器人动作编码</u> $\Longrightarrow$ 利用视频生成模型的**预训练先验**和**核心学习算法**来捕捉复杂的动作分布

- 生成<font color=green>未来状态 / 图像 / 富有含义的数值（例如 values: 预期累积奖励）</font>，这些同样以 latent frames 形式编码，从而在测试时规划具有更高成功概率的动作轨迹

- <font color=blue>可整合 policy rollout 数据以优化其世界模型，并实现更有效的规划 / model-based planning</font>

  在进行未来状态与价值预测时，Cosmos Policy 可采用 best-of-N sampling 策略：<font color=red>通过生成候选动作 $\longrightarrow$ 模拟其可能产生的未来状态 $\longrightarrow$ 按预测价值对这些状态进行排序 $\longrightarrow$ 执行价值最高的动作。</font>

【测评】

1. <font color=brown>在没有 model-based planning 下</font>: 

   Libero $98.5\%$ / RoboCasa $67.1\%$ / 具有挑战性的真实世界 bimanual 操作任务最高成功率 $93.6\%$

2. <font color=brown>在 model-based planning 夹持下</font>: 

   在两项具有挑战性的现实操作任务中，观察到平均任务完成率提高了$12.5\%$ 

3. 将 model-based 的 planning 方法与 model-free 变体进行比较

【baseline】

模仿学习 dp // video policy: UVA / Video Policy // 在下游演示数据微调的 SOTA vlas

【结论】

<font color=red>CosmosPolicy 能够通过经验学习来优化其**世界模型**与**价值函数**，并利用 model-based 的 planning 策略，在具有挑战性的任务中实现更高的成功率。</font>

**2 RELATED WORK**

**Video-based robot policies** 

**Vision-language-action models**

**World models and value functions**

world model 在强化学习上的经典工作：TD-MPC / Dreamer 系列

world model 偏现代性 / 规模化工作：

1. FLARE $\longrightarrow$ 向 Diffusion Transformer 序列添加可学习的未来 tokens ，以预测未来状态的紧凑表征
2. SAILOR $\longrightarrow$ 采用分离的世界模型和奖励模型，通过 MPPI 规划迭代搜索更优动作并优化基础策略
3. Latent Policy Steering $\longrightarrow$ 使用 optical flow 光流预训练一个 world model 作为跨具身本体 / embodiment-agnostic 动作表征模型，随后训练一个独立的价值函数以引导策略向奖励更高的状态发展

**3 PRELIMINARIES**

**Cosmos video model.**

Cosmos-Predict2-2B-Video2World: <font color=red>一种潜在视频扩散模型，该模型以**起始图像**和**文本描述**作为输入，将图像模态输入用 Wan2.1 spatiotemporal VAE tokenizer 编码成连续数值型 tokens, 同时将文本模态输入编码成 T5-XXL embeddings 作为 condition, 在 EDM 去噪评分匹配公式预训练下，能够预测后续帧以生成短视频</font> $\Longrightarrow$ <font color=blue>视频生成模型的输入范式<u>本身没有完美地 match</u> VLA 模型输入范式：缺少 proprioception 模态输入数据；同时没有 multi-view 多视角输入</font>
$$
\mathcal{L}(D_\theta,\sigma)=\mathbb{E}_{\mathbf{x}_0,\mathbf{c},\mathbf{n}}\left[\left\|\underbrace{D_\theta}_{\text{DiT}}(\underbrace{\mathbf{x}_0}_{图像编码连续\text{tokens}}+\underbrace{\mathbf{n}}_{标准正态分布};\underbrace{\sigma}_{噪声水平},\underbrace{\mathbf{c}}_{文本输入\text{embeddings}})-\mathbf{x}_0\right\|_2^2\right]
$$
内部实现上：

- 通过交叉注意力机制对 $\mathbf{c}$ 进行 $D_{\theta}$ 条件处理

  通过 adaptive layer normalization 自适应层归一化对 $\sigma$ 进行条件处理

- Wan2.1 tokenizer: $(1+T)\times H\times W\times 3$ 输入视频数据维度 $\longrightarrow$ $(1+\frac{T}{4})\times\frac{H}{8}\times\frac{W}{8}\times 16$ 输出维度

  <font color=red>首帧图像不进行时间压缩处理</font>，以便对单个输入图像进行条件化。训练过程中采用条件化 mask ，确保与输入图像对应的首帧潜在图像保持无噪声状态，而后续帧则被添加噪声。

**MDP formulation and imitation learning.**

*finite-horizon* Markov Decision Process: $\langle S,A,T,R,H\rangle$ 

稀疏奖励：只有 terminal reward $R(s_H,a_H)\in[0,1]$, 其余都是 0.0

**World models and value functions.**

- 世界模型能够根据当前状态和动作预测未来状态，从而近似真实环境的动态变化。

- 价值函数：
  $$
  V^\pi(s)=\mathbb{E}_{\tau\thicksim\pi}\left[\sum_{k=t}^H\gamma^{k-t}R(s_k,a_k)\mid s_t=s\right]\mathbb{E}_{\tau\thicksim\pi}\left[\gamma^{H-t}R(s_H,a_H)\mid s_t=s\right]\quad\text{结合稀疏奖励设置}
  $$
  仅采用 Monte Carlo returns 蒙特卡洛回报: <font color=red>将每次 rollout 中的状态转移标注成为观测到的回报 $\gamma^{H-t}R(s_H,a_H)$</font>.

**4 COSMOS POLICY: ADAPTING VIDEO MODEL FOR CONTROL & PLANNING**

**4.1 LATENT FRAME INJECTION: INCORPORATING NEW MODALITIES**

![](../_images/CosmosPolicy/CosmosPolicy-2.png)

> ====> 由于视频模型的分词方案，序列开头会设置一个 *"black image for placeholder"占位符图像*：首张图像采用独立编码，其余图像则按四张一组进行时间压缩。
>
> ====> 由于希望为<u>每种模态</u>和<u>每个摄像机视角</u>生成一个潜在帧，因此构建了每张图像的四个相同副本，如顶部行所示。每组四个相同图像对应单个时间步长，而非四个时间步长。

<font color=green>对于一个 $(1+\frac{T}{4})\times\frac{H}{8}\times\frac{W}{8}\times 16$ 的 latent frames 序列，最初对应视频中的图像，通过在现有图像 latent frames 之间插入新的 latent frames $\longrightarrow$ 将新增的机器人状态 / 动作块 / 状态值模态 / 来自其他摄像机视角的图像进行交错排列</font>

"To encode the new modalities as latent frames, we fill each $H^{\prime}\times W^{\prime}\times C^{\prime}$ latent volume with normalized and duplicated copies of the robot proprioception, action chunk, or value (where normalization simply consists of rescaling to $[−1,+1]$)."

$\longrightarrow$ <font color=red>将一些新的 / 非图像模态的数据编码以兼容 latent frames 序列的做法：(1) 将数据归一化到 $[−1,+1]$ 范围；然后（2）通过复制的方式进行填充，得到 $H^{\prime}\times W^{\prime}\times C^{\prime}$ 浅层张量.</font>

$\longrightarrow$ 该序列中模态的排序表示为 $(s,a,s^{\prime},V(s^{\prime}))$ 允许从左至右对动作 / 未来状态 / 未来状态值进行自回归解码

- 复制方法：

  【动作块】对于 $K=25$ 且 $d_{act}=14$ 的 action chunk, 原本张量维度是 $(K\times d_{act})$ 也就是 $(25, 14)$ 这是一个二维张量；将这个二维张量展平成一维向量 $(K\times d_{act}, )$ 也就是 $(350, )$ 维度的张量。这个张量被复制 $\frac{(H^{\prime}\times W^{\prime}\times C^{\prime})}{(K\times d_{act})}$ 次，例如对于 $(H^{\prime}\times W^{\prime}\times C^{\prime})$ 是 $(32, 32, 16)$ 的数据维度，那么这个一维展平的 action chunk 张量被复制 $\frac{(H^{\prime}\times W^{\prime}\times C^{\prime})}{(K\times d_{act})}=\frac{32\times 32\times 16}{25\times 14}=\frac{32\times 32\times 16}{350}=46.8114=47$ 次，然后重新 reshape 这个张量到 $(H^{\prime}\times W^{\prime}\times C^{\prime})$ 维度。

  【本体数据】与动作块类似，只是可能需要复制更多次

  【未来价值】因为价值本身是一个浮点数，直接元素级复制就行

---

该序列包含 11 个 latent frames, 满足 $(s,a,s^{\prime},V(s^{\prime}))$ 序列便于模型从左至右自回归生成：占位符 +

- 在 $t$ 时刻的状态 $s$ :【机器人本体（如末端执行器姿态或关节角度）+ 腕部视角图像 + 第一第三人称摄像头图像 + 第二第三人称摄像头图像】
- 长度为 $K$ 的动作块 $a$ :【动作块】
- 在 $t+K$ 时刻的状态 $s^{\prime}$ :【未来机器人本体感觉 + 未来手腕摄像头图像 + 未来第一第三人称摄像头图像 + 未来第二第三人称摄像头图像】
- 价值 $V(s^{\prime})$: 【未来状态值】

**4.2 JOINT TRAINING OF POLICY, WORLD MODEL, & VALUE FUNCTION**

**Implementing joint training objectives.**

联合训练三个优化目标：

1. $p(a,s^{\prime},V(s^{\prime})|s)$: 演示轨迹，占 data batch $50\%$

2. $p(s^{\prime},V(s^{\prime})|s,a)$: rollout 轨迹，占 data batch $25\%$

3. $p(V(s^{\prime})|s,a,s^{\prime})$: rollout 轨迹，占 data batch $25\%$

   可通过输入掩码选择将价值生成限定于 $(s,a,s^{\prime})$ 的子集 $\longrightarrow$ 状态值 $V(s^{\prime})$ / 状态-动作值 $Q(s,a)$ 

4. 这与常规的 BC 目标 / 世界模型目标存在差异，但是作者在实验验证额外的辅助目标效果更好

Rollout 轨迹数据集仅仅是演示数据集的超集，其中若存在失败的演示 (LIBERO / RoboCasa) 则还包含失败演示；在真实世界 Aloha 环境中，不存在失败的演示案例；在此情况下，演示数据集与 rollout 数据集具有等效性。

**Parallel vs. autoregressive decoding.**

- 并行解码可提升速度 $\longrightarrow$ direct policy evaluation without planning 
- 自回归解码可能提供更高质量的预测 $\longrightarrow$ evaluations with planning

**4.3 PLANNING WITH COSMOS POLICY'S WORLD MODEL AND VALUE FUNCTION**

仅通过演示进行训练不足以实现有效规划，因为数据仅涵盖成功结果 $\Longrightarrow$ 世界模型与价值函数只能观察到狭窄的状态-动作分布 $\Longrightarrow$ 难以超越该分布进行泛化 $\Longrightarrow$ 收集策略 rollout 数据并从这些经验中学习

**Learning from rollout experiences.**

base policy $\longrightarrow$ BC 训练后保留一份 checkpoint (policy model) $\longrightarrow$ post-training 再保留一份 checkpoint (planning model)

rollout dataset 收集：多样化初始条件 / 记录轨迹 / 记录成功-失败结果 $\Longrightarrow$ 大部分 rollouts 用于：

- 训练 world model: 用更多 / 更杂的数据分布来建模更本质的 world knowledge
- 训练 values: 使得 model 能从成功和失败中推理出更正确的 values

小部分 rollouts 用于训练 policy: 本身 BC 过的才部署，不必必须大计算量地继续 BC 了

**Model-based planning.** best-of-N 采样策略

1. 从 policy model 策略中抽取多个 action 方案

2. 利用 planning model 预测每个 action 方案的未来状态及价值

   每个 action 方案执行时调用世界模型 3 次，生成 3 个未来状态 // 每个未来状态调用价值函数 5 次，生成 3 个未来状态价值 $\longrightarrow$ 为每个 action 方案生成 15 个价值预测值

3. 选择并实施能实现预测价值最高的状态的 action 方案

4. 为加速搜索过程，工程采用并行推理技术，使用 $N$ 个 GPUs 进行 best-of-N 抽样，执行完整的动作块（而非仅执行部分动作块，如渐进式控制中所采用的方法）

**4.4 Inference**

在推理阶段，Cosmos Policy 会生成经过**去噪处理**的 latent frames $\Longrightarrow$ 逆向执行上述潜在注入过程

计算 latent frames 中所有 $\frac{(H^{\prime}\times W^{\prime}\times C^{\prime})}{(K\times d_{act})}$ 个副本的动作块平均值 $\Longrightarrow$ 反归一化至原始尺度 $\Longrightarrow$ 部署

提取数值 $\Longrightarrow$ 对整个潜在空间取平均值 $\Longrightarrow$ 反归一化至原始范围 $[0,+1]$

**5 EXPERIMENTS**

**5.1 EXPERIMENTAL SETUP**

**Libero** 过滤版用于 policy learning // 完整版用于 world model 建模和 value function 训练

**Robocasa** 针对每个任务，通过**五个不同平面布局和风格**的评估场景，每个场景 10 次试验，进行 50 次试验以评估成功率，并在 3 个随机种子下计算所有 24 个任务的平均成功率，总计 3600 次试验。50 条人类遥操作轨迹，与 Libero 同样数据过滤版 / 完整版做法。

**Real-world ALOHA robot tasks** 

![](../_images/CosmosPolicy/CosmosPolicy-3.png)

| Dataset / Benchmark | Gradient Steps | GPUs (Type) | Global Batch Size | Training Time | Action Chunk Size (Predicted) | Action Steps Executed Before Requery | Model Fine-tuning  |
| ------------------- | -------------- | ----------- | ----------------- | ------------- | ----------------------------- | ------------------------------------ | ------------------ |
| LIBERO              | 40K            | 64 × H100   | 1920              | 48 hours      | 16                            | 16                                   | Full (all weights) |
| RoboCasa            | 45K            | 32 × H100   | 800               | 48 hours      | 32                            | 16                                   | Full (all weights) |
| ALOHA               | 50K            | 8 × H100    | 200               | 48 hours      | 50                            | 50                                   | Full (all weights) |

| Method                | Training Strategy  | GPUs     | Training Time | Gradient Steps   | Batch Size          | Notes                  |
| --------------------- | ------------------ | -------- | ------------- | ---------------- | ------------------- | ---------------------- |
| Cosmos Policy         | Fine-tuning        | 8 × H100 | 48 hours      | 50K              | 200                 | Action chunk = 50      |
| $\pi_{0.5}$ / $\pi_0$ | Fine-tuning        | 8 × H100 | 48 hours      | 400K             | 256                 | Faster iteration speed |
| OpenVLA-OFT+          | Fine-tuning        | 8 × H100 | 48 hours      | 32K              | 96 (grad. acc. = 4) | Slower iteration speed |
| Diffusion Policy      | Train from scratch | 1 × H100 | 48 hours      | 72K (190 epochs) | 350                 | ~150M params           |

**5.2 COMPARING AGAINST STATE-OF-THE-ART IMITATION POLICIES WITHOUT PLANNING**

> Q1: Cosmos Policy 作为直接策略使用时，与最先进的模仿学习策略相比表现如何？

![](../_images/CosmosPolicy/CosmosPolicy-4.png)

"OpenVLA-OFT+ often reaches in between two candies rather than directly going for one; we hypothesize that its L1 regression of actions leads to inaccurate modeling of the action distribution in tasks with high multimodality." $\longrightarrow$ 在存在多个同样合理但彼此差异较大的动作选择时，用 L1 回归学习确定性动作会不可避免地产生 "均值化" 行为，导致机器人执行一个在物理上无效的折中动作（例如伸向两颗糖之间），从而无法正确建模高多模态的动作分布。

> Q2: Cosmos Policy 的不同组件有多重要？

$\longrightarrow$ 添加 mask 辅助学习目标消融 / 替换权重进行预训练权重消融

![](../_images/CosmosPolicy/CosmosPolicy-5.png)

Real-world ALOHA robot + 从头训练 Cosmos Policy: "折叠衬衫" 任务中平均得分 80.8 分，比完整版 Cosmos Policy 低 18.7 分。定性分析显示，从头训练的变体存在抖动动作，长期使用可能损伤机器人，因此我们终止了对该变体的进一步评估。

**5.3 EVALUATIONS OF COSMOS POLICY WITH MODEL-BASED PLANNING**

> Q3: Cosmos Policy 能否通过 rollouts 经验，建立精确的世界模型和价值函数，从而实现高效规划？

$\longrightarrow$ with / without planning $\longrightarrow$ 基于 648 rollout datasets 对 base Cosmos Policy 检查点进行微调

> Q4: 使用世界模型和状态价值函数进行搜索是否比使用 model-free Q 值函数更有效？

$\longrightarrow$ 不同的 planning 变体：$V(s^{\prime})$ mask $(s,a)$ 需要 world model 来预测 $s^{\prime}$ // $Q(s,a)$ mask $(s^{\prime})$ 直接使用已知的 $(s,a)$ 预测 Q 值 $\longrightarrow$ 类似 model-free 做法

![](../_images/CosmosPolicy/CosmosPolicy-6.png)

- for Q3: Model-based $V(s^{\prime})$ 比直接原 model 效果略微更好。
- for Q4: Model-based $V(s^{\prime})$ 具有更优的性能表现 $\longrightarrow$ 有效利用学习到的环境动力学特性，从而实现更高效且样本效率更高的规划。Model-free $Q(s,a)$ 效果表现略差 $\longrightarrow$ rollouts 数据量有限，难以准确学习 $Q$ 函数，并怀疑该模型可能因输入维度较高而出现过拟合现象。

**6 DISCUSSION**

**Limitations and future work:**

- substantially lower inference speed when using model-based planning

- 有效的规划需要大量 rollouts 数据以实现超越 demos 分布的精确预测。

  从较少的 rollouts 数据中学习将提高方法的可及性。

- best-of-N planning with one layer in the search tree

- 世界模型预测范围及规划深度
