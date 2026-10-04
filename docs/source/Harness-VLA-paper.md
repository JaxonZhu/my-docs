# Harness VLA 论文与源码解读：技能编排

论文原题：Harness VLA: Steering Frozen VLAs into Reliable Manipulation Primitives via Memory-Guided Agents

- 作者：Yixian Zhang、Huanming Zhang 等，通讯作者 Wenbo Ding、Chao Yu
- 团队：清华大学、Striding AI、普渡大学、中科院自动化所等
- 论文：[arXiv:2607.08448](https://arxiv.org/abs/2607.08448)，2026 年 7 月 9 日首次提交
- 项目：[Harness VLA](https://harnessvla.github.io/)
- 代码：[RLinf/RPent](https://github.com/RLinf/RPent)

Harness VLA 把冻结的 VLA 封装成可重试的局部操作原语：高层 planner 负责理解目标、重新定位、调整机器人姿态和组合任务，VLA 负责抓取、放置或操作夹具等接触环节。任务记忆保存已经验证过的原语调用结构，全局记忆记录成功条件和失败后的处理规则，执行时再根据当前观测重新绑定空间参数。

我最关注的是它如何改变 VLA 的使用方式：同一套权重，从连续执行整段任务的策略，变成由 planner 选择调用时机、检查结果并决定是否重试的工具。论文中，同一冻结后端在 LIBERO-Pro 上的成功率从 50.0% 提升到 82.4%；但这项改进包含 planner、记忆、感知与控制原语的共同作用，需要结合评估协议理解。

:::{note}
**阅读范围**：正文依据原始 PDF 笔记，并以 [2026 年 7 月 15 日的 v3](https://arxiv.org/html/2607.08448v3) 核对方法、主实验及图表编号。后续 [v5](https://arxiv.org/html/2607.08448v5) 新增 Flash Mode、真机演示与 planner 对照，并调整 RoboCasa 评估口径，文末单独说明。源码核对固定在 2026 年 10 月 4 日的 RPent 提交 `068cd64f`，不代表已完成训练、仿真或真机复现。
:::

## 一、为什么把 VLA 变成一个可调用的原语？

### 1.1 局部动作正确，任务仍可能做错

一个 VLA 可能已经学会抓住牛奶盒，但在目标容器改变、物体位置交换或任务顺序变化时，仍沿用训练中的动作习惯。此时问题不一定出在抓取动作本身，而可能是选错了对象、接近位置不合适，或在长任务的错误阶段调用了该技能。

论文中的冻结 $\pi_{\mathrm{RLinf}}$ 在标准 LIBERO 上达到 95.3%，在 LIBERO-Pro 扰动下则为 50.0%。作者据此研究如何通过外部编排复用已有的局部操作能力。（论文 §1、§3.2）

这是对所测试模型和扰动的观察，不能推广成“所有 VLA 都没有语义推理或长期规划能力”。同样，作者没有通过这里的实验定位出一个普遍缺失的“因果模块”。

### 1.2 高层推理与接触控制的分工

依靠语言模型调用运动 API，也有自己的难点。自由空间中的移动、腕部旋转或基座导航可以交给已有控制器；但不规则物体抓取、受限放置和关节物体操作通常更依赖学习到的视觉运动经验。

Harness VLA 为这两类能力分配了不同职责：

| 组件 | 主要职责 | 决策粒度 |
| --- | --- | --- |
| Agentic planner | 目标绑定、空间定位、任务顺序、失败恢复 | 每次原语调用 |
| 分析性原语 | 接近、运输、姿态调整、导航与释放 | 控制器内部执行 |
| 冻结 VLA 原语 | 局部抓取、接触与操作 | 动作 chunks |
| Harness | 执行接口、观测刷新、记录与预算管理 | 运行时协议 |

所谓“分析性原语”并不要求每一个底层控制器都使用同一种 IK 算法。论文附录 B 允许操作空间伺服、Jacobian 控制或 IK 等实现，只要对外接口保持相应语义。

```{figure} images/Harness-VLA/harness-vla-overview.png
:alt: Harness VLA 中的 planner 结合任务记忆与全局记忆，通过统一接口调用分析性原语和冻结 VLA
:width: 100%
:align: center

**原论文图 1：系统总览**。Planner 读取任务和观测，再选择原语；VLA 作为局部操作工具参与整个执行过程。
```

### 1.3 先构造合适条件，再调用技能

作者将 staging 作为重要步骤：planner 先从当前图像中确认目标，再用分析性原语把机器人调整到合适的接近位置或视角，然后调用 VLA。若接触失败，planner 可以重新定位、调整姿态，再进行一次局部尝试。

**我的理解**：部署扰动不一定都要由 VLA 权重吸收。Planner 可以主动改变技能的输入条件，把一个整体上陌生的任务拆成若干较熟悉的局部操作。这种分解是否有效，取决于局部技能本来会什么，以及高层能否判断和创造适当的调用条件。

```{figure} images/Harness-VLA/harness-vla-composition.png
:alt: 分析性原语连接多个适合 VLA 执行的局部状态区域，从而完成超出原训练轨迹分布的任务组合
:width: 100%
:align: center

**原论文图 2：原语组合的概念图**。图示解释作者的设计直觉，并不构成对训练分布覆盖范围的形式化证明。
```

## 二、Harness 如何组织执行闭环？

### 2.1 每次选择一个原语

论文将观测写为：

$$
o_t=(I_t^{\mathrm{rgb}},I_t^{\mathrm d},q_t),
$$

其中 RGB 提供语义与外观，深度提供度量几何，$q_t$ 包含机器人末端位姿和夹爪状态。给定任务语言 $\ell$、任务记忆 $M_{\mathrm{task}}$ 与全局记忆 $M_{\mathrm{global}}$，可以把高层过程概括为：

$$
c_t=\Pi(o_t,\ell,M_{\mathrm{task}},M_{\mathrm{global}}),
\qquad c_t\in\mathcal P.
$$

这个公式是对论文接口的整理：$c_t$ 是带参数的原语调用，不是底层关节力矩。Harness 执行原语，返回更新后的观测和日志，planner 再决定下一步，直到任务完成或预算耗尽。（论文 §2.1–2.2）

例如，一次拿取失败后，planner 可以重新观察物体，先改变接近位置，再调用 VLA；已经稳定拿住物体后，则可转用 `move_to` 和 `release` 完成运输与释放。每个步骤都要以执行结果为依据。

### 2.2 文件式 REPL 的接口契约

论文附录 A 描述 file-mediated REPL：环境 worker 持有仿真状态，planner 通过命令文件与观测文件交互。

| 文件或记录 | 含义 |
| --- | --- |
| `command.json` | 本轮原语名及参数 |
| `state_NN.json` | 任务语言、本体状态和成功信号 |
| RGB-D 与 world maps | 语义识别和空间定位证据 |
| `log_NN.json` | 命令状态、步数和可用的失败信息 |
| `done_NN.flag` 等 | 原语执行结束的同步信号 |

执行顺序为“发出命令 → 等待完成 → 读取新观测 → 再决策”。论文的 planner 不读取物体真实位姿或仿真器内部状态；用于定位的 world map 来自图像、深度与相机几何，并非直接提供物体的真值坐标。

附录提示 planner 在物体表面选取多个稳定像素，用中位数等统计量估计位置，并在相机、基座、物体或抓取状态改变后重新定位。反射、孔洞、边缘和背景像素仍可能带来错误。

### 2.3 原语结束不等于任务成功

`vla_act` 的局部停止条件可能是物体已抬起、接触状态变化、任务 predicate 已触发，或者调用预算耗尽。局部返回只表示控制权交回 planner；最终成功仍由 benchmark 的任务完成条件决定。

因此，“夹爪闭合”不等于“抓到了物体”，“物体靠近容器”也不等于“任务已经完成”。论文的状态与日志协议让 planner 在每次原语后检查结果；这与低层连续控制处于不同时间尺度。

## 三、记忆保存了什么，又如何迁移？

### 3.1 Task Specific Memory 与 Global Memory

| 记忆 | 保存内容 | 执行时的用途 |
| --- | --- | --- |
| 任务级 JSONL trace | 原语顺序、参数与调用位置 | 恢复解题步骤和技能切换点 |
| 任务级 JSON audit | 结果、策略、恢复决策与失败记录 | 理解参考解为何有效 |
| Global Memory | 跨任务成功规则和失败模式 | 判断何时调用、检查或重试 |

任务级记忆可以记录“调用 VLA 建立抓取 → 运输到目标 → 释放”的结构。全局记忆则可以提示：如果夹爪闭合但物体没有随末端移动，应按空抓处理，重新定位后再尝试。

这里的 learning 主要是把交互经验写入外部记忆，再作为上下文影响后续决策。Harness 编排阶段不通过任务奖励更新 planner 或冻结 VLA 的权重。

### 3.2 迁移的是步骤结构

参考轨迹中的坐标属于参考场景。部署时应重新识别当前对象、目标表面及夹具，再绑定每一步的空间参数。

正文强调把具体坐标参数化为感知查询；附录的示例 trace 仍包含数字坐标，但同时明确这些只能作为参考场景绑定，不能照搬。两种表述共同指向的要求是：**复用调用结构，重新计算空间参数**，而不是要求所有 JSONL 文件必须采用某一种符号表达格式。（论文 §2.2、附录 A、E.3）

**我的理解**：这比回放一条轨迹多了一层条件判断。记忆给出曾经成功的组织方式，当前观测决定它是否适用，以及应该在哪里执行；发生局部失败时，planner 仍需修改后续动作。

### 3.3 探索阶段与评估阶段

LIBERO、LIBERO-Pro 和 RoboCasa365 使用一个参考 seed 构造任务记忆。探索时允许 reset，时间预算较宽；成功后保存 trace 和总结。正式评估改用其他初始状态，禁止 reset，并受到更严格的预算约束。

“单次评估尝试”不表示 VLA 只能调用一次。同一个 episode 内可以重新定位、调整姿态和重试局部接触，只是不能通过重置整个环境重新开始。

还有两种不同的 zero-shot 设置需要分开：

- **LIBERO-Pro Goal 无记忆评估**：不检索目标设置的 Task Specific Memory，也不检索对应的 Global Memory。
- **RoboTwin C2R**：保留从 clean 设置获得的任务记忆，直接迁移到 randomized 设置；不在 randomized 设置进行额外探索、适配或微调。

因此，“冻结权重”“没有目标设置探索”和“完全不使用记忆”是三个不同条件，不能都简称为同一种零样本能力。

## 四、固定原语库与 VLA 后端

### 4.1 六个分析性原语加一个 VLA 原语

论文共享操作词汇表包含 **6 个分析性原语和 1 个 VLA 原语**。RoboCasa365 另增加 2 个移动基座原语；用于探索的 reset 不计入操作原语数量。（论文表 1、附录 B）

| 原语 | 作用 |
| --- | --- |
| `move_to` | 将末端移动到世界坐标目标 |
| `move_pose` | 同时改变位置和部分姿态变量 |
| `rotate_wrist` | 设置腕部 yaw |
| `rotate_pitch` | 设置腕部 pitch |
| `set_gripper` | 控制夹爪开合 |
| `release` | 按释放条件打开夹爪 |
| `vla_act` | 调用冻结 VLA 执行局部接触操作 |
| `navigate_to` | RoboCasa 的世界坐标基座导航 |
| `move_base` | RoboCasa 的局部基座速度控制 |

词汇统一不代表所有环境暴露完全相同的接口。例如 `move_pose` 在 LIBERO 中直接提供，在 RoboCasa 中可由旋转与移动组合表达。RoboTwin 通过 `arm` 参数指定左臂、右臂或双臂模式，交接任务由已有原语组合实现，并未额外定义一个独立的 handover 原语。

### 4.2 `vla_act` 是有边界的局部尝试

Planner 为 VLA 指定任务条件 prompt、最大 chunk 数和停止条件 $\tau$。冻结后端根据实时相机与本体观测产生动作，直到局部停止条件满足或预算耗尽。

以下是论文接口风格的示例，说明“调用什么”与“何时返回”；它不是对当前 RPent 所有环境都通用的可直接运行命令：

```json
{
  "action": "vla_act",
  "prompt": "grasp the black bowl",
  "max_chunks": 30,
  "stop": "object_lifted"
}
```

一次 `vla_act` 可以包含多个动作 chunks，后端可以在 chunk 间重新读取观测。高层 planner 则通常要等原语返回后才能重新介入。因此，planner 层的反馈间隙不等于整个 VLA burst 都使用同一张图像，也不能把一次原语调用等同于一个 chunk。

### 4.3 不同 benchmark 使用不同冻结模型

| Benchmark | 冻结后端 | 评估前的来源 |
| --- | --- | --- |
| LIBERO / LIBERO-Pro | $\pi_{\mathrm{RLinf}}$ | RLinf 的 `pi05_libero130_fullshot`，基于 $\pi_{0.5}$ 的 SFT checkpoint |
| RoboCasa365 | RLDX-1 | 官方 RoboCasa checkpoint |
| RoboTwin C2R | LingBot-VLA | 论文团队在 RoboTwin 上后训练的 checkpoint |

同一个 benchmark 内，直接 VLA baseline 与 Harness VLA 复用相同后端；跨 benchmark 则不是同一套权重。这支持“统一原语抽象可适配多种后端”的结论，不表示一个 VLA 已经零样本跨越了所有机器人形态。

原笔记保存了 LingBot-VLA 的训练配置，需要将它放在正确的时间阶段：这是评估前的 RoboTwin 后训练，而非 Harness 编排阶段训练。（论文附录 D.3、表 14）

| 配置项 | 论文报告 |
| --- | --- |
| 优化器 | AdamW，weight decay 0 |
| 学习率 | $10^{-4}$；视觉编码器 $10^{-6}$ |
| 损失 | L1 Flow Matching |
| Chunk / flow steps | 50 / 10 |
| 最大序列长度 | 2048 |
| 最大 action / state 维度 | 75 / 75 |
| Global batch size | 256 |
| 图像与相机 | 224×224；top、左腕、右腕 |
| 精度与分布式 | bf16/fp32 mixed precision，FSDP2 |

后训练完成后，直接 baseline 和 Harness 评估均冻结该 checkpoint。因此，“Harness 无需额外微调”不能扩展成“所有后端从未进行任务相关训练”。

## 五、评估协议与主要结果

### 5.1 1790 次 rollout 的组成

| Benchmark | 任务数 | 每任务评估 seeds | 报告 rollout 数 |
| --- | ---: | --- | ---: |
| 标准 LIBERO | 40 | 10 | 400 |
| LIBERO-Pro | 80 | 10 | 800 |
| RoboCasa365 target50 | 50 | 10 / 5 / 5 | 340 |
| RoboTwin C2R | 50 | 5 | 250 |
| 合计 | 220 | — | 1790 |

这是论文附录 C 汇总的基准评估规模，不包含参考 seed 的记忆构造过程，也不是把全部方法和消融实验累加后的总运行次数。四类协议不同，不应将它们混成一个没有定义权重的总体成功率。

RoboCasa 的 50 个任务由 18 个 Atomic-Seen、16 个 Composite-Seen 和 16 个 Composite-Unseen 构成。对应评估次数为 $18\times10=180$、$16\times5=80$ 和 $16\times5=80$。Seen / Unseen 指任务模板是否出现在相应预训练集合中，不是说具体场景或轨迹是否完全见过。

RoboTwin C2R 的 clean 参考实例和 randomized 评估实例都经过官方 scripted expert 的可行性验证。论文说明该验证独立于 Harness VLA，不根据 Harness 的成功结果选择 seeds。

### 5.2 标准 LIBERO 保持相近表现

下表保留使用相同冻结后端的直接比较，单位为 %：（论文表 2）

| 方法 | Spatial | Object | Goal | LIBERO-10 | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| $\pi_{\mathrm{RLinf}}$ | 99.0 | 96.0 | 97.0 | 89.0 | 95.3 |
| Harness VLA（CC） | 97.0 | 100.0 | 94.0 | 93.0 | 96.0 |

CC 指 Claude Code planner。96.0% 对应 384/400 次成功；部分套件提高，部分降低，更合适的解读是标准设置下保持有竞争力的表现，而非所有任务都获益。

### 5.3 LIBERO-Pro 的指令与位置扰动

LIBERO-Pro 在四个套件上分别施加两类扰动：T 为 instruction redirection，改变任务指向；S 为 position swap，交换物体位置而保留相应任务要求。每个测试项有 10 个任务、每任务 10 个评估 seeds。（论文表 3）

为便于阅读，将结果按测试项逐行排列：

| 测试项 | 冻结 VLA | RATS | Harness Codex | Harness CC |
| --- | ---: | ---: | ---: | ---: |
| Spatial-T | 42.0 | 31.0 | 81.0 | 94.0 |
| Spatial-S | 59.0 | 29.0 | 69.0 | 80.0 |
| Object-T | 71.0 | 63.0 | 94.0 | 88.0 |
| Object-S | 78.0 | 61.0 | 91.0 | 90.0 |
| Goal-T | 45.0 | 36.0 | 75.0 | 87.0 |
| Goal-S | 42.0 | 43.0 | 66.0 | 87.0 |
| LIBERO-10-T | 49.0 | 未报告 | 52.0 | 71.0 |
| LIBERO-10-S | 14.0 | 未报告 | 49.0 | 62.0 |
| Overall | 50.0 | 43.8（仅 6 项） | 72.1 | 82.4 |

这里的冻结 VLA 是 $\pi_{\mathrm{RLinf}}$。与相同后端直接执行相比，CC 版本提升 **32.4 个百分点**。论文摘要强调的“比 RATS 高 38.6 个百分点”来自 $82.4-43.8$，但两行 Overall 覆盖范围不同：前者为 8 项，后者仅为 6 项，不能作为完全同口径的配对比较。

如果只对齐 RATS 报告的前六项，按表内数值重新计算，Harness CC 均值约为 87.7%，RATS 为 43.8%，差约 43.8 个百分点。这是基于公开表格的重算；不同系统的模型、原语接口及评估来源仍需考虑。

CC 与 Codex 在此基准相差约 10.3 个百分点，但这张表没有隔离空间推理、提示词遵循或工具使用等原因。后面的 RoboCasa 结果中 Codex 反而更高，因此不能据此给两个 planner 排一个跨任务通用的能力顺序。

```{figure} images/Harness-VLA/harness-vla-grounding.png
:alt: 标准任务、扰动后的冻结 VLA 和 Harness VLA 的终态对比，显示目标重新绑定的效果
:width: 100%
:align: center

**原论文图 3：目标重新绑定的示例**。这些终态帧有助于理解方法如何改变任务执行，不能替代对所有失败原因的统计归因。
```

### 5.4 RoboCasa365 按任务数加权

| 方法 | Atomic-Seen | Composite-Seen | Composite-Unseen | 任务加权 Overall |
| --- | ---: | ---: | ---: | ---: |
| RLDX-1 | 60.0 | 21.3 | 5.0 | 30.0 |
| Harness VLA（Codex） | 91.6 | 56.3 | 13.8 | 55.4 |
| Harness VLA（CC） | 79.4 | 47.5 | 15.0 | 48.6 |

来源为论文表 4 与 §3.2。这里 Overall 使用三个分组的任务数作为权重：

$$
S_{\mathrm{overall}}
=\frac{18S_{\mathrm{atomic}}+16S_{\mathrm{seen}}+16S_{\mathrm{unseen}}}{50}.
$$

以 Codex 行为例，代入得到约 55.4%。由于三个分组的每任务评估 seeds 数不同，它不是把 340 次 rollout 直接合并后的成功比例。

Codex 在 Composite-Seen 上相对 RLDX-1 提高 35.0 个百分点，与任务分解和技能组合的设计方向一致；Composite-Unseen 仍只有 13.8%，表明未见组合模板依然困难。仅凭这三个数值，无法单独测出记忆或 planner 对增益的贡献。

### 5.5 RoboTwin C2R 的迁移范围

| 方法 | 成功率（%） |
| --- | ---: |
| LingBot-VLA 直接执行 | 50.4 |
| Harness VLA（Codex） | 58.0 |
| Harness VLA（CC） | 58.4 |

在相同后训练后端上，CC 版本提高 8.0 个百分点。（论文表 6）这里的迁移从 clean trace 出发，不在 randomized 设置重新探索；它不是完全没有任务记忆的测试。

论文也列出 GR00T-N1.7、$\pi_{0.5}$ 和 StarVLA 等外部结果，但它们不是用于隔离编排收益的同后端对照。上表更直接回答“把同一个冻结 VLA 交给 harness 后有什么变化”。

## 六、哪些实验支持这种分工？

### 6.1 记忆对两种扰动的帮助不同

| LIBERO-Pro Goal 设置 | Goal-T | Goal-S |
| --- | ---: | ---: |
| 有任务与全局记忆 | 87.0 | 87.0 |
| 无目标设置记忆 | 79.0 | 31.0 |

论文表 5 的无记忆评估表明：在 Goal-T 上，高层在线推理已经能处理相当一部分语义重新绑定；在 Goal-S 上，去掉记忆后下降更明显。

该结果支持记忆对空间扰动任务有帮助，但因为两类记忆一同移除，无法从这组实验单独判断 Task Specific Memory 与 Global Memory 各自的贡献。“Goal-T 不依赖记忆”也过强，它仍比有记忆配置低 8 个百分点。

### 6.2 重试的价值在于先观察和调整

论文限制每个 episode 允许调用 VLA 的次数，观察累计成功率如何变化。三个基准中，前几次调用带来较明显提升，继续增加调用后逐渐接近完整 harness 的表现。（论文图 4）

```{figure} images/Harness-VLA/harness-vla-invocations.png
:alt: 三个基准中累计成功率随每个 episode 允许的 VLA 调用次数增加而变化的曲线
:width: 100%
:align: center

**原论文图 4：VLA 调用次数上限分析**。蓝色虚线为相应冻结策略基线，灰色虚线为完整 harness 参考线；横轴是原语调用次数，不是动作 chunk 数。图内参考值与主表不完全相同，此处按原图读取，不混用统计范围。
```

这些曲线说明 planner 选择的重复调用具有价值，但并未独立区分重新定位、重新 staging、随机重试和增加总预算的收益。曲线的饱和点也不能直接当作任务“接触密度”的测量值。

```{figure} images/Harness-VLA/harness-vla-retry.png
:alt: LIBERO-Pro 牛奶盒放入篮子和 RoboCasa 平底锅预浸泡任务中，planner 调整位置并重试局部 VLA 操作
:width: 100%
:align: center

**原论文图 5：局部失败后的恢复示例**。Planner 根据未完成的接触或放置结果调整机器人，再调用 VLA；重复调用被嵌入任务步骤，而非无条件地重放同一个动作。
```

### 6.3 最后一步归谁完成，不等于全部贡献归谁

论文将成功 rollout 按“哪个原语触发最终任务 predicate”分类。LIBERO-Pro 常在 VLA 建立抓取后，由分析性运输或释放完成；RoboCasa 与 RoboTwin 中，更多终态需要夹具操作或双臂接触。（论文图 6）

```{figure} images/Harness-VLA/harness-vla-attribution.png
:alt: 成功 rollout 中，最终任务完成条件在分析性原语或 VLA 原语后触发的比例
:width: 100%
:align: center

**原论文图 6：任务完成归因**。统计分母是成功 rollout；它记录最后触发完成条件的原语类别，不能解释为组件贡献率。
```

调用次数统计提供了另一种观察角度：（论文附录 F、表 19）

| 环境 | 分析性原语占比 | `vla_act` 占比 |
| --- | ---: | ---: |
| LIBERO Pro-family | 84.2% | 15.8% |
| RoboCasa365 | 64.7% | 35.3% |
| RoboTwin C2R | 52.6% | 47.4% |

这里按原语调用次数计数，并排除 reset、渲染、笔记等辅助操作。一次 VLA 调用可能包含多个 chunks，因此这些百分比不代表执行时间、计算量或控制步数占比。

## 七、RPent 实现与论文接口

### 7.1 论文抽象与当前实现的对应关系

论文的核心接口是串行的“执行一个原语并返回观测”，公开 RPent 则将环境、VLA 与感知能力组织为服务，并提供不同 planner 后端。文件记录负责可追溯性，RPC 等通信机制负责组件交互；两者可以并存，不能把文件与 socket 理解成互斥的系统形态。

原笔记关注了 `pi0_pick`、`pi0_doubled` 等具体 VLA 调用，以及 SAM3 分割、深度反投影和 Markdown 记忆上下文。这些属于实现层，论文用统一的 `vla_act` 抽象描述局部 VLA 操作。下面对照 [RPent 提交 `068cd64f`](https://github.com/RLinf/RPent/tree/068cd64f4f175179a13c67d429a4d00ca56fb196)，将实际接口与成功信号分开理解。

| LIBERO 实现 | 核对结果 | 阅读时的含义 |
| --- | --- | --- |
| `pi0_pick` | 默认最多 24 chunks；末端下降 0.10 m 后再上升 0.05 m，且夹爪开度在 `[0, 0.06)` 时可局部返回成功 | 是末端与夹爪的启发式检查，仍可能空抓 |
| `pi0_doubled` | 默认最多 20 chunks，`success` 绑定环境的正式终止信号 | 中间接触动作是否有效，需要另看图像和状态 |
| `_vlm_chunk` | 每个 chunk 后刷新观测，再供下一次模型预测 | VLA 内部仍有反馈；planner 等原语返回后介入 |
| 评估 toolkit | 不注册 reset 工具 | 当前实现从可调用接口层面约束评估重置 |

源码位置：[VLA 调用与停止条件](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/robots/libero/tools.py#L151)、[评估工具注册](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/robots/libero/toolkit.py#L87)。这些默认值属于该提交的 LIBERO 实现，不能推广到全部 benchmark，也不能直接认定为 v3 实验时使用的阈值。

### 7.2 感知辅助和复现入口

SAM3 mask 可以帮助选择物体像素，也可由 planner 根据图像直接选取像素，再查询 [`back_project`](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/robots/libero/tools.py#L1819)。后者读取预计算的 world map，将像素与三维位置对应。分割工具本身不意味着获得物体真实坐标，也不保证透明、反光或遮挡场景的定位准确。

该提交的[官方复现说明](https://github.com/RLinf/RPent/blob/068cd64f4f175179a13c67d429a4d00ca56fb196/docs/source-en/rst_source/resources/harnessvla.rst#L120)区分环境入口：LIBERO 使用 `reproduce/libero`，RoboCasa 使用 `main`，RoboTwin 使用 `reproduce/robotwin`。因此，早期笔记中“公开版本仅包含 LIBERO”的描述已不适合作为当前发布范围。

当前主分支的记忆组织和排行榜也已演进。官方说明要求保留历史代码与数据的配对，并明确这些入口不等于重新验证过的历史复现结果。因此，本文分别标明论文版本与源码提交，不用当前排行榜替换 v3 主表。

## 八、后续版本补充与证据边界

### 8.1 Flash Mode 与真机演示

2026 年 9 月 24 日的 [v5 §4](https://arxiv.org/html/2607.08448v5#S4) 增加了 Flash Mode 与真机演示。Flash 将成功的原语序列整理为 task card，重新定位对象和目的地后按预先记录的顺序执行，并使用预设检查与有限重试，执行期间无需在线 planner。它更适合子目标顺序固定、对象可直接定位的任务；出现新子目标或未预设的恢复情况时仍需在线规划。

双 Franka 真机部分展示了目标重定向、按顺序操作、抓取恢复和新的双臂任务组合。论文将这些记录明确定位为**定性演示**，并由人工恢复初始场景及确认任务成功，不能将其当作具有相同试验规模的真机成功率对照。因此，原笔记中“没有真机验证”的范围应限定到早期版本。

Flash 报告 72.63%（581/800）的成功率，但使用报告评估的同一批 seeds 选择最高成功率的 task card，属于回顾性的最佳卡结果。时间表中，Flash 统计被选来源轨迹的工具执行时间，Codex 统计 planner duration。论文已说明这些数字不能直接证明端到端加速比，也不属于严格独立留出数据上的选卡评估。

### 8.2 新 planner 与 RoboCasa 统计变化

[v5 附录 H](https://arxiv.org/html/2607.08448v5) 增加 GPT-6 Astra planner，报告 LIBERO-Pro 92.63%、RoboCasa 59.20%。该附录还将原主文的 Codex / Claude Code 后端标为 GPT-5.5 / Opus-4.7；v3 实验设置只明确使用了两个 agent，因此本文主表保留原来的命名。

RoboCasa 主表也发生了变化：v5 改为全部 50 个任务各评估 5 个 seeds，共 250 次，而非 v3 的 340 次。v5 表 4 给出 Codex 57.2%、CC 48.8%，正文却仍出现 57.1% / 48.6%，存在表述不一致。本文第 5 节保留 v3 的协议与数值；使用新版结果时应注明采用哪一张表，不能与旧版的总次数拼接。

### 8.3 我如何看待结论的强度？

最直接的证据是：在多个仿真 benchmark 中，对相同冻结后端增加 harness 后，总成功率提高。语义重新绑定、重新 staging 与接触恢复的示例，为这种改进提供了可理解的机制解释。

尚不能从这些结果得出的结论包括：固定原语库总是优于技能扩展；所有增益都来自某一类记忆；或者 planner 已经把任意失败变成了可恢复事件。论文也没有通过不同大小原语库的受控对比证明前一种普遍结论。

我会特别关注以下限制：

- **感知与目标识别**：选错对象或深度定位不准，会让局部控制在错误前提下执行。
- **调用期间的反馈间隙**：planner 无法在原语执行的任意时刻介入，快速变化或持续精细接触可能要求更紧密的反馈。
- **长任务中的恢复边界**：物体倾倒、不可达或局部误差累积，并不总能靠再次调用解决。
- **记忆构造成本**：主结果不计入参考 seed 探索开销；缺少可行参考解时，few-shot 流程的收益也未必成立。
- **成本与可复现性**：需要分别记录规划时间、物理执行时间、调用成本、模型版本和失败分布，成功率无法替代这些指标。

我的理解是，Harness VLA 展示了一种实用的能力复用方式：通过显式任务分解和观测反馈，学习在什么条件下调用已有技能。模型的局部能力仍然重要，外部编排则决定这些能力能否在当前任务中被正确地组合和恢复。

## 资料入口

- [论文 v3](https://arxiv.org/html/2607.08448v3)：本文主实验、原语词汇及图表编号的依据。
- [论文 v5](https://arxiv.org/html/2607.08448v5)：Flash Mode、真机演示、新 planner 与后续评估变化。
- [官方项目页](https://harnessvla.github.io/)与 [RPent 仓库](https://github.com/RLinf/RPent)：实现、文档与演示入口。

配图摘自原论文 v3，保留原图内容，中文图题为阅读说明。论文采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。
