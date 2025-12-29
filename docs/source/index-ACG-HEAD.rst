ACG
===============

论文标题

*ACG: Action Coherence Guidance for Flow-based VLA models*

上周末 (也就是 2025 年 12 月 28 号) 刚把 :math:`\pi_{0.6}^{\ast}` 这个工作看完. 这篇工作中基于动作优势学习的基础是 CFG 算法, 同时这篇工作也提到了如何控制阈值而不是修改 CFG Guidance Scale 来引导模型推理表现更好.

阅读完 :math:`\pi_{0.6}^{\ast}` 这篇文章后, 还想到自己的论文仓库里也还有一篇基于 Flow-matching 的, 主打 training-free 的方式在 test-time 引导模型生成更符合预期动作的 VLA 模型, 那么这篇工作就是 ACG.

ACG 的目标是优化 Flow-based VLA 模型在生成动作序列时的连贯性. 具体来说, ACG 通过引入一个动作连贯性评分函数, 在推理过程中对生成的动作进行评估和调整, 以确保动作序列在时间和空间上的一致性. 该方法不需要额外的训练步骤, 只需在推理阶段进行调整, 因此被称为 training-free.

最本质思想就是: 作者认为 attention map 是控制动作连贯性的关键, 因此破坏这个 attention map 就能破坏动作连贯性. 这样就能输出不能产生连贯动作的辅助 VLA 模型, 接着套用 CFG 的思路优化训练目标和推理过程, 让 VLA 模型尽可能偏离这个不连贯的 VLA 去噪向量越远越好.

整体论文可读性高, 实验部分的问题 / 数据 和 论文表述 可以做到环环相扣. 其实话说话来, 这篇文章的实现成本很低 [doge].


.. toctree::
   :maxdepth: 1
   :caption: Contents of ACG:

   ACG-paper