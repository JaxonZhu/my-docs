MemoryVLA
=========

标题：MemoryVLA 阅读与思考

文案：相比于上一篇 RoboChemist 使用广义 VLA 来解决长程 / 灵巧的化学操作，这篇的立意在于研究记忆增强的狭义 VLA 用以解决通用的操作类任务，因此这篇文章重点在于设计架构 / 训模型和大规模评测。

提及到设计架构，就可以拿出之前那篇小红书笔记<VLA 中 VLM 与 Action 融合 五种可能的方式>来作对照：（1）工作中的 cognitive token 其实是输入序列中具有特殊意义 token 的对应位置输出，可以看成是 "condition fusion" 特点就是"聚合表征"；（2）记忆增强后的认知表征 c 和加噪声的 action tokens 序列合并，一同输入到 action expert 也算 "condition fusion" ，本质上又复用了一次聚合的信息；（3）在 action expert 中添加了一层 perception attention ，用于让 action tokens 序列能直接对 perceptual tokens 序列访问，这属于 "cross-attention fusion" 方法。

接下来是 memory 部分，使用脑科学的机制引入短时记忆 (working memory) / 长期记忆 (用 memory bank 具象化) / 记忆检索和整合 / 增强动作专家的过程。这部分，包括视觉特征提取的 Perceptual Compression 模块，直接精读论文还是略模糊，需要结合代码细看。

读完模型架构的时候，比较困惑的问题是这个模型是如何被训练的，更具体的在于这个训练过程的 dataloader 是如何建立的？如果是常规 dataloader 将数据拆分成 samples 不能在训练过程中维护到 memory. 因此精读了 appendix 部分关于 dataloader 的设计，自己更加理解了。

还有启发的是，这种 long-horizon + memory 的仿真 / 真机任务应该如何设计？论文开头部分提到了："现实操作任务中，在执行前后几乎无视觉差异，难以判断该动作是否已完成"这也是论文后面几个真机任务的设计思路，比如按按钮，因为按按钮前后的视觉状态变化很小，如果没有记忆，可能会很容易重复按下按钮。

.. toctree::
   :maxdepth: 1
   :caption: Contents of MemoryVLA:

   MemoryVLA-paper
