AILOT
===============

论文标题

*Align Your Intents: Offline Imitation Learning Via Optimal Transport*

这篇文章是 2025 ICLR 的文章，这种基于 Optimal Transport 的离线轨迹奖励重标注的论文每年都存在，但是每年都没有很多篇。

这篇文章作为读者方可以从两个点切入：（1）首先是 2023 年发表的 OTR 只是在 state-only 上进行 OT 求解并做奖励重标注，没有上升到 image-based 轨迹奖励重标注的能力，且原始 OT 的算法也没有这样标注 image trajectory 的能力，因此这篇 AILOT 的创新点就在做表征工程：引入了浅层意图表征，用 successor features 来构建 / 训练。（2）另一个切入点在于 OTR 这篇工作的 cost 函数是基于对数据集状态的 cosine 进行计算，这就使得 cost 函数只考虑到了状态空间的关联，并没有对时序上进行对齐，也就是考虑到了“点”没考虑到“线段”，因此这篇 AILOT 的创新点就将把时序上存在差异的状态，通过表征方法，映射到表征空间上，且映射后的两个 embeddings 平方范数随时序差异成正比。

因为自己阅读 OTR 和 AILOT 更希望学到方法而不想太多关注实验部分，所以 D4RL 的涨点分析就忽略了。另外这篇工作的实现是基于 JAX 的... 这个会在实操上存在困难。

吐槽一下，这个读论文真是一个轮回！2025年年初读的是 successor features 的论文，也是用三个网络来构建一个价值函数；现在 2026 年年初还在读跟 successor features 相关的论文。。。


.. toctree::
   :maxdepth: 1
   :caption: Contents of AILOT:

   AILOT-paper