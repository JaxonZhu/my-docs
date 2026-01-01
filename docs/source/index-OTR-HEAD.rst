OTR
===============

论文标题: Optimal Transport For Offline Imitation Learning

这是一篇 2023 年的 Offline RL 论文，里面提出的解决方案对我而言比较有用，而且整体算法不难，且内部一个核心模块是有现成 module 目测可以调。

思路就是使用沃瑟斯坦距离来衡量 expert 数据集状态遍历分布和 unlabeled 数据集状态遍历分布的差异性 / 距离度量，而且二者都不需要任何事先的奖励标注。粗糙的理解就是当某条 episode 的状态遍历偏离 expert 状态遍历分布，那么就会受到比较大的负值奖励，反之则小。这就造成使用 Offline RL 得到的 policy 对 expert 具有高度相关性，这个也在实验部分提到了；另外就是这个方案应该对 expert 状态遍历分布外数据有敏感性，可以被应用到分布外（状态）数据的检测。

但是这个算法大概率只在 state-based 上做了实验，是否可以上升到 image-based / vision-language based 这种自带多模态 + 大规模的演示轨迹，不明确。但是自己调研到后面几年也有更新的相关工作，可以再看看。

阅读来看，就是很标准的 RL 顶会文章 hhhh 读起来就是一套一套的，实验部分的问题提出 分析 论述都很实在。


.. toctree::
   :maxdepth: 1
   :caption: Contents of OTR:

   OTR-paper