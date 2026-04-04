MemER
========

MemER: Scaling Up Memory For Robot Control Via Experience Retrieval

这篇文章跟 RoboChemist 类似，用广义 VLA 解决 long-horizon 操作任务，但是 RoboChemist 是化学场景，MemER 是通用场景。

整体来看，因为是广义 VLA，所以设计架构参考<VLA 中 VLM 与 Action 融合 五种可能的方式>算是 "Dual-System Architecture" 架构。这个工作在训练过程中可将 VLM 和 VLA 分离训练。这篇工作比较复杂 / 比较难读的是 引导 VLM 生成语义子任务和选出关键帧作为记忆重点机制。VLM 根据语言指令 / 场景视觉图片 和 前序记忆重点，生成当前语义子任务 + 多帧候选关键帧，再通过简单 1 维单链聚类算法，从候选关键帧组中更进一步选择关键帧。

接下来实验部分比较值得看的是对每个长程任务都设计了除成功率以外的任务特定指标。对于依赖记忆探索型任务，可以记录不必要探索的路径；计数任务可以设计差异化指标；以及长程任务本身可设计多步评分机制，来弥补任务总成功率指标的内在信息 / 内在规律丢失。另外就是实验设计上，在已知 API 收发信息速度较慢的情况下，使用 held-out 数据集离线推理，进一步设计评估的方式也值得看。


.. toctree::
   :maxdepth: 1
   :caption: Contents of MemER:

   MemER-paper
