GR-RL
===============

这是字节团队在 2025 年 11 月出的文章, 专门介绍了使用 offline / online 强化学习方法对 VLA 进行后训练, 完成极具挑战性的 "把鞋带穿进鞋孔" 里这个任务.

自己也从这篇文章中学到不少技术, 像类 Hindsight 轨迹数据集增广 / task progress model 用于预测任务完成度 / 镜像对称数据集增广, 这些的可实践性相当高; 同时分布型 RL 算法用于 flow-based model 也十分值得细思, 因为这份工作没有开源, 所以很值得想想 "自己应该如何设计 / 调参这样的算法".

论文整体可读性以及图片展示都很好, 很喜欢这篇文章.

.. toctree::
   :maxdepth: 1
   :caption: Contents of GR-RL:

   GR-RL-paper