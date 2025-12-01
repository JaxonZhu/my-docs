RLinf-VLA
===============

RLINF-VLA: A Unified And Efficient Framework For VLA+RL Training

这篇文章不侧重算法创新，而是在框架构建上做了很多优化。同时在优化过程中发现了 PPO / GRPO 的实践性总结。

对于论文中出现的 PPO 算法中单个动作的价值估计优于 chunk 级价值估计现象，可能的说法是 chunk-level 价值把整个 chunk 当成一个宏动作，只能输出一个数值，它无法预测 chunk 内每个动作的不同重要性，结果是价值函数难以拟合（论文图 10b 中 chunk-level 的损失曲线位于 action-level 上方）；同时在优势计算上，chunk 内有些动作贡献高，有些动作贡献低，但 PPO 被强迫对它们给予同样的 advantage ：“优势类型与所有粒度不小于自身粒度的对数概率类型兼容”。而 action-level critic 可能的说法是价值函数从同一个观察预测 chunk 内每一步的未来收益贡献，使 PPO 获得更细粒度的 credit assignment 。

作者在仿真数据上，用不同规模的专家演示轨迹训练 OpenVLA 做 SFT ，这些轨迹总共对应约 1.26 M 状态动作对；随着轨迹数量增加，OpenVLA 的性能提升到大约在 16000 条轨迹左右后趋于平稳饱和。博客是基于 arxiv 2510.06710v1 论文版本解读的，个人猜想这版论文的真机实验部分可能缺少了一定量的内容。

作者在 FUTURE WORK 中提到会在框架中增加 SAC 相关算法，值得期待。

.. toctree::
   :maxdepth: 1
   :caption: Contents of RLinf-VLA:

   RLinf-VLA-paper

