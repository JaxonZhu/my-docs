iRe-VLA
===============

作者在 Introduction 中提到，直接将标准 RL 算法应用于大型 VLA 会导致训练过程不稳定并引发性能下降，这也和之前看过的 Action-chunk-PPO 和 SimpleVLA-RL 相呼应：前者在 RL 过程中引入 SFT 和动态缓冲区，后者直接先 SFT 一个有一定能力的 base 模型作为 RL 的起点，同时都说明了 RL-only 难起作用。

与 Action-Chunk-PPO 算法类似，iRe-VLA 提出了一个在线数据集，用于不断吸收 RL 探索出成功的轨迹。

实验中对比了 PPO-replay ，也就是在每个 PPO online RL 训练完一个 task 后，用专家演示轨迹再跑一次一小段的 SFT replay 来抵抗灾难性遗忘。

另外，作者只是提到，基于优化目标进行 RL ，但是并没有介绍到底是哪个具体的 RL 算法。猜测在仿真环境的 online RL Stage 用的就是 PPO 系列；到真机实验那部分才换成 SACfD 。

在真机实验中，需要注意的是作者的“VLA = 预训练 VLM + 随机初始化 action head”，因此在真机任务上需要先采集一定量（2k episode）来训这个 VLA ，而不是像现在 OpenVLA-OFT 这种已经预训练过的，发布出去的 VLA 。

里面还有两个说法/表述也需要关注：(1) 在用标准 PPO 全参微调 VLA 时候，梯度存在噪声，这对 VLA 模型预训练表征产生了负面影响，但是 (2) 在线机器人动作数据能够增强 VLM 上层的表征能力，从而提升 VLA 模型在未见任务中的泛化能力。

论文很短，就 6 页，arXiv:2501.16664v1 版本上也没有显示开源，感觉这个仿真可以开出来看看。。。

.. toctree::
   :maxdepth: 1
   :caption: Contents of iRe-VLA:

   iRe-VLA-paper