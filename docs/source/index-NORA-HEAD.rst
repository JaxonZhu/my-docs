NORA
===============

目前 NORA 系列 VLA 发布两项工作:

*NORA: Neural Orchestrator for Robotics Autonomy*

这个工作相对简单, 发布了基于 Qwen VLM 和 FAST+ 分词器的小参数量模型，并在常见 benchmark 上与其他基线对比.

*NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-based Preference Rewards*

这个工作相对复杂, 主要包含两份工作量: 在原始预训练 NORA-1 的基础上增加了类似 $\pi_{0}$ 的 flow-matching based action expert, 先增强了一波基础模型的能力; 第二份工作量是引入了以世界模型为原型 + 真值动作条件引导下的启发式奖励信号，二者协同进一步对 VLA 使用 DPO 算法后训练.

感到意外的是文章语言表达, NORA-1 相对清晰直观; 但是到了 NORA-1.5 这个版本文章的可读性, 或者说表达上更正式了.

NORA-1.5 是具有可实践性的: 基于视觉世界模型来进一步改造 / 微调出奖励预测模型这一点, 视觉世界模型本身基于海量数据预训练, 在下游可以尝试多数据集微调得到一个特定环境 / 特定具身实体的奖励预测器; 基于奖励预测器, 让 VLA policy 采样多个 action chunk 标注奖励, 可行性也相对高.

.. toctree::
   :maxdepth: 1
   :caption: Contents of NORA:

   NORA-1-paper
   NORA-15-paper

