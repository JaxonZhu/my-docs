NORA
===============

目前 NORA 系列 VLA 发布两项工作:

*NORA: Neural Orchestrator for Robotics Autonomy*

这个工作相对简单, 发布了基于 Qwen VLM 和 FAST+ 分词器的小参数量模型，并在常见 benchmark 上与其他基线对比.

*NORA-1.5: A Vision-Language-Action Model Trained using World Model- and Action-based Preference Rewards*

这个工作相对复杂, 引入了以世界模型为原型的动作条件引导下的奖励信号, 同时还增加了启发式奖励信号，二者协同进一步对 VLA 做 DPO 后训练.

.. toctree::
   :maxdepth: 1
   :caption: Contents of NORA:

   NORA-1-paper

