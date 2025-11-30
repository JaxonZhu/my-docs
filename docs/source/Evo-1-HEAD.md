Evo-1: Lightweight Vision-Language-Action Model with Preserved Semantic Alignment
===============

可以说 Evo-1 相当于国内版 SmolVLA 了，点赞~
	
原生 VLM 是什么意思？与 post-hoc alignment 不同，后者只是对纯文本大模型进行改造以处理图像，而 InternVL3 联合学习大规模多模态和文本语料库中的语言与视觉理解能力。从 InternVL 的开源文档上可以看出，InternVL-3 包含了 InternViT-300M 和 Qwen2.5-0.5B ，也就是这俩组合成了 InternVL-3 ；若不了解 InternVL 系列模型的话，可能会有“为什么论文内容是 InternVL-3 但是论文 VLA 架构图中是 InternViT-300M ？”这样的疑问。
	
无需机器人数据预训练，我想这个意思具体是指 Evo-1 没有类似 OpenVLA 系或者 pi 系使用 OXE 这样 robot datasets 预训练 action-expert 的过程，而是在具体任务（仿真 / 实机）上采集的演示轨迹，直接进行两阶段训练。
	
Evo-1 是 InternViT-300M + Qwen2.5-0.5B ，而 SmolVLA 是 SigLIP + SmolLM 。在 SmolVLA 中也用到了令牌重排技术来减少视觉 token 数量，但是 SmolVLA 的做法更激进。SmolVLA 中取 VLM 解码器的前一半，Evo-1 中同样使用语言解码器的前中部分，同时也在消融实验中做了比较。这俩 VLA 的推理占用显存都在 2.0G 左右，可操作性很高。

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Evo-1-paper
