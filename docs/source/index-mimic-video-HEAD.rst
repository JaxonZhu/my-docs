mimic-video
============

2025 年 12 月发布的工作，提出了一个模型架构，输入 video-text pairs, 将基于 diffusion / flow-matching 过程的生成式视频模型的中间时刻中间层的表征张量，提取出来作为 condition, 加入到同样基于 diffusion / flow-matching 过程的 action decoder, 解码出动作序列。文章观点明确，指出了 VLM-based VLA 的不足，并非常自然地引入了 视频模态数据 和 视频预测模型。论文中的两个信息看起来挺有价值：（1）通过观点提出 / 实验 / 分析等内容，作者表达出 VAM 这样的 world model 并不需要解码出清晰的未来时刻帧，中间甚至初始去噪 / 流时刻的表征最有价值；（2）VAM 的显著优势，在于解决样本效率 / 训练数据量等问题；结合作者对 VLM-based VLA 等工作的批判来看，VAM 的样本效率提升可以理解为给 action decoder 减负了。

结合之前看的 memory-based VLA 相关文章，这些文章提到了"短期记忆"实质上捕获当前时刻的前若干帧作为输入，并且也提到"短期记忆"可以缓解机械臂移动时的自我遮挡问题。在阅读 mimic-video 这个工作，因为视频预测模型需要输入 video-text pairs，自然包含了 short-horizon memory, 因此文中提到"有效消除由遮挡引起的视觉不确定性"与 memory-based VLA 相关文章表述近似，实际是使用 VAM 解决了 memory 的问题。

.. toctree::
   :maxdepth: 1
   :caption: Contents of mimic-video:

   mimic-video-paper
