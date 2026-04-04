Dit4Dit
=======

看完 mimic-video 后再看这篇 DiT4DiT 工作，会发现有超多相似之处。(1) 批判 VLM-based VLA 的说法是类似的，都是批判 VLM 在静态 web 数据上预训练，没有捕获动力学和时空联系，进而引出 video model; (2) 文章结构上也是先用 small experiments 进行说法验证，都强调 video-model as backbone 是可行的; (3) 同样模型架构上，都强调不依赖重建完整的帧，而是使用去噪 / 流过程的中间去噪特征作为动作解码的 condition; (4) 在实验部分都认为早期去噪步骤带来的表征对 action decoder 有利，然后逐渐递减。

.. toctree::
   :maxdepth: 1
   :caption: Contents of Dit4Dit:

   Dit4Dit-paper
