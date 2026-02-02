LingBot-Depth
===============

Masked Depth Modeling for Spatial Perception

读这篇文章带着这样的一个问题，是否能把这个 model 作为 robot policy 的视觉 backbone? 尽管文章里面提供了一个 robot policy 的案例，但是这个案例是将（有 / 无经过 LingBot-Depth 补全的）RGB-D 数据转成点云，然后经过不同的 backbone 提取 RGB / D 模态的特征结合 diffusion policy 来预测灵巧手部姿态。但是作为预训练视觉 model, 是否可以获得隐式的 RGB 和深度的联合表征然后承接下游 action generation 模块？

首先从他们数据集分布来看，涵盖的 indoor 场景都是普遍具身期望落地的场景，数据集分布上看具有强关联性；其次就是从 model 结构上看，LingBot-Depth 的 encoder 部分生成的 Latent Contextual Tokens 作为 tokens 序列包含了 RGB 和深度的联合编码，我想就是它们在 Abstract 上提到的“跨 RGB 与深度模态的对齐潜在表征”，也可以用；同时，保留一个 [cls] token 以跨模态捕获全局上下文，可以拿来用。

让我觉得很棒的是他们 Fig.14 的深度补全，首先图中的透明物体确实对 RGB-D 相机而言难度大；另外我还看到一个细节是，放置这些物体的桌面表面应该是带有凹凸不平的竖状纹理的，四张图中 RGB-D 相机都观察到了纹理，而相机捕获的数据确实是失真 / 错误值，emmmmm 但是 LingBot-Depth 确实补全了，把整个有竖状条纹的桌面当成平面处理了。


.. toctree::
   :maxdepth: 1
   :caption: Contents of LingBot-Depth:

   LingBot-Depth-paper