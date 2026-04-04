CosmosPolicy
============

这篇文章的一作是 Moo Jin Kim 也是 OpenVLA-OFT 的作者，所以读这篇论文会带来和读 OpenVLA-OFT 相似的感觉。从 论文的术语表述 / 实验部分的设计和论证闭环 / 附录部分的训练数据 来看都跟 OpenVLA-OFT 味道很像。

这篇文章从 video model 出发的预测未来帧出发，不改变模型结构 / 不增加新的模块下，通过魔改输入数据，将各种视觉 / 非视觉模态注入后的 latent frames 进行加噪-去噪操作，同时在全参微调模型的范式下多角度构建微调损失函数，实现了用少量演示数据完成 policy / world-model / value prediction 任务三合一。

让我比较难理解的就是：为什么非视觉模态的本体 / 动作块 / 价值等数据可以通过"复制填充"的方式 inject 到 latent frames ？可能有几点吧（1）作者的目标是不引入新模型、不改变训练算法，那么肯定不能用 learned network 来预测非视觉模态的 latent frames 再 inject 了，这样引入 learnable 模块变相地增加了模型架构和学习算法；（2）复制填充操作的特点是，这样操作后的 latent frames 不会影响预训练时候学习到的视觉 tokens 统计和时序依赖，本质上是把这些非视觉模态嵌入为"全局一致的 latent token"，类似于 [CLS] token，从而在不修改架构的前提下，使预训练视频扩散模型能够在其原生的时序生成机制中联合建模动作 / 未来状态 / 价值。在随后的全参微调训练中逐步"接纳"这些 tokens.

.. toctree::
   :maxdepth: 1
   :caption: Contents of CosmosPolicy:

   CosmosPolicy-paper
