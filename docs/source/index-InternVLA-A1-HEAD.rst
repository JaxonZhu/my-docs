InternVLA-A1
===============

InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation

这个是上海 AI Lab 在 2025 年底到 2026 年初发布的 VLA, 关注度很高.

总体是一篇非常完整的工作了: (1) 模型结构上, 引入了三个 experts 组件分别承担场景理解 / 未来状态生成 / 动作生成三个任务. 三个 experts 用 MoE 方式组织, 从底层看主要是 "联合注意力机制 + 前序序列 KV 缓存 + attention mask" 策略. 生成前瞻性视觉状态的 generative expert 使用了 cosmos 进行浅层编码解码, 兼顾推理性能和未来预测. 生成未来动作块的 action expert 则和 :math:`\pi_{0.5}` 类似. (2) 数据集上, 这篇工作一直在称赞他们的前序工作 InternData-A1 这个合成仿真数据集, 同时也掺了真机数据集. (3) 这篇工作还有工程性创新, 提出 Load-balanced Parallel Training 优化训练调度这样. (4) 实验部分也很完整: 动态-静态多任务 / 跨本体 / 兼顾 benchmark 涨点.

同时 Limitations 部分也很实在, 提到了 Understanding Expert 没有被监督; 同时还提到为了提升推理效率, 在 Generative Expert 降低了图像预测的保真度，这限制了生成未来帧的粒度. emmmmm 尽管如此, 如果可以把某个特定任务执行过程中的 前瞻性视觉预测 可视化出来, 可以让这份报告更出彩~



.. toctree::
   :maxdepth: 1
   :caption: Contents of InternVLA-A1:

   InternVLA-A1-paper
   InternVLA-A1-model-PartA