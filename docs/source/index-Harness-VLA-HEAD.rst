Harness VLA 阅读与源码分析
================================================================================

Harness VLA 将冻结的 VLA 封装为可重试的局部操作原语，由高层 planner 结合任务记忆和全局经验，处理目标绑定、空间定位、任务组合与失败恢复。

这篇笔记重点梳理记忆如何迁移、分析性控制与 VLA 如何分工，并核对 LIBERO-Pro、RoboCasa365 和 RoboTwin 的评估口径；同时补充 RPent 实现入口及后续版本的真机演示。

.. toctree::
   :maxdepth: 1
   :caption: 文章目录

   Harness-VLA-paper
