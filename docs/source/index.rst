.. JaxonZhu-Documents documentation master file, created by
   sphinx-quickstart on Sat Nov 29 23:27:01 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

JaxonZhu · 具身智能笔记
================================================================================

.. image:: images/personal_page/JaxonZhu.png
   :alt: This is me.
   :width: 300px

你好，我是 JaxonZhu，一名具身算法工程师。

这里记录我的 VLA 论文阅读、源码分析和模型实践，以及智能体强化学习与 LeRobot 学习笔记。

我目前关注的方向：

- 跟踪 VLA 模型进展，复现公开 benchmark。
- 将 VLA 适配到具体的下游任务。
- 探索强化学习和奖励建模在 VLA 后训练中的应用。
- 关注数据采集、分布式训练和真实环境部署中的工程问题。

.. note::

   我更关心一个方法能否真正跑起来，以及从论文到实际任务之间，还需要补上哪些工作。

这里记录了什么
--------------------------------------------------------------------------------

- **论文解读**：从模型结构、训练方法和实验设计出发，整理 VLA 相关论文，也写下自己的理解和疑问。
- **源码分析**：拆解 :doc:`Evo-1 的网络结构 <Evo-1-networks>` 和 :doc:`InternVLA-A1 的联合注意力 <InternVLA-A1-model-PartA>`，对照论文理解实现。
- **实践记录**：记录 :doc:`Evo-1 在 ALOHA 仿真任务上的微调过程与观察 <Evo-1-aloha-finetune>`。
- **工具笔记**：整理 :doc:`LeRobot 命令行工具的实现原理 <lerobotindex-Terminal-Implementation-Principle>` 等学习笔记。

联系我
--------------------------------------------------------------------------------

- 邮箱： jbzhu1999@gmail.com
- GitHub： https://github.com/JaxonZhu
- 小红书： RetrievalAG

================================

.. toctree::
   :maxdepth: 1
   :caption: VLA 阅读与实践

   index-Pi-x-VLA-HEAD
   index-Xiaomi-Robotics-0-HEAD
   index-TrAct-HEAD
   index-SOP-HEAD
   index-SCIZOR-HEAD
   index-Evo-1-HEAD
   index-NORA-HEAD
   index-HiORS-HEAD
   index-iRe-VLA-HEAD
   index-RLinfVLA-HEAD
   index-GR-RL-HEAD
   index-ACG-HEAD
   index-OTR-HEAD
   index-AILOT-HEAD
   index-RoboDopamine-HEAD
   index-RoboReward-HEAD
   index-LingBot-Depth-HEAD
   index-InternVLA-A1-HEAD
   index-RoboChemist-HEAD
   index-MemER-HEAD
   index-MemoryVLA-HEAD
   index-mimic-video-HEAD
   index-Dit4Dit-HEAD
   index-CosmosPolicy-HEAD

.. toctree::
   :maxdepth: 1
   :caption: 智能体强化学习

   index-DISTRL-HEAD

.. toctree::
   :maxdepth: 1
   :caption: LeRobot 工具笔记

   lerobotindex-Terminal-Implementation-Principle
