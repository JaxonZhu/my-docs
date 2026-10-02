# $\pi_{0.6}^{\ast}$ 论文解读：RECAP 后训练方法

论文原题：$\pi_{0.6}^{\ast}$: a VLA That Learns From Experience

这篇图片笔记整理 RECAP 如何把机器人自主执行的经验和人工纠错用于 VLA 后训练。阅读主线是：训练价值函数、估计动作优势，再把优势信息作为条件来改进策略。基础模型的结构见 [$\pi_{0.6}$ 模型卡解读](Pi-06-model-card.md)。

按主题阅读：[问题与基础概念](#recap-background) · [RECAP 方法](#recap-method) · [实现细节](#recap-implementation) · [实验与讨论](#recap-evaluation)。

(recap-background)=
## 问题背景与强化学习基础

![π0.6* 阅读笔记第 1 页：RECAP 概览与研究动机](images/Pi-06-star/Pi-0.6-starPaperReading_01.png)

![π0.6* 阅读笔记第 2 页，问题背景部分](images/Pi-06-star/Pi-0.6-starPaperReading_02.png)

![π0.6* 阅读笔记第 3 页：优势函数与正则化强化学习](images/Pi-06-star/Pi-0.6-starPaperReading_03.png)

(recap-method)=
## RECAP：价值估计与优势条件策略

从数据收集、价值函数训练和策略改进这三个环节展开，重点看优势信息如何进入策略，以及作者为什么采用优势条件化的方法。

![π0.6* 阅读笔记第 4 页：RECAP 三个环节与模型示意图](images/Pi-06-star/Pi-0.6-starPaperReading_04.png)

![π0.6* 阅读笔记第 5 页，RECAP 方法部分](images/Pi-06-star/Pi-0.6-starPaperReading_05.png)

![π0.6* 阅读笔记第 6 页：策略提取与优势加权回归的比较](images/Pi-06-star/Pi-0.6-starPaperReading_06.png)

![π0.6* 阅读笔记第 7 页，RECAP 方法部分](images/Pi-06-star/Pi-0.6-starPaperReading_07.png)

![π0.6* 阅读笔记第 8 页，RECAP 方法部分](images/Pi-06-star/Pi-0.6-starPaperReading_08.png)

(recap-implementation)=
## 算法流程与实现细节

这里把方法落到 $\pi_{0.6}$ 的输入、训练目标和价值估计上，并用数值示例串起奖励、价值与优势的计算。

![π0.6* 阅读笔记第 9 页：RECAP 算法与优势条件输入](images/Pi-06-star/Pi-0.6-starPaperReading_09.png)

![π0.6* 阅读笔记第 10 页，模型与系统实现部分](images/Pi-06-star/Pi-0.6-starPaperReading_10.png)

![π0.6* 阅读笔记第 11 页，模型与系统实现部分](images/Pi-06-star/Pi-0.6-starPaperReading_11.png)

![π0.6* 阅读笔记第 12 页：优势计算示例与训练流程](images/Pi-06-star/Pi-0.6-starPaperReading_12.png)

(recap-evaluation)=
## 数据采集、实验评估与讨论

先接上自主执行和人工干预数据的使用方式，再看各任务的成功率、吞吐量与迭代效果。最后整理作者对人工反馈、探索方式和离线迭代的讨论。

![π0.6* 阅读笔记第 13 页：人工干预、迭代训练与评估任务](images/Pi-06-star/Pi-0.6-starPaperReading_13.png)

![π0.6* 阅读笔记第 14 页，实验评估部分](images/Pi-06-star/Pi-0.6-starPaperReading_14.png)

![π0.6* 阅读笔记第 15 页，实验评估部分](images/Pi-06-star/Pi-0.6-starPaperReading_15.png)

![π0.6* 阅读笔记第 16 页：多轮 RECAP 迭代的吞吐量与成功率](images/Pi-06-star/Pi-0.6-starPaperReading_16.png)

![π0.6* 阅读笔记第 17 页，实验评估部分](images/Pi-06-star/Pi-0.6-starPaperReading_17.png)

![π0.6* 阅读笔记第 18 页，实验评估部分](images/Pi-06-star/Pi-0.6-starPaperReading_18.png)

![π0.6* 阅读笔记第 19 页：行为纠正讨论、方法局限与未来工作](images/Pi-06-star/Pi-0.6-starPaperReading_19.png)
