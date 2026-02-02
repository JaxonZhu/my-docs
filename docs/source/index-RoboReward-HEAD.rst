RoboReward
===============

Robo-Reward: A Dataset And Benchmark For Vision-Language Reward Models In Robotics

根据论文前面的 episodic reward 定义 + 后面的 (v, t, r) 三元组 + 后面得出来多离散值奖励，在没实际 infer 他们开源的模型前来看，能否理解成从本质上把 VLM 改造成一个视频分类模型，这个模型输入视频并以文本指令为条件，输出 1-5 离散标签作为 reward ？

最后用他们在 VLM 上微调的、能预测 5 个离散 episodic reward 数值的 RewardModel 在 DSRL 上的表现，发现不如人类标注出的二元的 episode reward 的离散奖励数值的表现。可能从这样理解：（1）人类标注的其实算作 ground-truth reward, 从而说明他们训出来的 VLM reward model 能接近于人类实际观测得到的 ground-truth 的奖励；（2）还有就是可能存在的情况是，VLM 的是视觉输入，那么可能就会架设眼在手外的相机来捕获机械臂和操纵物体，可能受到视角 / 噪声 以及机械臂自己移动时候的遮挡，VLM reward model 会预测错误进而影响 RL. 

但是它的反事实轨迹重标注，这个工程 pipeline 好实现呀


.. toctree::
   :maxdepth: 1
   :caption: Contents of RoboReward:

   RoboReward-paper