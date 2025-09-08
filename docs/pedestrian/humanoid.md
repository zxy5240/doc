# [人形机器人仿真](https://github.com/google-deepmind/mujoco/network/dependents)

<!-- 更新到最新的仓库： https://github.com/parthh01/rl_stuff 
更新到的页面： https://github.com/google-deepmind/mujoco/network/dependents?dependents_before=MjE1NjQ0OTU3NjQ -->

## Mujoco

[官方的移动任务实现](https://github.com/google-deepmind/dm_control/tree/main/dm_control/locomotion)

[层次世界模型实现人形全身控制](https://github.com/nicklashansen/puppeteer)

[模仿学习基准专注于使用 MuJoCo 执行复杂的运动任务](https://github.com/robfiras/loco-mujoco)

[全身控制的层次世界模型](https://github.com/nicklashansen/puppeteer)

[MyoSuite](https://github.com/MyoHub/myosuite) - 使用 MuJoCo 物理引擎模拟的肌肉骨骼模型要解决的环境/任务的集合，并包含在 OpenAI gym API 中

[dm_robotics：为机器人研究创建和使用的库、工具和任务](https://github.com/google-deepmind/dm_robotics)

### 建模

[OpenSim 肌肉骨骼模型转](https://github.com/MyoHub/myoconverter)

### 比赛

[足球射门、乒乓球对打](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2025) - 其他 [mujoco_ros2_control](https://github.com/moveit/mujoco_ros2_control) 


## ROS

[乐聚机器人控制](https://github.com/LejuRobotics/kuavo-ros-opensource) - 包含 Mujoco 仿真环境

[将 ROS 与 MuJoCo 结合使用的封装器、工具和附加 API](https://github.com/ubi-agni/mujoco_ros_pkgs) - 支持 Noetic

[一款一体化 ROS 软件包 RoTools](https://github.com/DrawZeroPoint/RoTools) - 用于高级机器人任务调度、视觉感知、路径规划、仿真以及直接/远程操控。它利用 BehaviorTree 实现快速的任务构建和协调，并提供各种实用程序来弥合真实/模拟机器人与高级任务调度程序之间的差距。

## OpenSim

[使用 MuJoCo 物理引擎模拟的肌肉骨骼模型要解决的环境](https://github.com/MyoHub/myosuite) - 包含在 OpenAI gym API 中

[将 opensim 4.0+ MSK 模型转换为 MuJoCo 格式的工具](https://github.com/MyoHub/myoconverter) - 具有优化的肌肉运动学和动力学



## 强化学习

[使用 OpenAI Gym 环境的 xArm6 机器人强化学习框架](https://github.com/julio-design/xArm6-Gym-Env) - 该模型使用深度确定性策略梯度(DDPG) 进行连续动作，并使用后见之明经验回放(HER)

[四足动物-斯坦福小狗文档和训练学习者](https://github.com/Prakyathkantharaju/Quadruped)

[基于深度 Q 网络的 TensorFlow 2 强化学习实现](https://github.com/metr0jw/DeepRL-TF2-DQN-implementation-for-TensorFlow-2)

## 模仿学习

[通过语境翻译进行观察模仿](https://github.com/medric49/imitation-from-observation) - 一种基于演示训练代理模仿专家的算法

## 控制

[一种基于视觉模型的强化算法 Dreamer](https://github.com/adityabingi/Dreamer) - 它学习一个世界模型，该模型从高级像素图像中捕捉潜在动态，并完全在从学习到的世界模型中想象的部署中训练控制代理

[倒立摆](https://github.com/dhruvthanki/mj_InvertedPendulum) - 使用基于优化的操作空间控制来跟踪双摆的末端执行器位置


## 数据

[用于模拟人形控制的多任务数据集](https://github.com/microsoft/MoCapAct)

## 研究

[谷歌研究库](https://github.com/google-research/google-research)

[信念状态编码器/解码器](https://github.com/lucidrains/anymal-belief-state-encoder-decoder-pytorch) - 似乎产生了一种可与波士顿动力手工算法（四足动物 Spot）相媲美的策略

[深度强化学习中的首因偏差](https://github.com/evgenii-nikishin/rl_with_resets) - 深度强化学习代理的 JAX 实现，带有重置功能

[构建目标驱动的具身化大脑模型](https://github.com/ccnmaastricht/angorapy)

[稳定神经近似的逆向经验重放](https://github.com/google-research/look-back-when-surprised)

## 平台

[Gym](https://github.com/openai/gym) - 用于开发和比较强化学习算法，它提供了一个用于在学习算法和环境之间进行通信的标准 API，以及一组兼容该 API 的标准环境。已迁移至 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 

[通才generalist](https://github.com/grahamannett/generalist)


## 工具

[一个基于 C++ 的批处理环境池 EnvPool](https://github.com/sail-sg/envpool) - 基于 C++ 的高性能并行环境执行引擎（矢量化环境），适用于通用 RL 环境

[JAX（Flax）实现具有连续动作空间的深度强化学习算法](https://github.com/ikostrikov/jaxrl)

[用于处理 MuJoCo 中使用的复合 Wavefront OBJ 文件的 CLI](https://github.com/kevinzakka/obj2mjcf)

[用于执行无梯度优化的 Python 工具箱](https://github.com/facebookresearch/nevergrad)



### 杂项

[InterGP](https://github.com/tdardinier/InterGP) - 收集数据、训练代理的流程

[Docker Wiki 和示例](https://github.com/dotd/docker_wiki)

[强化学习实验](https://github.com/parthh01/rl_stuff)

[这是Spinning Up的一个克隆版本，目标是使用最新的 PyTorch 版本](https://github.com/haha1227/spinningup-pytorch)

[Fast Campus 强化学习](https://github.com/Junyoungpark/ReinforcementLearningAtoZ)