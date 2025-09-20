# [人形机器人仿真](https://github.com/google-deepmind/mujoco/network/dependents)

<!-- 更新到最新的仓库： https://github.com/Co-Evolve/mujoco-utils 
更新到的页面： https://github.com/google-deepmind/mujoco/network/dependents?dependents_before=MzE3MzAyNzY4NjU -->

- [__Mujoco__](#Mujoco)
    - [人的建模](#humanoid_model)
    - [比赛](#tournament)
- [__ROS__](#ros)
- [__人__](#humanoid)
- [__强化学习__](#rl)
    - [DQN](#dqn)
    - [AC](#ac)
    - [多智能体](#multi_agent)
    - [模仿学习](#imitation)
    - [分层强化学习](#hierarchy)
    - [逆强化学习](#inverse_rl)
    - [元强化学习](#meta_rl)
- [__感知__](#perception)
- [__规划__](#planning)
- [__控制__](#control)
- [__测试__](#test)
- [__数据__](#data)
- [__大模型__](#llm)
- [__建模__](#modelling)
- [__研究__](#research)
- [__教程__](#tutorial)
- [__学习__](#learning)
- [__任务__](#task)
- [__平台__](#platform)
- [__工具__](#tool)
- [__杂项__](#misc)



## Mujoco <span id="Mujoco"></span>

[官方的移动任务实现](https://github.com/google-deepmind/dm_control/tree/main/dm_control/locomotion)

[层次世界模型实现人形全身控制](https://github.com/nicklashansen/puppeteer)

[模仿学习基准专注于使用 MuJoCo 执行复杂的运动任务](https://github.com/robfiras/loco-mujoco)

[全身控制的层次世界模型](https://github.com/nicklashansen/puppeteer)

[MyoSuite](https://github.com/MyoHub/myosuite) - 使用 MuJoCo 物理引擎模拟的肌肉骨骼模型要解决的环境/任务的集合，并包含在 OpenAI gym API 中

[将机器人送入家庭并收集数据](https://github.com/AlexanderKhazatsky/household_robots)

[dm_robotics：为机器人研究创建和使用的库、工具和任务](https://github.com/google-deepmind/dm_robotics)

### 人的建模 <span id="humanoid_model"></span>

[OpenSim 肌肉骨骼模型转到 MuJoCo](https://github.com/MyoHub/myoconverter)

[用于 mujoco 模拟的机器人模型集合](https://github.com/anupamkaul/mujoco_menagerie)

[模和模拟人机交互任务的源代码](https://github.com/BaiYunpeng1949/heads-up-multitasker)

[用于研究婴儿认知发展的平台](https://github.com/trieschlab/MIMo) - 可产生视觉、触觉、本体感觉和前庭系统模拟感官输入的模块组成

[Menagerie：MuJoCo物理引擎高质量模型集合](https://github.com/google-deepmind/mujoco_menagerie)

### 比赛 <span id="Mujoco"></span>

[足球射门、乒乓球对打](https://sites.google.com/view/myosuite/myochallenge/myochallenge-2025) 

[网球环境下的多智能体DDPG](https://github.com/m-fili/Tennis_MADDPG)


## ROS  <span id="ros"></span>

[乐聚机器人控制](https://github.com/LejuRobotics/kuavo-ros-opensource) - 包含 Mujoco 仿真环境

[将 ROS 与 MuJoCo 结合使用的封装器、工具和附加 API](https://github.com/ubi-agni/mujoco_ros_pkgs) - 支持 Noetic，- 其他 [mujoco_ros2_control](https://github.com/moveit/mujoco_ros2_control) 

[车道跟随器与强化学习](https://github.com/3N4N/ros-rl)

[基于 ROS2 的户外 SLAM 和自主导航](https://github.com/adeeb10abbas/outdoor_ros2)

[使用 Docker 构建 ROS2 运行环境](https://github.com/cpx0/ros2_docker)

[适用于 ROS 机器人的 FastRLAP 实现、相关的 Gazebo 环境，以及用于越野驾驶的 MuJoCo 环境](https://github.com/kylestach/fastrlap-release)

[一款一体化 ROS 软件包 RoTools](https://github.com/DrawZeroPoint/RoTools) - 用于高级机器人任务调度、视觉感知、路径规划、仿真以及直接/远程操控。它利用 BehaviorTree 实现快速的任务构建和协调，并提供各种实用程序来弥合真实/模拟机器人与高级任务调度程序之间的差距。

## 人 <span id="humanoid"></span>

[使用 MuJoCo 物理引擎模拟的肌肉骨骼模型要解决的环境](https://github.com/MyoHub/myosuite) - 包含在 OpenAI gym API 中

[将 opensim 4.0+ MSK 模型转换为 MuJoCo 格式的工具](https://github.com/MyoHub/myoconverter) - 具有优化的肌肉运动学和动力学

[在MuJoCo中建模和模拟人机交互任务](https://github.com/User-in-the-Box/user-in-the-box) - 用户采用具有感知能力（例如自我中心视觉）的肌肉驱动生物力学模型进行建模，并通过强化学习进行训练以解决交互任务

[利用肌肉学习：拟人化任务中数据效率和鲁棒性的优势](https://github.com/martius-lab/learningwithmuscles)

[从任意跌倒状态起身](https://github.com/tianxintao/get_up_control) - [其他](https://github.com/sumanth-tangirala/get-up-control-trajectories)

[具有内部复杂性的网络模型架起了人工智能与神经科学的桥梁](https://github.com/helx-20/complexity)

[训练和比较人形 AI 代理完成1v1 近战](https://github.com/rodSiry/bagarreio)

[探索与行为相关的神经网络](https://github.com/danielcson/dsc_capstone_q1) - 模仿行为来模拟大脑

[sin-cassie-rl-python](https://github.com/cwjwudi/sin-cassie-rl-python)

[双足步行者的Gym环境](https://github.com/cameronpm1/bpwgym)

[机器人学习的模块化仿真框架和基准](https://github.com/ARISE-Initiative/robosuite) - 包括人形机器人

[使用 mujoco 和类人神经机械模型（而非人形机器人）实现 DeepMimic](https://github.com/DanielCastillo03/DeepMimic_Research)

[仿生机器人](https://github.com/MHaqui/Biomecanical_project)

[构建意识与决策机制](https://github.com/oyako-li/TPG)

[用于机器人、深度强化学习和神经科学研究的 Python 工具](https://github.com/shandilya1998/neurorobotics)



## 强化学习 <span id="rl"></span>

[使用 OpenAI Gym 环境的 xArm6 机器人强化学习框架](https://github.com/julio-design/xArm6-Gym-Env) - 该模型使用深度确定性策略梯度(DDPG) 进行连续动作，并使用后见之明经验回放(HER)

[四足动物-斯坦福小狗文档和训练学习者](https://github.com/Prakyathkantharaju/Quadruped)

[强化学习算法的最小实现及其他强化学习相关实验](https://github.com/PierreLaur/rl_experiments)

[RL 控制和预测方法的实现（PyTorch 中的 DRL）](https://github.com/MythraV/rl_lib)

[基于技能的基于模型的强化学习](https://github.com/clvrai/skimo)

[RNN + PPO pytorch 实现](https://github.com/Amaranth819/RecurrentPPO)

[gym上强化学习的一些实现](https://github.com/paoyw/RL_gym)

[基于运动原语的 RL 算法的测试设置](https://github.com/freiberg-roman/mp-rl)

[使用 Openai-gym 进行强化学习](https://github.com/ashiskb/RL-workspace)

[基于因果模型的强化学习工具包](https://github.com/polixir/causal-mbrl)

[从头开始实现 rl2](https://github.com/neverparadise/RL2_from_scratch)

[不同 RL 和轨迹优化算法的实现](https://github.com/Daniellayeghi/Mujoco_Python_Sandbox)

[在 OpenAI Gym 环境中为超级马里奥兄弟实现强化学习](https://github.com/sampadk04/openai-super-mario-bros) - 使用近端策略优化 (PPO) 算法

[基于模型的连续强化学习中的随机值梯度](https://github.com/facebookresearch/svg)

[RL-Project](https://github.com/AntonFlorey/RL-Project)

[基于状态扰动的无模型强化学习探索](https://github.com/Curiouskid0423/rho_exploration)

[强化学习](https://github.com/murataliev/reinforcement_learning)

[RL_project_2022](https://github.com/oselin/RL_project_2022)

[16831_RL_trading](https://github.com/quantingxie/16831_RL_trading)

[rl_project](https://github.com/fabioscap/rl_project)

[不同深度 Q 网络的有效性研究](https://github.com/00ber/Deep-Q-Networks)

[使用 Policy-Gradient 方法在 OpenAI-Gym 中训练代理](https://github.com/till2/policy-gradient-methods)

[在线强化学习算法](https://github.com/superboySB/mindspore-cleanrl)

[CQL_AWAC_ICQL](https://github.com/bvanbuskirk/CQL_AWAC_ICQL)

[基于 DDPG Keras实现示例的 TD3](https://github.com/jnachyla/usd-proj)

[CleanRL 是一个深度强化学习库](https://github.com/gosu0rZzz/thesis_exp)

[基于 PyTorch 构建的强化学习算法的实现](https://github.com/kartik2309/RLPack) - 它已针对高负载工作负载进行了优化，后端支持 CUDA 和 OpenMP（取决于硬件可用性）

[模块化单文件强化学习算法库](https://github.com/sdpkjc/abcdrl)

[rl](https://github.com/meinczinger/rl)

[保守 Q 学习 (CQL)](https://github.com/Songlm3/CQL)

[Pytorch 实现的 MuZero 用于 Gym 环境](https://github.com/DHDev0/Muzero) - 支持动作空间和观察空间的任何离散、Box 和 Box2D 配置

[基于 Tensorflow 的 DDPG 实现](https://github.com/alexsandercaac/DDPG-Tensorflow) - 使用 DVC 跟踪管道进行实验

[极限 Q 学习：无熵的最大熵强化学习](https://github.com/Div99/XQL)

[利用奖励序列分布进行视觉强化学习的泛化](https://github.com/MIRALab-USTC/RL-CRESP)

[用于稳健深度强化学习的状态对抗性 PPO](https://github.com/Aortz/SA-PPO)

[使用 PPO 训练 SNS](https://github.com/DrYogurt/SNS-PPO)

[OpenAI Gym 环境的强化学习代理](https://github.com/prestonyun/GymnasiumAgents)

[rl](https://github.com/rishiagarwal2000/rl)

[使用深度 Q 学习训练一个代理，让它在一个大的方形环境中收集尽可能多的黄色香蕉](https://github.com/m-fili/DeepQLearning)

[使用基于策略的方法解决 CartPole 问题](https://github.com/m-fili/CartPole_HillClimbing)

[使用交叉熵的连续山地车](https://github.com/m-fili/CrossEntropy)

[强化学习算法的清晰框架和实现](https://github.com/wzcai99/XuanCE-Tiny)

[强化学习 RAINBOW 算法的部分（重新）实现](https://github.com/Dzirik/RAINBOW)

[使用 REINFORCE 算法解决 CartPole](https://github.com/m-fili/reinforce_cartpole)

[探索无模型等变强化学习在运动中的应用](https://github.com/chengh-wang/BBQ)

[基于图像的循环强化学习](https://github.com/CRosero/imgrerl)

[PPOimplementation](https://github.com/Porthoos/PPOimplementation)

[一种解决reacher环境的DDPG算法](https://github.com/m-fili/Reacher_DDPG)

[Q 值函数作为障碍函数](https://github.com/dtch1997/ql_clbf)

[模块化可扩展强化学习](https://github.com/Catwork-LLC/clowder)

[Transformer 作为深度强化学习的骨干](https://github.com/maohangyu/TIT_open_source)

[学徒强化第二阶段](https://github.com/CafeKrem/internship_DL_project)

[使用 Gymnasium 简单实现 PPO](https://github.com/KJaebye/ppo-mujoco)

[TD3](https://github.com/LTBach/TD3)

[reinforcement_learning_dataframe_matching](https://github.com/lhcnetop/reinforcement_learning_dataframe_matching)

[基础设施目标条件强化学习者](https://github.com/spyroot/igc)

[多目标最大后验策略优化](https://github.com/ToooooooT/MOMPO)

[强化学习项目](https://github.com/JereKnuutinen/Reinforcement_learning_project)

[离线强化学习算法](https://github.com/stanford-iris-lab/d5rl)

[用于样本有效目标条件强化学习的度量残差网络](https://github.com/Cranial-XIX/metric-residual-network)

### DQN  <span id="dqn"></span>

[扩展深度 Q 网络模型以支持多模态输入](https://github.com/alexbaekey/DeepRL-multimodal)

[将各种改进与强化学习算法相结合](https://github.com/Deonixlive/modulated-DQN) - 试图遵循三个关键原则：数据效率、可扩展性和更快的训练速度

[基于深度 Q 网络的 TensorFlow 2 强化学习实现](https://github.com/metr0jw/DeepRL-TF2-DQN-implementation-for-TensorFlow-2)

[在 lunarlander 和 bipedalwalker 上测试的 DQN 和 DDPG 的 PyTorch 实现](https://github.com/YingXu-swim/lunarlander-bipedalwalker)

[Q学习在二十一点中的应用](https://github.com/mishonenchev/BlackJackAI)

[重症监护应用的多准则深度 Q 学习](https://github.com/alishiraliGit/multi-criteria-deep-q-learning)

[面向重症监护应用的安全领域知识辅助深度强化学习](https://github.com/notY0rick/multi_criteria_dqn)

[dqn-探索-集成](https://github.com/pranavkrishnamoorthi/dqn-exploration-ensemble)

[使用 OpenAI gym 环境训练 DQN 的简单脚本](https://github.com/eyalhagai12/simple_dqn)

[DQN_AC](https://github.com/bvanbuskirk/DQN_AC)

### AC <span id="ac"></span>

[软动作者-评论家：基于随机动作者的离线策略最大熵深度强化学习](https://github.com/DSSC-projects/soft-actor-critic)

[用于机器人环境交互任务的演员-评论家模型预测力控制器的实验验证](https://github.com/unisa-acg/actor-critic-model-predictive-force-controller)

[SAC](https://github.com/LTBach/SAC)

[针对 CS285 的深度 Q 学习、Actor Critic 和 Soft Actor Critics 算法的实现](https://github.com/phongtheha/Reinforcement-Learning)

[实现的主要算法是 Soft Actor-Critic (SAC)](https://github.com/tomaskak/neural)

[强化学习的数学基础项目 03 - 连续控制](https://github.com/radajakub/soft-actor-critic)

### 多智能体 <span id="multi_agent"></span>

[个别奖励扶助的多智能体强化学习](https://github.com/MDrW/ICML2022-IRAT)

[多任务参与者评论家学习](https://github.com/anzeliu/multitask_actor_critic_learning)

[PyTorch 和 Ray 用于分布式 RL](https://github.com/fatcatZF/rayrl-atari)



### 模仿学习 <span id="imitation"></span>

[通过语境翻译进行观察模仿](https://github.com/medric49/imitation-from-observation) - 一种基于演示训练代理模仿专家的算法

[使机械臂模仿另一只手臂的方向](https://github.com/Kisfodi/MimicArm)

[通过模仿行为来理解大脑](https://github.com/jimzers/DSC180B-A08)

[利用扩散模型作为高表达性的策略类别，用于行为克隆和策略正则化](https://github.com/ep-infosec/21_twitter_diffusion-rl)

[模仿预训练](https://github.com/davidbrandfonbrener/imitation_pretraining)

[柔性机器人非线性模型预测控制的安全模仿学习](https://github.com/shamilmamedov/flexible_arm)

[Imitation-Learning](https://github.com/vsreyas/Imitation-Learning)

[易于运行的模仿学习和强化学习框架](https://github.com/Ericonaldo/ILSwiss)

[四足动物行为克隆实验](https://github.com/dtch1997/quadruped-bc)

[通过行为学习进行观察模仿](https://github.com/medric49/ifobl)

### 分层强化学习 <span id="hierarchy"></span>

[使用 Pytorch、OpenAI Gym 和 Mujoco 进行机器人分层强化学习](https://github.com/mrernst/hrl_robotics_research)

[hierarchy_Reinforcement_Learning](https://github.com/yourlucky/hierarchy_Reinforcement_Learning)

[分层强化学习](https://github.com/yourlucky/Picker-Hierarchical-RL)

[测试稳定比例微分控制器中 mujoco 的 SPD 实现](https://github.com/rohit-kumar-j/SPD_Controller_Mujoco)

### 逆强化学习 <span id="inverse_rl"></span>

[通过贝叶斯心理理论进行稳健逆强化学习](https://github.com/ran-weii/btom_irl)

[Inverse_RL](https://github.com/werkaaa/Inverse_RL)


### 元强化学习 <span id="meta_rl"></span>

[评估复杂任务分布中的元强化学习算法](https://github.com/mhelabd/Meta-RL)

[人人皆可学习的元学习](https://github.com/ChoiDae1/Meta-learning-Study)

[PAC-贝叶斯离线元强化学习](https://github.com/outshine-J/PAC-Bayesian-Offline-Meta-Reinforcement-Learning)

[Meta QLearning 实验优化机器人步行模式](https://github.com/gokulp01/meta-qlearning-humanoid)

[Meta-World 是一个开源基准，用于开发和评估用于连续控制机器人操作环境的多任务和元强化学习算法](https://github.com/Farama-Foundation/Metaworld)

[Optm-MetaRL](https://github.com/LucienJi/OptmMeta-RL)

[分布式分层元强化学习器](https://github.com/spyroot/DH-MAML)


## 感知 <span id="perception"></span>

[物体检测与追踪](https://github.com/GUVENAli/yolov5-object-detection-tracking)

[动作捕捉环境](https://github.com/hartikainen/mocap-environments)


## 规划 <span id="planning"></span>

[外展机器人学习决策](https://github.com/chrisyrniu/neurips22_outreach_robot_learning_for_decision_making)

[MoCapAct和dm_control的扩展，用于避障任务](https://github.com/Team-Zircon/ZirconProject)

[使用 Graph Transformer 规划装配序列](https://github.com/AIR-DISCOVER/ICRA_ASP)

[虚拟工厂环境中的自主Transpalet导航项目](https://github.com/erdnj/Robotics)

[通过对不确定道路系统进行持续数据收集来进行路线优化](https://github.com/BroknLogic/dmaProject) - 包裹递送算法，使其能够在更新道路系统信息的同时安排递送。

[用于欠驱动机器人手的硬件/软件协同优化](https://github.com/adikulkarni11/Underactuated-Robotic-Hands)


## 控制 <span id="control"></span>

[一种基于视觉模型的强化算法 Dreamer](https://github.com/adityabingi/Dreamer) - 它学习一个世界模型，该模型从高级像素图像中捕捉潜在动态，并完全在从学习到的世界模型中想象的部署中训练控制代理

[基于对比示例的控制](https://github.com/khatch31/laeo)

[机器人蛇形运动](https://github.com/alexandrubalotescu/Robot-Snake-Locomotion)

[MPC_MBPO](https://github.com/bvanbuskirk/MPC_MBPO)

[通过在连接每条腿的两个连杆和躯干的八个铰链上施加扭矩来协调四条腿向前移动](https://github.com/ake1999/TD3_Ant_v4)

[探索关节空间中潜在地标](https://github.com/dtch1997/latent-landmarks)

[ Kinova Gen3 机器人控制](https://github.com/jerrywrx/kinova_control)

[为 Allegro Hand（一款拥有 16 个独立可控关节的机械手）实现了比例积分微分 (PID) 控制器](https://github.com/premtc/Human_Robot_Hand_Grasping_Mujoco)

[利用强化学习和 VAE 控制千足虫](https://github.com/savanagrawal/Millipede-Control-with-Reinforcement-Learning-and-VAEs)

[刚体操作](https://github.com/barikata1984/rigid-body-manipulation)

[倒立摆](https://github.com/dhruvthanki/mj_InvertedPendulum) - 使用基于优化的操作空间控制来跟踪双摆的末端执行器位置


## 测试 <span id="test"></span>

[评估了 RL 领域的特征提取](https://github.com/clement-chupin/BenchNeuralNework)

[l2r 基准测试](https://github.com/arav-agarwal2/l2r-benchmarks)

[Mujoco测试平台](https://github.com/AIRLABkhu/MujocoTestbed)

[用于测试/评估 mujoco 物理模拟器的沙盒](https://github.com/implementedrobotics/mujoco-sandbox)

[offline_rl_benchmark_by_argo](https://github.com/hjcwuhuqifei/offline_rl_benchmark_by_argo)

[验证gymnasium_roboticsmujoco 环境的 MuJoCo 模型变化](https://github.com/Kallinteris-Andreas/gym-mjc-v5-model-validation) - [其他](https://github.com/Kallinteris-Andreas/gymnasium-mujuco-v5-envs-validation)

[视觉泛化的强化学习基准](https://github.com/gemcollector/RL-ViGen)

[mujoco_test](https://github.com/Geryyy/mujoco_test)

[Safety-Gymnasium：统一的安全强化学习基准](https://github.com/PKU-Alignment/safety-gymnasium)

[机器人优化基准](https://github.com/dawsonc/robotics_optimization_benchmarks)

[RLXBench](https://github.com/oladayosolomon/RLXBench)

[mujoco-motoman-test](https://github.com/DanManN/mujoco-motoman-test)

[BenchSuite](https://github.com/LeoIV/BenchSuite)

[使用 PyTorch 的 functorch 的稳定基线](https://github.com/functorch/sbft)

[l2r 基准测试](https://github.com/learn-to-race/l2r-lab)

[针对机器人操作的基准测试集](https://github.com/xiangyanfei212/RMBench-2022)

## 数据 <span id="data"></span>

[用于模拟人形控制的多任务数据集](https://github.com/microsoft/MoCapAct)

[使用 MuJoCo 生成的数据集的 NeRF 的 Pytorch 实现](https://github.com/volunt4s/mujoNeRF)

[包含 Machines in Motion 实验室中使用的机器人描述](https://github.com/machines-in-motion/mim_robots)

[R2D2：住宅机器人演示数据集](https://github.com/zehanma/r2d2_autolab)

[域随机化示例](https://github.com/ugo-nama-kun/domain_randomization)

[从完全离线策略数据中学习](https://github.com/dldnxks12/Offline-RL)

[结果分享](https://github.com/ykh6581394/resultshare)

[训练或发展可控且多样化的级别生成器](https://github.com/smearle/control-pcgrl)

[可变形物体操控沙盒](https://github.com/nicholasprayogo/dom_sandbox)


## 大模型 <span id="llm"></span>

[将 ChatGPT 集成到机器人控制过程中，以实现零样本规划和控制](https://github.com/andrei-calin-dragomir/gpt-controller)

[使用 3GPP 文件微调不同的 LLM](https://github.com/hang-zou/LLM_FT_3GPP)

[大型语言模型项目想法](https://github.com/abdalrahmenyousifMohamed/LLM)

[扩大规模并精简：语言引导的机器人技能习得](https://github.com/real-stanford/scalingup)


## 建模 <span id="modelling"></span>

[获取机器人 URDF](https://github.com/Engineering-Geek/robot_arm)

[MuJoCo 的 3x3x3 拼图立方体模型](https://github.com/kevinzakka/mujoco_cube)

[主动视觉强化学习的环境集合](https://github.com/elicassion/active-gym)

[扩展控制器环境](https://github.com/RX-00/ext_ctrl_envs) - 包括推车上的倒立摆、弹簧加载倒立摆

[飞行和漂浮模型，例如四旋翼飞行器、悬挂有效载荷的四旋翼飞行器等](https://github.com/vkotaru/udaan)

[使用 V-HACD 将凹网格分解为凸包来创建 MJCF 模型](https://github.com/tudorjnu/stl2mjcf)

[用于训练四足机器人的gym](https://github.com/dtch1997/gymnasium-quadruped)

[建筑物内的测试环境的 3D 模型](https://github.com/AIMotionLab-SZTAKI/AIMotionLab-Virtual)

## 研究 <span id="research"></span>

[谷歌研究库](https://github.com/google-research/google-research)

[信念状态编码器/解码器](https://github.com/lucidrains/anymal-belief-state-encoder-decoder-pytorch) - 似乎产生了一种可与波士顿动力手工算法（四足动物 Spot）相媲美的策略

[包含SoftGym环境的基准算法](https://github.com/ducphuE10/equiRL)

[进化机器人 Python——脑体协同优化框架](https://github.com/Co-Evolve/erpy)

[任意跌倒状态起身](https://github.com/TeshShin/UE5-GetupControl) - UE5

[使用新颖的 Hyena 连续卷积核作为 Transformer 的替代方案，以便在离线强化学习中高效地捕捉长距离依赖关系](https://github.com/andrewliu2001/hpml-project)

[DeFog: 随机丢帧下的决策变换器](https://github.com/hukz18/DeFog)

[通过准度量学习实现最优目标达成强化学习](https://github.com/quasimetric-learning/quasimetric-rl)

[基于 DeepMind Control Suite 实现的具有对称性的 MDP 集合](https://github.com/sahandrez/symmetry_RL)

[利用多源工作负载知识促进指数顾问学习](https://github.com/XMUDM/BALANCE)

[保守离线策略评估的幻觉控制](https://github.com/tobiabir/hambo)

[HIQL：以潜在状态为行动的离线目标条件强化学习](https://github.com/seohongpark/HIQL)

[符合道德规范的 rl](https://github.com/MasonN808/ethically-compliant-rl)

[使用封闭式策略改进算子的离线强化学习](https://github.com/cfpi-icml23/cfpi)

[引入基于评论家估计的不确定性抽样](https://github.com/juliusott/uncertainty-buffer)

[提升 AI 对齐研究工程技能的资源](https://github.com/callummcdougall/ARENA_2.0)

[自适应强化学习的表征学习](https://github.com/stevenabreu7/adaptiveRL2) - 使用可微分可塑性、状态空间模型和深度强化学习

[用示例代替奖励：通过递归分类进行基于示例的策略搜索 的 pytorch 实现](https://github.com/Ricky-Zhu/RCE)

[具有大型语言模型的辩证多机器人协作](https://github.com/MandiZhao/robot-collab)

[通过多任务策略提炼解决任务干扰](https://github.com/AndreiLix/mutlitask_policy_distillation)

[使用去噪扩散概率模型的轨迹生成、控制和安全性](https://github.com/nicob15/Trajectory-Generation-Control-and-Safety-with-Denoising-Diffusion-Probabilistic-Models)

[策略转移终身RL](https://github.com/zhouzypaul/policy-transfer-lifelong-rl)

[基于幻觉输入偏好的强化学习](https://github.com/calvincbzhang/hip-rl)

[对比贝叶斯自适应深度强化学习](https://github.com/ec2604/ContraBAR)

[可控性感知的无监督技能发现](https://github.com/seohongpark/CSD-locomotion)

[深度方差加权（DVW）的官方实现](https://github.com/matsuolab/Deep-Variance-Weighting-MinAtar)

[合成经验回放 (SynthER) 是一种基于扩散的方法](https://github.com/conglu1997/SynthER) - 可以对强化学习 (RL) 代理收集的经验进行任意上采样，从而大幅提升采样效率和扩展优势

[受控的多样性与偏好：迈向学习多样化的所需技能](https://github.com/HussonnoisMaxence/CDP)

[SNS-Toolbox 方法论文中关于不同类型优化的代码](https://github.com/DrYogurt/SNS-Toolbox-Optimization)

[具有重要性加权参与者-学习者架构的可扩展分布式深度强化学习](https://github.com/KSB21ST/IMPALA_memory_maze)

[从受限专家演示中学习软约束](https://github.com/ashishgaurav13/ICL)

[离线强化学习的扩散策略](https://github.com/Zhendong-Wang/Diffusion-Policies-for-Offline-RL)

[通过中间目标的监督学习进行强化学习](https://github.com/StanfordAI4HI/waypoint-transformer)

[用于测试概念或尝试重现孤立问题的简单区域](https://github.com/EricCousineau-TRI/repro)

[通过扩散概率模型进行强化学习的策略表示](https://github.com/BellmanTimeHut/DIPO)

[突破强化学习中重放率障碍，实现连续控制](https://github.com/proceduralia/high_replay_ratio_continuous_control)

[可控性感知的无监督技能发现](https://github.com/seohongpark/CSD-manipulation)

[解决 OpenAI Gym 中的神经元中间算法遗传算法的问题](https://github.com/DouglasDacchille/GA-InvertedPendulum)

[从梦想到控制：通过潜在想象力学习行为，在 PyTorch 中实现](https://github.com/juliusfrost/dreamer-pytorch)

[预测模型延迟校正强化学习](https://github.com/CAV-Research-Lab/Predictive-Model-Delay-Correction)

[最佳评估成本跟踪](https://github.com/JudithEtxebarrieta/OPTECOT)

[等变模型在潜在对称域中的惊人有效性](https://github.com/pointW/extrinsic_equi)

[基于目标的随机优化替代方法](https://github.com/WilderLavington/Target-Based-Surrogates-For-Stochastic-Optimization)

[利用进化策略进化人工神经网络实现虚拟机器人控制](https://github.com/kenjiroono/NEAT-for-robotic-control)

[机器人环境的安全迁移学习](https://github.com/f-krone/SafeTransferLearningInChangingEnvironments)

[基于 DeepMind Control Suite 实现的具有变化奖励和动态的上下文 MDP](https://github.com/SAIC-MONTREAL/contextual-control-suite)

[SIMCSUM](https://github.com/timkolber/mtl_sum)

[研究基于模型的强化学习中的不确定性量化](https://github.com/aidanscannell/unc-mbrl)

[通过压缩学习选项](https://github.com/yidingjiang/love)

[NaturalNets](https://github.com/neuroevolution-ai/NaturalNets)

[去噪 MDP：比世界本身更好地学习世界模型](https://github.com/facebookresearch/denoised_mdp)

[深度强化学习中的首因偏差](https://github.com/evgenii-nikishin/rl_with_resets) - 深度强化学习代理的 JAX 实现，带有重置功能

[基于近似模型的安全强化学习屏蔽](https://github.com/sacktock/AMBS)

[利用扩散模型作为高表达性的策略类别](https://github.com/twitter/diffusion-rl) - 用于行为克隆和策略正则化

[构建目标驱动的具身化大脑模型](https://github.com/ccnmaastricht/angorapy)

[稳定神经近似的逆向经验重放](https://github.com/google-research/look-back-when-surprised) - [其他](https://github.com/llv22/google-research-forward)

## 教程 <span id="tutorial"></span>

[MuJoCo 模拟平台入门教程](https://github.com/tayalmanan28/MuJoCo-Tutorial)

[适合所有人的人工智能书籍](https://github.com/YeonwooSung/ai_book)


## 学习 <span id="learning"></span>

[采样策略梯度扩展](https://github.com/BharathRajM/Sampled-Policy-Gradient-and-variants)

[强化学习教程](https://github.com/weiminn/reinforcement_learning_tutorials)

[cs285](https://github.com/johnviljoen/cs285)

[CS 285 作业](https://github.com/LeslieTrue/cs285_fall22_hw_sol)

[利用 MuJoCo 进行深度强化学习](https://github.com/danimatasd/MUJOCO-AIDL)

[大学强化学习考试（9 CFU）材料的组成部分](https://github.com/ProjectoOfficial/ReinforcementLearningProject)

[2022 年高级机器学习 (AML) 课程项目的最终代码](https://github.com/marcopra/RL-vision-based)

[关于课程作业的一些解决方案](https://github.com/RbingChen/GoodGoodStudy)

[关于 dm_control 的 AI 原理强化学习项目](https://github.com/Otsuts/SAC-GAARA)

[RL相关项目](https://github.com/Pippo809/rl_projects) - 模仿学习、策略梯度

[用于强化学习研究的快速且可定制的gym兼容零售店环境](https://github.com/kenminglee/COMP579-FinalProject)

[本课程包括建模不确定性、马尔可夫决策过程、基于模型的强化学习、无模型强化学习函数近似、策略梯度、部分可观察的马尔可夫决策过程](https://github.com/SpringNuance/Reinforcement-Learning)

[使用 Gymnasium 和 Mujoco 构建强化学习的示例](https://github.com/ramonlins/rlmj)

[cs285深度强化学习](https://github.com/notY0rick/cs285_deep_reinforcement_learning)

[解决Gym问题和其他机器学习实践](https://github.com/qio1112/GymSolutions)

[人工智能中心 2023 年春季项目的存储库](https://github.com/xiaoxiaoshikui/Machine-Learning-Project-for-ETH-AI-Center)

[加州大学伯克利分校 CS285 深度强化学习 2022 年秋季](https://github.com/xd00099/CS285-DeepReinforcementLearning-Berkeley)

[fa22-cs285-project](https://github.com/inferee/fa22-cs285-project)

[一些流行的深度强化学习算法的实现](https://github.com/Manaro-Alpha/RL_Algos)

[DeepRL-CS285](https://github.com/minyonggo/DeepRL-CS285)

[一些训练和微调决策转换器的实验](https://github.com/bhaveshgawri/decision-transformer-transfer-learning)

[学习强化学习的笔记](https://github.com/asuzukosi/reinforcement-learning-study-notes)

[强化学习](https://github.com/MarcoDiFrancesco/reinforcement-learning)

[cs285hw](https://github.com/Grant-Tao/cs285hw)

[CS 285 佳乐的作业](https://github.com/JialeZhaAcademic/UCB-CS-285)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/Manaro-Alpha/CS285_DeepRL_hw_sols)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/Hoponga/cs285)

[伯克利 CS 285的作业：深度强化学习、决策和控制 2021](https://github.com/anna-ssi/UCBerkley-CS285-homework-2021)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/Applewonder/CS285-2022)

[伯克利 CS 285 深度强化学习、决策与控制 2022 年秋季](https://github.com/Arenaa/CS-285)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/brunonishimoto/cs285-drl)

[XAI611项目提案](https://github.com/tlatjddnd101/xai611_project_proposal_2023)

[dm_control 的 AI 原理强化学习项目](https://github.com/Otsuts/SAC-GAARA)

[关于机器学习和控制的笔记本](https://github.com/alefram/notebooks)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/rsha256/CS285_HW)

[加州大学伯克利分校 cs 285 课程作业](https://github.com/smoteval/reinforcement_learning_berkeley_assignments)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/safooray/rl_berkeley)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/A-Abdinur/RLHomeWorkTutorials)

[CS234 最终项目](https://github.com/chuanqichen/CS234_Final_Project)

[强化学习课程练习的实现](https://github.com/juuso-oskari/ReinforcementLearning)

[强化学习练习](https://github.com/taehwanHH/prac_Reinforcement-Learning/tree/main/ReinforcementLearningAtoZ-master)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/jongwoolee127/cs285-homework)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/akashanand93/rl)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/NewGamezzz/cs285-DeepRL-Fall2022)

[USD-22Z-Projekt](https://github.com/olek123456789/USD-22Z-Projekt)

[CS 285 深度强化学习课程材料](https://github.com/Curiouskid0423/deeprl)

[IASD 硕士深度强化学习课程的作业](https://github.com/webalorn/DRL_assignements) - 基于课程Berkeley CS 285：深度强化学习、决策和控制

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/chengwym/CS285)

[学习CS285时做的作业](https://github.com/lijunxian111/CS285)

[cs285HW](https://github.com/YuquanDeng/cs285HW)

[CS839-最终项目](https://github.com/CohenQU/CS839-FinalProject)

[831project](https://github.com/miao3210/831project)

[强化学习课程的练习和项目代码](https://github.com/ChristianMontecchiani/RL_course)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/kiran-ganeshan/cs285)

[cmu_rob831_fall](https://github.com/AsrarH/cmu_rob831_fall)

[高级机器学习（AML）课程项目启动代码](https://github.com/gabrieletiboni/aml22-rl)

[数据分析与人工智能课程考试项目起始代码](https://github.com/gabrieletiboni/daai22-rl)

[毕业论文](https://github.com/Balssh/Thesis)

[CS285 的最终项目代码库：加州大学伯克利分校的深度强化学习](https://github.com/chirag-sharma-00/CS285-Project)

[CS285-Research-Project](https://github.com/aaggarw99/CS285-Research-Project)

[HPC_3](https://github.com/LBatov/HPC_3)

[使用 KNN 算法根据观察结果预测动作](https://github.com/abhayrcodes/cs285knn)

[一个利用强化学习、线性代数和机器人技术概念的实践项目](https://github.com/virajbshah/rl-inator)

[2022/2023 自主代理课程练习](https://github.com/lucalazzaroni/Autonomous-Agents)

[CS 285 家庭作业：深度强化学习](https://github.com/reecehuff/CS285_Homeworks)

[CIFAR-10-练习](https://github.com/RETELLIGENCE-IWEN/CIFAR-10-Practice)

[CS285 - 深度强化学习资料](https://github.com/Naghipourfar/cs285)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/GeoffNN/c285-HW)

[策略梯度](https://github.com/bvanbuskirk/PolicyGradient/tree/main/hw2)

[ELEC-E812课程作业](https://github.com/johnson-li/ELEC-E8125)

[用于 CS 391R 课程项目的击球机器人](https://github.com/jbonyun/cs391r)

[ÚFAL 课程 NPFL122](https://github.com/ufal/npfl122)

[伯克利 CS 285的作业：深度强化学习、决策和控制](https://github.com/berkeleydeeprlcourse/homework_fall2022)

## 任务 <span id="task"></span>

[基于物理的乒乓球](https://github.com/AayushSabharwal/physics-pong)

[四轴飞行器有效载荷抓取与运输轨迹规划与控制设计](https://github.com/antalpeter1/tdk-2022)

[空气曲棍球挑战赛](https://github.com/AirHockeyChallenge/air_hockey_challenge)

[可用于开发机器人 3D 装箱问题的求解器的gym环境](https://github.com/floriankagerer/bed-bpp-env)

[测试 RL 在量子控制中的应用](https://github.com/AnikenC/QuantumControlWithRL) - 特别关注电路级和脉冲级门校准任务

[用于机器人插入任务的 MuJoCo 模拟](https://github.com/gintautas12358/Mujoco-Eleanor)

[曲棍球环境中的强化学习](https://github.com/JSteegmueller/The-Q-Learners)

[通过双手灵活性掌握钢琴演奏技巧](https://github.com/halo2718/Commented-RoboPianist)

[自动驾驶汽车SoC](https://github.com/shambhavii13/Autonomous_Moving_Vehicle_SoC)

[使用 Panda 的非常简单的 MuJoCo 拾取和放置任务](https://github.com/volunt4s/Simple-MuJoCo-PickNPlace)

[三足蚂蚁](https://github.com/ugo-nama-kun/three_legged_ant)

[使用 NEAT RL 算法解决 ATARI Retro Pong](https://github.com/MatSaf123/neat-retro-pong)

[蚂蚁六腿环境](https://github.com/savanagrawal/Gymnasium-MuJoCo-Ant-Six-Legs)

[在 iCub 人形机器人上重现与 RL 项目相关的灵巧操作实验的代码](https://github.com/hsp-iit/rl-icub-dexterous-manipulation)

[空气曲棍球挑战赛的源代码](https://github.com/sombitd/AirRL)

[四旋翼飞行器利用钩式机械手抓取和运输有效载荷](https://github.com/AIMotionLab-SZTAKI/quadrotor-hook-1DoF)

[人工生命模拟器](https://github.com/Limonka11/ArtificialLifeSimulator) - 结合了 PPO 和进化算法

[四轴飞行器](https://github.com/mnasser02/gymnasium_quadcopter)

[山地车强化学习](https://github.com/ryota0051/rl-mountaincar)

[DRL_Taxi_Custom](https://github.com/Douglch/DRL_Taxi_Custom)

[工业机器人机械手（KUKA KR16-2）接住发出的网球](https://github.com/LuanGBR/Kuka_RL_Control)

[在 OpenAI gym 中从 Ant-v4 环境衍生的自定义环境中实现 PPO，以学习穿越模板障碍](https://github.com/Ketan13294/PPO-ant)

[使用凸模型预测控制（MPC）的四足动物运动的 Python 实现](https://github.com/yinghansun/pympc-quadruped)

[添加新环境：四旋翼飞行器](https://github.com/bzx20/Brax_mydrone)

[基于深度学习的代理使用 GUI 玩贪吃蛇游戏](https://github.com/wanghan8866/MyThirdYearProject1)

[使用 MyCobot 的机械臂任务强化学习框架](https://github.com/matinmoezzi/MyCobotGym)

[通过深度强化学习灵巧地弹奏钢琴](https://github.com/google-research/robopianist)

[使用 RL 和低级控制器控制四轴飞行器](https://github.com/Prakyathkantharaju/quadcopter)

## 平台 <span id="platform"></span>

[Gym](https://github.com/openai/gym) - 用于开发和比较强化学习算法，它提供了一个用于在学习算法和环境之间进行通信的标准 API，以及一组兼容该 API 的标准环境。已迁移至 [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) 

[Agility Robotics 的 Cassie 机器人的 mujoco 模拟因尽可能快地向前行走/奔跑而获得奖励](https://github.com/perrin-isir/gym-cassie-run) - [其他](https://github.com/AlexandreChenu/gym_cassie_dcil)

[强化学习的高性能分布式训练框架](https://github.com/PaddlePaddle/PARL)

[在本地、Slurm 和 GCP 上运行 RL 代码](https://github.com/bstadie/rl_starter_kit)

[开发用于机器人任务的 RL 代理的环境](https://github.com/alefram/TEG)

[可定制的多用途生产系统框架]()

[包含 REINFORCE、AC2、SAC 和 PPO 等热门算法的实现，并集成到 Gymnasium 环境](https://github.com/zachoines/RL-Grimoire)

[ReDMan 是一个开源模拟平台，为可靠的灵巧操作提供了安全 RL 算法的标准化实现](https://github.com/PKU-Alignment/ReDMan)

[Ray 由一个核心分布式运行时和一组用于加速 ML 工作负载的 AI 库组成](https://github.com/ray-project/ray)

[Jax 中实现的强化学习算法集合](https://github.com/nutorbit/rl-zoo)

[机器人学习的统一框架](https://github.com/vikashplus/robohive)

[一种多功能模块化框架，使用框图方法运行/模拟动态系统](https://github.com/implementedrobotics/BlockFlow)

[在加速器硬件上进行大规模并行刚体物理模拟](https://github.com/google/brax)

[通才generalist](https://github.com/grahamannett/generalist)


## 工具 <span id="tool"></span>

[一个基于 C++ 的批处理环境池 EnvPool](https://github.com/sail-sg/envpool) - 基于 C++ 的高性能并行环境执行引擎（矢量化环境），适用于通用 RL 环境

[用于强化学习的机器人模拟环境集合](https://github.com/Farama-Foundation/Gymnasium-Robotics)

[用于处理MuJoCo Python 绑定和dm_control 的实用程序](https://github.com/kevinzakka/mujoco_utils)

[通过潜在想象力进行学习的行为](https://github.com/falheisen/BE_dreamer)

[Mechcat Mujoco 查看器](https://github.com/ialab-yale/meshcat-mujoco)

[加速多智能体强化学习的程序环境生成](https://github.com/tomalner18/MAax)

[结构化的模块化设置，用于使用 Ray RLlib 库训练强化学习 (RL) 模型](https://github.com/artificial-experience/ray-rllib-proto)

[用于机器人操作的模块化接口](https://github.com/raunaqbhirangi/manimo)

[统一原生 MuJoCo (MJC) 和 MuJoCo-XLA (MJX) 中实现的环境的开发和接口](https://github.com/Co-Evolve/mujoco-utils)

[专注于快速构建 DQN 模型原型](https://github.com/odiaz1066/lagomorph)

[包含几个具有正定成本函数的 gym 环境，旨在与稳定的 RL 代理兼容](https://github.com/rickstaa/stable-gym)

[Transformer (TIT) 中 Transformer 作为深度强化学习骨干的官方实现](https://github.com/anonymoussubmission321/TIT_anonymous)

[深度强化学习库，提供高质量的单文件实现](https://github.com/IanWangg/CleanRL-Projects) - [其他](https://github.com/eleninisioti/dirtyrl)

[基于 OpenAI 的 RL 库](https://github.com/MnSBlog/Pinokio.RL)

[提供了一个用于在学习算法和环境之间进行通信的标准 API](https://github.com/fliegla/diffDrive)

[包含 Google Research发布的代码](https://github.com/Rulial/GoogleRe-Pi)

[为硕士论文项目的开发和一些研究活动提供环境](https://github.com/unisa-acg/oracle-force-optimizer)

[RL-Bandits](https://github.com/MuhangTian/RL-Bandits)

[适用于 ML 和 AI 项目/实验的实用小模板](https://github.com/WillieCubed/ai-project-template)

[用于开发和比较强化学习算法的工具包](https://github.com/drakyanerlanggarizkiwardhana/gym)

[使用 Unity ML-Agents (AI) 进行深度强化学习的 3D 包装](https://github.com/bryanat/Reinforcement-Learning-Unity-3D-Packing)

[一些基于 MuJoCo 物理引擎构建的 (C/C++) 示例和扩展](https://github.com/wpumacay/mujoco-ext)

[Mujoco Deepmind 的 Python 绑定中存储库mujoco_panda的实现](https://github.com/varadVaidya/mujoco_arm)



[另一个 Python RL 库](https://github.com/tfurmston/tfrlrl)

[深度强化学习算法的简单单文件实现](https://github.com/vcharraut/rl-basics)

[PyTorch 中基于模型的强化学习的最小库](https://github.com/aidanscannell/mbrl-under-uncertainty)

[标准化机器学习的集成中间件框架](https://github.com/fhswf/MLPro)

[用于将MJCF（MuJoCo 建模格式）机器人模型文件中的有限元素转换为 URDF 的脚本](https://github.com/Yasu31/mjcf_urdf_simple_converter)

[OpenAI Gym 环境使用 pybullet 来制作Tyrannosaur](https://github.com/bingjeff/trex-gym)

[具有研究友好特性的深度强化学习算法的高质量单文件实现（PPO、DQN、C51、DDPG、TD3、SAC、PPG）](https://github.com/vwxyzjn/cleanrl)

[通用人工智能实验室开发的容器](https://github.com/HorizonRoboticsInternal/gail-container)

[强化学习库之间的互操作](https://github.com/epignatelli/helx)

[Mujoco并行模拟](https://github.com/Daniellayeghi/MujocoParallel)

[现代机器学习论文的实现，包括 PPO、PPG 和 POP3D](https://github.com/rusenburn/Axel)

[机器学习和数据科学的附加软件包](https://github.com/nixvital/ml-pkgs)

[Emei 是一个用于开发因果强化学习算法的工具包](https://github.com/polixir/emei)

[YAROK - 另一个机器人框架](https://github.com/danfergo/yarok)

[JAX（Flax）实现具有连续动作空间的深度强化学习算法](https://github.com/ikostrikov/jaxrl) 

[用于处理 MuJoCo 中使用的复合 Wavefront OBJ 文件的 CLI](https://github.com/kevinzakka/obj2mjcf)

[用于执行无梯度优化的 Python 工具箱](https://github.com/facebookresearch/nevergrad)



## 杂项 <span id="misc"></span>

[InterGP](https://github.com/tdardinier/InterGP) - 收集数据、训练代理的流程

[ACM AI 所有研讨会内容代码等的存储库](https://github.com/acmucsd/acm-ai-workshops) - 内容按季度组织

[Docker Wiki 和示例](https://github.com/dotd/docker_wiki)

[ClearML_SCHOOL](https://github.com/MichaelNed/ClearML_SCHOOL)

[一个最小（但独立）的 MuJoCo 模拟器来运行模拟](https://github.com/mosesnah-shared/mujoco-py-v2)

[微电网的 IRIS 代码](https://github.com/shukla-yash/IRIS-Minigrid)

[漩涡示例](https://github.com/DeaconSeals/maelstrom-examples)

[GCPrior](https://github.com/magenta1223/GCPrior)

[人工生命环境](https://github.com/IBN5101/APx-IP)

[简单的独立平面推动焦点示例](https://github.com/UM-ARM-Lab/pushing_FOCUS)

[CQL_sepsis](https://github.com/NanFang2023/CQL_sepsis)

[长期记忆系统](https://github.com/grahamseamans/ltm)

[OCMR](https://github.com/rpapallas/OCMR)

[HybridSim](https://github.com/dmiller12/HybridSim)

[关键用户旅程（CUJ）](https://github.com/woshiyyya/CUJ)

[orax](https://github.com/ethanluoyc/orax)

[MAZE](https://github.com/DuangZhu/MAZE)

[safetyBraxFramework](https://github.com/YusenWu2022/safetyBraxFramework)

[旅游预测项目](https://github.com/VorkovN/TourismPredictionProject)

[InfusedHKS](https://github.com/SSHAY007/InfusedHKS)

[mario-icm](https://github.com/denmanorwatCDS/mario-icm)

[inctxdt](https://github.com/grahamannett/inctxdt)

[web3env](https://github.com/Crinstaniev/web3env)

[T-AIA-902](https://github.com/SimonMonnier/T-AIA-902)

[rl_air-hockey_telluride](https://github.com/lunagava/rl_air-hockey_telluride)

[panda_robot](https://github.com/Yuchengxiao997/panda_robot)

[Praktikum](https://github.com/dima2139/Praktikum)

[VKR](https://github.com/Crechted/VKR)

[crow](https://github.com/ghl3/crow)

[CRA_push](https://github.com/UT-Austin-RobIn/CRA_push)

[包含数据集处理、遗传算法、神经网络等](https://github.com/BartoszBrodowski/computational-intelligence)

[RLproject](https://github.com/ttopeor/RLproject)

[展示了平面二维机器人，但可以立即将其推广到空间三维机器人](https://github.com/jongwoolee127/redundancy_resolution)

[p8_sewbot](https://github.com/kasperfg16/p8_sewbot)

[smarts_git](https://github.com/rongxiaoqu/smarts_git)

[一个沙盒仓库](https://github.com/jeh15/sandbox)

[eth-rl](https://github.com/Crinstaniev/eth-rl)

[用于 SRL 实验室实践的 Jupyter 笔记本](https://github.com/srl-ethz/lab-practice-nbs)

[Demo 282 Guarrera](https://github.com/matteoguarrera/demo_282)

[crazyflie_backflipping](https://github.com/AIMotionLab-SZTAKI/crazyflie_backflipping)

[强化学习实验](https://github.com/parthh01/rl_stuff)

[gail_demo](https://github.com/archana53/gail_demo)

[npds-workspace](https://github.com/ashiskb/npds-workspace)

[Advanced_Software](https://github.com/jiseok99/Advanced_Software)

[Reinforcement-Learning](https://github.com/leopoldlacroix/Reinforcement-Learning)

[fyp_v1](https://github.com/derekcth-wm21/fyp_v1)

[模块化部署](https://github.com/olivier-serris/ModularRollouts)

[玩具 ML 项目](https://github.com/pauldb89/ml)

[skill-basedGCRL](https://github.com/magenta1223/skill-basedGCRL)

[ML/DL/CS 领域的一些工作清单](https://github.com/vshmyhlo/research) - 包括基于 GAN 的图像生成、物体检测、神经机器翻译、相似性和度量学习、语音转文本、文本转语音

[CHTC 上的 Mujoco](https://github.com/NicholasCorrado/CHTC)

[从各种来源尝试的实践课程](https://github.com/BhaskarJoshi-01/Competitive-Programming)

[这是Spinning Up的一个克隆版本，目标是使用最新的 PyTorch 版本](https://github.com/haha1227/spinningup-pytorch)

[TradeMasterReBuild](https://github.com/DVampire/TradeMasterReBuild)

[Fast Campus 强化学习](https://github.com/Junyoungpark/ReinforcementLearningAtoZ)

[Reddit 评论机器人是一个基于 Python 的自动回复器](https://github.com/benparrysapps/comment-meme-generator)

[一些强化学习的算法](https://github.com/etu7912a48/RL_algorithm) - 使用的环境是Windows10上的Python 3.10

[Gym的欠驱动机器人](https://github.com/RonAvr/Underactuated_with_gym)