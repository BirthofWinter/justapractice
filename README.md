# 手语动作生成：PyTorch 渐进式练习

这个项目用合成的二维手臂骨架，演示从基础回归到手语动作生成中的关键想法。面向刚接触神经网络的同学；每个实验尽量只增加一个新概念。无需下载数据集或预训练模型。

## 环境

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## 先运行基础 MLP

python train.py --experiment baseline --epochs 20

程序会生成“抬手、挥手、指向”三类动作。每段动作有 24 帧、5 个二维关节。基础实验把动作类别转换成 one-hot，输入单隐藏层 MLP，预测整段关节坐标。

## 渐进式实验

- baseline：类别到整段姿态的基础回归。
- bone：增加骨骼向量损失，比较关节坐标和肢体结构约束。
- temporal：使用一维卷积整合相邻帧，加入速度损失。
- diffusion：训练去噪网络，从随机噪声迭代生成动作。
- multi_condition：加入方向、手臂长度和人物位置条件，并用五档量化做离散条件的简化演示。
- multi_head：分开预测身体和手部，再拼成完整骨架。
- keyframe：只给首帧、中间帧和末帧，学习补全整段动作。

例如：

python train.py --experiment bone --epochs 5 --samples 256 --batch_size 32

## 训练输出

outputs 目录会保存每轮训练和验证损失的 CSV 文件，以及目标动作、预测动作的图片和训练曲线。图片用不同颜色表示时间顺序。

## 学习顺序

1. 阅读 src/data.py，打印样本形状，理解一个训练样本是什么。
2. 阅读 train.py，跟踪 Dataset、DataLoader、模型、损失、反向传播和参数更新。
3. 比较 baseline、bone、temporal，每次只改变一个因素。
4. 再阅读 diffusion.py，理解加噪训练和反向去噪采样。
5. 最后试多条件、多头预测和关键帧补全。

这里使用的是教学玩具数据和简化模块，不是真实手语识别或高质量视频生成系统。