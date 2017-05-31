# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置带噪声的线性数据
num_examples = 50
# 这里会生成一个完全线性的数据
X = np.array([np.linspace(-2, 4, num_examples), np.linspace(-6, 6, num_examples)])
# 数据展示
# plt.figure(figsize=(4,4))
# plt.scatter(X[0], X[1])
# plt.show

# 这里给数据增加噪声
X += np.random.randn(2, num_examples)
# 数据展示
# plt.figure(figsize=(4,4))
# plt.scatter(X[0], X[1])
# plt.show

# 我们的目标就是通过学习，找到一条拟合曲线，去还原最初的线性数据
# 把数据分离成 x 和 y
x, y = X
# 添加固定为 1 的 bias
x_with_bias = np.array([(1., a) for a in x]).astype(np.float32)

# 用来记录每次迭代的 loss，之后用于展示结果
losses = []
# 迭代次数
training_steps = 50
# 学习率，也叫做步长，表示我们在梯度下降时每次迭代所前进的长度，过大则学不到准确的值，过小则训练太慢
learning_rate = 0.002

# TensorFlow 中所有的代码都需要在 session 中
with tf.Session() as sess:
    # 设置所有的张量，变量和操作
    # 输入层是 x 值和 bias 节点
    input = tf.constant(x_with_bias)
    # target 是 y 的值，需要被调整成正确的尺寸（就是转置一下）
    target = tf.constant(np.transpose([y]).astype(np.float32))
    # weights 是变量，每次循环都会变，这里直接随机初始化（高斯分布，均值 0，标准差 0.1）
    weights = tf.Variable(tf.random_normal([2, 1], 0, 0.1))
    
    # 初始化所有的变量
    tf.global_variables_initializer().run()
    
    # 设置循环中所要做的全部操作
    # 对于所有的 x，根据现有的 weights 来产生对应的 y 值，也就是计算 y = w2 * x + w1 * bias
    yhat = tf.matmul(input, weights)
    # 计算误差，也就是预计的 y 和真实的 y 的区别
    yerror = tf.subtract(yhat, target)
    # 我们想要最小化 L2 损失，是误差的平方，会惩罚大误差，放过小误差
    loss = tf.nn.l2_loss(yerror)
    # 上面的 loss 函数相当于
    # loss = 0.5 * tf.reduce_sum(tf.multiply(yerror, yerror))
    
    # 执行梯度下降
    # 更新 weights，比如 weights += grads * learning_rate
    # 使用偏微分更新 weights
    update_weights = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    # 上面的梯度下降相当于
    # gradient = tf.reduce_sum(tf.transpose(tf.multiply(input, yerror)), 1, keep_dims=True)
    # update_weights = tf.assign_sub(weights, learning_rate * gradient)
    
    # 现在我们定义了所有的张量，也初始化了所有操作（每次执行梯度下降优化）
    for _ in range(training_steps):
        # 重复跑，更新变量
        update_weights.run()
        # 如果没有用 tf.train.GradientDescentOptimizer，就要用下面的方式
        # sess.run(update_weights)
        
        # 记录每次迭代的 loss
        losses.append(loss.eval())
    
    # 训练结束
    betas = weights.eval()
    yhat = yhat.eval()

# 展示训练趋势
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=.3)
fig.set_size_inches(10, 4)
ax1.scatter(x, y, alpha=.7)
ax1.scatter(x, np.transpose(yhat)[0], c="g", alpha=.6)
line_x_range = (-4, 6)
ax1.plot(line_x_range, [betas[0] + a * betas[1] for a in line_x_range], "g", alpha=.6)
ax2.plot(range(0, training_steps), losses)
ax2.set_ylabel("Loss")
ax2.set_xlabel("Training steps")
plt.show()