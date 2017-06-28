# -*- coding: utf-8 -*-
import gzip, binascii, struct, numpy
import matplotlib.pyplot as plt
import numpy as np
import os
from six.moves.urllib.request import urlretrieve
import tensorflow as tf

# 这里需要翻墙，不然下载巨慢
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = "./mnist-data"

# 如果下载好了，那么就不会再次下载
def maybe_download(filename):
    """A helper to download the data files if not present."""
    if not os.path.exists(WORK_DIRECTORY):
        os.mkdir(WORK_DIRECTORY)
    filepath = os.path.join(WORK_DIRECTORY, filename)
    if not os.path.exists(filepath):
        filepath, _ = urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    else:
        print('Already downloaded', filename)
    return filepath

# 这里把所有的训练数据都搞下来
train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

# 如果遇到 MacOS 的 Python as a framework 的问题，参考
# https://stackoverflow.com/questions/29433824/unable-to-import-matplotlib-pyplot-as-plt-in-virtualenv
# 这里我们先看看数据集里有什么，只是一个展示，并不会对图片进行预处理
def sanity_check():
    with gzip.open(test_data_filename) as f:
        # Print the header fields.
        for field in ['magic number', 'image count', 'rows', 'columns']:
            # struct.unpack reads the binary data provided by f.read.
            # The format string '>i' decodes a big-endian integer, which
            # is the encoding of the data.
            print(field, struct.unpack('>i', f.read(4))[0])
        
        # Read the first 28x28 set of pixel values. 
        # Each pixel is one byte, [0, 255], a uint8.
        buf = f.read(28 * 28)
        image = numpy.frombuffer(buf, dtype=numpy.uint8)
    
        # Print the first few values of image.
        print('First 10 pixels:', image[:10])

        # We'll show the image and its pixel value histogram side-by-side.
        # 输出原始图片和直方图，来看看具体的样子
        _, (ax1, ax2) = plt.subplots(1, 2)

        # To interpret the values as a 28x28 image, we need to reshape
        # the numpy array, which is one dimensional.
        ax1.imshow(image.reshape(28, 28), cmap=plt.cm.Greys)
        ax2.hist(image, bins=20, range=[0,255])
        plt.show()

        # 这里是把 [0, 255] 映射到 [-0.5, 0.5] 之后的展示
        # Let's convert the uint8 image to 32 bit floats and rescale 
        # the values to be centered around 0, between [-0.5, 0.5]. 
        # 
        # We again plot the image and histogram to check that we 
        # haven't mangled the data.
        scaled = image.astype(numpy.float32)
        scaled = (scaled - (255 / 2.0)) / 255
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(scaled.reshape(28, 28), cmap=plt.cm.Greys)
        ax2.hist(scaled, bins=20, range=[-0.5, 0.5])
        plt.show()

        # 这里读取 Label，也是一个测试而已
    with gzip.open(test_labels_filename) as f:
        # Print the header fields.
        for field in ['magic number', 'label count']:
            print(field, struct.unpack('>i', f.read(4))[0])

        print('First label:', struct.unpack('B', f.read(1))[0]) 

# 简单显示一下，然后进行之后的步骤
sanity_check()

# 处理图片数据
IMAGE_SIZE = 28
PIXEL_DEPTH = 255

# 这个函数会提取并处理数据
def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
  
    For MNIST data, the number of channels is always 1.

    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and dimensions; we know these values.
        bytestream.read(16)

        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
        data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
        return data

train_data = extract_data(train_data_filename, 60000)
test_data = extract_data(test_data_filename, 10000)

# 这里把处理后的输出展示下
print('Training data shape', train_data.shape)
_, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(train_data[0].reshape(28, 28), cmap=plt.cm.Greys)
ax2.imshow(train_data[1].reshape(28, 28), cmap=plt.cm.Greys)
plt.show()

# 接下来处理标签，我们需要把类别处理成向量，如果是第二类，那么对应 [0,1,0,...,0]，即第二个位置为 1
NUM_LABELS = 10

def extract_labels(filename, num_images):
    """Extract the labels into a 1-hot matrix [image index, label index]."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        # Skip the magic number and count; we know these values.
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    # Convert to dense 1-hot representation.
    return (numpy.arange(NUM_LABELS) == labels[:, None]).astype(numpy.float32)

train_labels = extract_labels(train_labels_filename, 60000)
test_labels = extract_labels(test_labels_filename, 10000)

# 同样测试一下数据
print('Training labels shape', train_labels.shape)
print('First label vector', train_labels[0])
print('Second label vector', train_labels[1])

# 这里我们把数据分成训练、测试和验证集
VALIDATION_SIZE = 5000

validation_data = train_data[:VALIDATION_SIZE, :, :, :]
validation_labels = train_labels[:VALIDATION_SIZE]
train_data = train_data[VALIDATION_SIZE:, :, :, :]
train_labels = train_labels[VALIDATION_SIZE:]

train_size = train_labels.shape[0]

print('Validation shape', validation_data.shape)
print('Train size', train_size)


# 这里开始定义模型
# 从原始输入开始，进行卷积(convolution)和池化(max pooling)处理，在全连接层之前会用 ReLU 
# 作为激活函数，最后用 softmax 来处理输出，把类别信息转化成概率，训练的时候使用 Dropout
#
# 准备模型可以分三步
# 1. 定义变量，来保存我们要训练的权重 weights
# 2. 定义模型的图结构
# 3. 把模型的图分别用于训练、测试和验证（复制几份）

# 先处理好变量
# 从效率的角度考虑，我们会把样本分组，这里是一个组的样本数量
BATCH_SIZE = 60
# 因为是灰度图，所以只有一个通道 channel
NUM_CHANNELS = 1
# 固定随机种子，保证每次的结果一致（不然没办法验证数据和模型）
SEED = 42

# 我们在这里把训练数据和类别标签『喂』给模型，不过这里只是一个占位符(placeholder)
# 真正训练的时候，这些节点在每一步会获取批量数据
train_data_node = tf.placeholder(
  tf.float32,
  shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
train_labels_node = tf.placeholder(tf.float32,
                                   shape=(BATCH_SIZE, NUM_LABELS))

# 对于验证和测试数据，直接保存到一个常量节点里即可（不存在训练的过程，不需要是变量）
validation_data_node = tf.constant(validation_data)
test_data_node = tf.constant(test_data)

# 下面的变量保存着所有的需要训练的权重。后面的参数定义了这些变量的初始化条件
# 用高斯分布初始化卷积的 weights
conv1_weights = tf.Variable(
  tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                      stddev=0.1,
                      seed=SEED))
# 初始的 bias 为 0
conv1_biases = tf.Variable(tf.zeros([32]))
# 第二层的卷积权重，32 个输入（对应上面的 32），然后下面是 64 维
conv2_weights = tf.Variable(
  tf.truncated_normal([5, 5, 32, 64],
                      stddev=0.1,
                      seed=SEED))
# 同理，bias 也是 64 维，但是这里用 0.1
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
# 然后是一个全连接的网络，共 512 维，为什么呢，因为我们有卷积和池化的存在，所以是 32*64/4
# (?这里我也不是很确定)
fc1_weights = tf.Variable(  # fully connected, depth 512.
  tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                      stddev=0.1,
                      seed=SEED))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
fc2_weights = tf.Variable(
  tf.truncated_normal([512, NUM_LABELS],
                      stddev=0.1,
                      seed=SEED))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]))

print('变量设置完毕')

# 定义好了各种需要训练的变量，我们可以在 TensorFlow 图中把这些变量连起来了
# 这里我们用一个函数来返回我们需要的 tf graph，这里有一个参数来控制是训练还是其他
# 如果是训练，我们需要使用 dropout

def model(data, train=False):
    """模型定义"""
    # 2D 卷积，使用相同 padding，意思是输入的 feature 大小和输出的一致，
    # strides 是一个四维数组 [image index, y, x, depth]
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')

    # 对卷积和偏置做 ReLU 操作
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    # 池化，这里我们的 pooling window 是 2，每个 stride 是 2
    # Max pooling. The kernel size spec ksize also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')
    conv = tf.nn.conv2d(pool,
                        conv2_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')

    # 把 feature map 转为 2D 矩阵，并传给全连接网络
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    # Add a 50% dropout during training only. Dropout also scales
    # activations such that no rescaling is needed at evaluation time.
    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

# 定义了图的基本结构，我们就可以分别为 训练、测试和验证来提取模型了（也会根据不同的类型做一些自定义）
# train_prediction 保存训练的图，使用 cross-entropy loss 和 weight regularization
# 我们也会在训练的过程中调整学习率（通过 exponential_decay 操作来完成，会使用 MomentumOptimizer）

# 验证和测试的图比较简单，我们只需要使用验证和测试集作为输入，用 softmax 分类器作为输出

# 训练的计算
# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, True)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
  labels=train_labels_node, logits=logits))

# L2 正则化
# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0)
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
  0.01,                # Base learning rate.
  batch * BATCH_SIZE,  # Current index into the dataset.
  train_size,          # Decay step.
  0.95,                # Decay rate.
  staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate,
                                       0.9).minimize(loss,
                                                     global_step=batch)

# Predictions for the minibatch, validation set and test set.
train_prediction = tf.nn.softmax(logits)
# We'll compute them only once in a while by calling their {eval()} method.
validation_prediction = tf.nn.softmax(model(validation_data_node))
test_prediction = tf.nn.softmax(model(test_data_node))

# 准备好了训练、测试和验证的模型之后，我们就可以来真正执行训练了。
# 所有的操作都需要在 session 中，在 python 中像是
# with tf.Session() as s:
# ...training / test / evaluation loop...

# 但是我们这里想要保持 session 方便我们去探索训练的过程，使用 InteractiveSession

# 我们先创建一个 session 并初始化我们刚才定义的变量
s = tf.InteractiveSession()

# Use our newly created session as the default for subsequent operations.
s.as_default()

# 初始化刚才定义的变量
tf.global_variables_initializer().run()

# 我们现在可以开始训练了，这里我们用 minibatch 的方法（而不是一次只训练一个样本）
BATCH_SIZE = 60

# 提取第一个 batch 的数据和标签
# Grab the first BATCH_SIZE examples and labels.
batch_data = train_data[:BATCH_SIZE, :, :, :]
batch_labels = train_labels[:BATCH_SIZE]

# This dictionary maps the batch data (as a numpy array) to the
# node in the graph it should be fed to.
feed_dict = {train_data_node: batch_data,
             train_labels_node: batch_labels}

# Run the graph and fetch some of the nodes.
_, l, lr, predictions = s.run(
  [optimizer, loss, learning_rate, train_prediction],
  feed_dict=feed_dict)

print(predictions[0])

# The highest probability in the first entry.
print('First prediction', numpy.argmax(predictions[0]))

# But, predictions is actually a list of BATCH_SIZE probability vectors.
print(predictions.shape)

# So, we'll take the highest probability for each vector.
print('All predictions', numpy.argmax(predictions, 1))

print('Batch labels', numpy.argmax(batch_labels, 1))

correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(batch_labels, 1))
total = predictions.shape[0]

print(float(correct) / float(total))

confusions = numpy.zeros([10, 10], numpy.float32)
bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(batch_labels, 1))
for predicted, actual in bundled:
  confusions[predicted, actual] += 1

plt.grid(False)
plt.xticks(numpy.arange(NUM_LABELS))
plt.yticks(numpy.arange(NUM_LABELS))
plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest')

def error_rate(predictions, labels):
    """Return the error rate and confusions."""
    correct = numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1))
    total = predictions.shape[0]

    error = 100.0 - (100 * float(correct) / float(total))

    confusions = numpy.zeros([10, 10], numpy.float32)
    bundled = zip(numpy.argmax(predictions, 1), numpy.argmax(labels, 1))
    for predicted, actual in bundled:
        confusions[predicted, actual] += 1
    
    return error, confusions

# 这里训练 n 轮，每轮都是 minibatch
train_round = 3
for i in range(train_round):
    print("Training Round ", i+1 )
    # Train over the first 1/4th of our training set.
    steps = train_size // BATCH_SIZE
    for step in range(steps):
        # Compute the offset of the current minibatch in the data.
        # Note that we could use better randomization across epochs.
        offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
        batch_data = train_data[offset:(offset + BATCH_SIZE), :, :, :]
        batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
        # This dictionary maps the batch data (as a numpy array) to the
        # node in the graph it should be fed to.
        feed_dict = {train_data_node: batch_data,
                    train_labels_node: batch_labels}
        # Run the graph and fetch some of the nodes.
        _, l, lr, predictions = s.run(
        [optimizer, loss, learning_rate, train_prediction],
        feed_dict=feed_dict)
        
        # Print out the loss periodically.
        if step % 100 == 0:
            error, _ = error_rate(predictions, batch_labels)
            print('Step %d of %d' % (step, steps))
            print('Mini-batch loss: %.5f Error: %.5f Learning rate: %.5f' % (l, error, lr))
            print('Validation error: %.1f%%' % error_rate(
                validation_prediction.eval(), validation_labels)[0])

test_error, confusions = error_rate(test_prediction.eval(), test_labels)
print('Test error: %.1f%%' % test_error)

plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(False)
plt.xticks(numpy.arange(NUM_LABELS))
plt.yticks(numpy.arange(NUM_LABELS))
plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');

for i, cas in enumerate(confusions):
    for j, count in enumerate(cas):
        if count > 0:
            xoff = .07 * len(str(count))
            plt.text(j-xoff, i+.2, int(count), fontsize=9, color='white')
plt.show()

plt.xticks(numpy.arange(NUM_LABELS))
plt.hist(numpy.argmax(test_labels, 1))
plt.show()

