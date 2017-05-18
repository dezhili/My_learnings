'''
此教程将展示如何在高难度的语言模型中训练循环神经网络。
该问题的目标是获得一个能确定语句概率的概率模型。
为了做到这一点，通过之前已经给出的词语来预测后面的词语。
我们将使用 PTB(Penn Tree Bank) 数据集，这是一种常用来衡量模型的基准，
同时它比较小而且训练起来相对快速。


LSTM 
模型的核心是由一个LSTM单元组成，其可以在某时刻处理一个词语，以及计算语句的延续性的概率。
网络的存储状态由一个零矢量初始化并在读取每一个词语后更新。而且将batch_size 为最小批量来处理数据

lstm = rnn_cell.BasicLSTMCell(lstm_size)
# 初始化 LSTM 存储状态.
state = tf.zeros([batch_size, lstm.state_size])

loss = 0.0
for current_batch_of_words in words_in_dataset:
    # 每次处理一批词语后更新状态值.
    output, state = lstm(current_batch_of_words, state)

    # LSTM 输出可用于产生下一个词语的预测
    logits = tf.matmul(output, softmax_w) + softmax_b
    probabilities = tf.nn.softmax(logits)
    loss += loss_function(probabilities, target_words)

截断反向传播
为使学习过程易于处理，通常的做法是将反向传播的梯度在（按时间）展开的步骤上照一个固定长度(num_steps)截断。 
通过在一次迭代中的每个时刻上提供长度为 num_steps 的输入和每次迭代完成之后反向传导，
这会很容易实现。

输入
在输入 LSTM 前，词语 ID 被嵌入到了一个密集的表示中(查看 矢量表示教程)。
这种方式允许模型高效地表示词语，也便于写代码：

损失函数

多个LSTM层堆叠

'''