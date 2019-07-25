#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import tensorflow as tf
import os
import sys


# # 数据处理

# In[35]:


def get_data(file):
    x = []
    y = []
    try:
        f = open(file)
        fc = f.readlines()
        print(len(fc))
        for line in fc:
            try:
                y_t,x_t = line.strip().split('\t')
                x.append(x_t)
                y.append(y_t)
            except :
                print(line)
                print(line.strip().split('\t'))
                pass
        return x,y
    except IOError:
        line = ""


# In[6]:


import json
def load_dict(dict_path):
    char_dict_re = dict()
    #dict_path = os.path.join(DATA_PATH, 'words.dict')
    with open(dict_path, encoding='utf-8') as fin:
        char_dict = json.load(fin)
    for k, v in char_dict.items():
        char_dict_re[v] = k
    return char_dict, char_dict_re


# In[8]:


import jieba
def data_process(text_str):
    if len(text_str) == 0:
        print('[ERROR] data_process failed! | The params: {}'.format(text_str))
        return None
    text_str = text_str.strip().replace('\s+', ' ', 3)
    #jieba.lcut 返回中文分词的list
    return jieba.lcut(text_str)

# 每次处理一段话
def word2id(text_str, word_dict, max_seq_len=128):
    if len(text_str) == 0 or len(word_dict) == 0:
        print('[ERROR] word2id failed! | The params: {} and {}'.format(text_str, word_dict))
        return None

    sent_list = data_process(text_str)
    sent_ids = list()
    for item in sent_list:
        if item in word_dict:
            sent_ids.append(word_dict[item])
        else:
            # unknown key 和pad
            sent_ids.append(word_dict['_UNK_'])

    if len(sent_ids) < max_seq_len:
        sent_ids = sent_ids + [word_dict['_PAD_'] for _ in range(max_seq_len - len(sent_ids))]
    else:
        sent_ids = sent_ids[:max_seq_len]
    return sent_ids


def data2id(x,word_dict):
    x = list(map(lambda t:word2id(t,word_dict),x))
    return x


# In[10]:


from sklearn.model_selection import train_test_split
x,y = get_data('./data/带标签短信.txt')
x= data2id(x,char_dict)
X_train, X_test, y_train, y_test = train_test_split( np.array(x), np.array(y), test_size=0.3, random_state=42)
    


# In[11]:


def get_batch(x,size = 32):
    data_size = len(x)
    num = np.floor(data_size/size)
    index = np.arange(data_size)
    index = np.append(index,np.random.randint(0,num,size - data_size%size))
    np.random.shuffle(index)
    index  = index.reshape(-1,size)
    return index

def get_next_batch(x,y,index,size = 32,step = 0):
    return x[index[step]],y[index[step]]



                      
    


# In[1]:



word_dict, word_dict_res = load_dict('./data/words.dict')
vocab_size = max(word_dict.values()) + 1


# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')


# 超参
embedding_dim = 64      # 嵌入层大小
dnn_dim = 128           # Dense层大小
max_seq_len = 128       # 最大句长
num_filters = 64        # 卷积核数目
kernel_size = 5         # 卷积核尺寸
learning_rate = 1e-3    # 学习率
numclass = 2            # 类别数

tf.reset_default_graph()
# 传值空间
input_x = tf.placeholder(tf.int32, shape=[None, max_seq_len], name='input_x')
input_y = tf.placeholder(tf.int32, shape=[None], name='input_y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')


# define embedding layer
# embedding 参照word embedding
with tf.variable_scope('embedding'):
    # 标准正态分布初始化
    input_embedding = tf.Variable(
        tf.truncated_normal(shape=[vocab_size, embedding_dim], stddev=0.1), name='encoder_embedding')

with tf.name_scope("cnn"):
    # CNN layer
    x_input_embedded = tf.nn.embedding_lookup(input_embedding, input_x)
    conv = tf.layers.conv1d(x_input_embedded,num_filters,5,padding='same',name='conv')
    # global max pooling layer
    pooling = tf.reduce_max(conv, reduction_indices=[1])


with tf.name_scope("score"):
    # 全连接层，后面接dropout以及relu激活
    fc = tf.layers.dense(pooling, dnn_dim, name='fc')
    fc = tf.contrib.layers.dropout(fc, keep_prob)
    fc = tf.nn.relu(fc)
    # 分类器
    logits = tf.layers.dense(fc, numclass, name='fc1')
    y_pred_cls = tf.argmax(tf.nn.softmax(logits), 1, name='y_pred')  # 预测类别

with tf.name_scope("optimize"):
    # 将label进行onehot转化
    one_hot_labels = tf.one_hot(input_y, depth=numclass, dtype=tf.float32)
    # 损失函数，交叉熵
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_labels)
    loss = tf.reduce_mean(cross_entropy)
    # 优化器
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("accuracy"):
    # 准确率
    correct_pred = tf.equal(tf.argmax(one_hot_labels, 1), y_pred_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')

with tf.name_scope("summary"):
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary = tf.summary.merge_all()

best_score = 0
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

    # dataset.get_step() 获取数据的总迭代次数
    total_step = int(np.ceil(len(X_train)/32) )
    valid_step = int(np.ceil(len(X_test)/64) )
    index = get_batch(X_train,size = 32)
    valid_index = get_batch(X_test,size = 64)
    for step in range(total_step):
        x_train, y_train = get_next_batch(X_train,y_train,index,size = 32,step = step)
        x_val, y_val = get_next_batch(X_test,y_test,index,size = 64,step = step%valid_step)

        fetches = [loss, accuracy, train_op]
        feed_dict = {input_x: x_train, input_y: y_train, keep_prob: 0.5}
        loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)

        valid_acc = sess.run(accuracy, feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
        summary = sess.run(merged_summary, feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        cur_step = str(step + 1)
        print('The Current step per total: {} | The Current loss: {} | The Current ACC: {} |'
              ' The Current Valid ACC: {}'.format(cur_step, loss_, accuracy_, valid_acc))
        if step % 100 == 0:  # 每隔100个step存储一次model文件
            model.save_model(sess, MODEL_PATH, overwrite=True)







