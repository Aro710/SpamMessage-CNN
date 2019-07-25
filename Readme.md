# 垃圾短信分类

## 数据集

数据集包含80万条中文短信，内容涉及广告、推销、生活等

来源：[https://github.com/hrwhisper/SpamMessage](https://github.com/hrwhisper/SpamMessage)

## 数据预处理

> 把中文句子切分成词
将词转换成字典中对应的数值
pad成固定长度

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

## 生成批

    def get_batch(x,size = 32):
    		data_size = len(x)
    		num = np.floor(data_size/size)
    		index = np.arange(data_size)
    		index = np.append(index,np.random.randint(0,num,size - data_size%size))
    		np.random.shuffle(index)
    		index = index.reshape(-1,size)
    return index
    
    def get_next_batch(x,y,index,size = 32,step = 0):
    		return x[index[step]],y[index[step]]

## 加载字典

    import json
    def load_dict(dict_path):
        char_dict_re = dict()
        #dict_path = os.path.join(DATA_PATH, 'words.dict')
        with open(dict_path, encoding='utf-8') as fin:
            char_dict = json.load(fin)
        for k, v in char_dict.items():
            char_dict_re[v] = k
        return char_dict, char_dict_re

## 训练

    word_dict, word_dict_res = char_dict, char_dict_re
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
        saver = tf.train.Saver()
        # dataset.get_step() 获取数据的总迭代次数
        total_step = int(np.ceil(len(X_train)/32) )
        valid_step = int(np.ceil(len(X_test)/64) )
        index = get_batch(X_train,size = 32)
        valid_index = get_batch(X_test,size = 64)
        for step in range(total_step):
            x_train, y_train = get_next_batch(X_train,Y_train,index,size = 32,step = step)
            x_val, y_val = get_next_batch(X_test,Y_test,valid_index,size = 64,step = step%valid_step)
            fetches = [loss, accuracy, train_op]
            feed_dict = {input_x: x_train, input_y: y_train, keep_prob: 0.5}
            loss_, accuracy_, _ = sess.run(fetches, feed_dict=feed_dict)
            valid_acc = sess.run(accuracy, feed_dict={input_x: x_val, input_y: y_val, keep_prob: 1.0})
            summary = sess.run(merged_summary, feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            cur_step = str(step + 1)+'/'+str(total_step)
    
            if step % 100 == 0:  # 每隔100个step存储一次model文件
                print('The Current step per total: {} | The Current loss: {} | The Current ACC: {} |'
                  ' The Current Valid ACC: {}'.format(cur_step, loss_, accuracy_, valid_acc))
                saver.save(sess, MODEL_PATH+str(step)+".ckpt")

## 训练结果

很快正确率到1

> The Current step per total: 1/17500 | The Current loss: 0.6916527152061462 | The Current ACC: 0.625 | The Current Valid ACC: 0.828125
The Current step per total: 101/17500 | The Current loss: 0.0650424212217331 | The Current ACC: 1.0 | The Current Valid ACC: 0.984375
The Current step per total: 201/17500 | The Current loss: 0.003119946923106909 | The Current ACC: 1.0 | The Current Valid ACC: 0.984375
The Current step per total: 301/17500 | The Current loss: 0.03028859570622444 | The Current ACC: 0.96875 | The Current Valid ACC: 0.984375
The Current step per total: 401/17500 | The Current loss: 0.0019716531969606876 | The Current ACC: 1.0 | The Current Valid ACC: 0.984375
The Current step per total: 501/17500 | The Current loss: 0.009549517184495926 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 601/17500 | The Current loss: 0.27358531951904297 | The Current ACC: 0.96875 | The Current Valid ACC: 1.0
The Current step per total: 701/17500 | The Current loss: 0.0004539302608463913 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 801/17500 | The Current loss: 0.0010753478854894638 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 901/17500 | The Current loss: 0.004611284472048283 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 1001/17500 | The Current loss: 0.09567641466856003 | The Current ACC: 0.96875 | The Current Valid ACC: 1.0
The Current step per total: 1101/17500 | The Current loss: 0.0032468487042933702 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 1201/17500 | The Current loss: 0.009668770246207714 | The Current ACC: 1.0 | The Current Valid ACC: 0.984375
The Current step per total: 1301/17500 | The Current loss: 0.01671394519507885 | The Current ACC: 1.0 | The Current Valid ACC: 0.984375
The Current step per total: 1401/17500 | The Current loss: 0.0019437115406617522 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 1501/17500 | The Current loss: 0.002643697429448366 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 1601/17500 | The Current loss: 0.004031606484204531 | The Current ACC: 1.0 | The Current Valid ACC: 1.0
The Current step per total: 1701/17500 | The Current loss: 0.0023941840045154095 | The Current ACC: 1.0 | The Current Valid ACC: 1.0