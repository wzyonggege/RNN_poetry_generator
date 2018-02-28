# coding:utf-8

import argparse
import sys
import os
import time
import numpy as np
import collections
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.legacy_seq2seq as seq2seq

BEGIN_CHAR = '^'
END_CHAR = '$'
UNKNOWN_CHAR = '*'
MAX_LENGTH = 100
MIN_LENGTH = 10
max_words = 3000
epochs = 50
poetry_file = 'poetry.txt'
save_dir = 'log'


class Data:
    def __init__(self):
        self.batch_size = 64
        self.poetry_file = poetry_file
        self.load()
        self.create_batches()

    def load(self):
        def handle(line):
            if len(line) > MAX_LENGTH:
                index_end = line.rfind('。', 0, MAX_LENGTH)
                index_end = index_end if index_end > 0 else MAX_LENGTH
                line = line[:index_end + 1]
            return BEGIN_CHAR + line + END_CHAR

        self.poetrys = [line.strip().replace(' ', '').split(':')[1] for line in
                        open(self.poetry_file, encoding='utf-8')]
        self.poetrys = [handle(line) for line in self.poetrys if len(line) > MIN_LENGTH]
        # 所有字
        words = []
        for poetry in self.poetrys:
            words += [word for word in poetry]
        counter = collections.Counter(words)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        words, _ = zip(*count_pairs)

        # 取出现频率最高的词的数量组成字典，不在字典中的字用'*'代替
        words_size = min(max_words, len(words))
        self.words = words[:words_size] + (UNKNOWN_CHAR,)
        self.words_size = len(self.words)

        # 字映射成id
        self.char2id_dict = {w: i for i, w in enumerate(self.words)}
        self.id2char_dict = {i: w for i, w in enumerate(self.words)}
        self.unknow_char = self.char2id_dict.get(UNKNOWN_CHAR)
        self.char2id = lambda char: self.char2id_dict.get(char, self.unknow_char)
        self.id2char = lambda num: self.id2char_dict.get(num)
        self.poetrys = sorted(self.poetrys, key=lambda line: len(line))
        self.poetrys_vector = [list(map(self.char2id, poetry)) for poetry in self.poetrys]

    def create_batches(self):
        self.n_size = len(self.poetrys_vector) // self.batch_size
        self.poetrys_vector = self.poetrys_vector[:self.n_size * self.batch_size]
        self.x_batches = []
        self.y_batches = []
        for i in range(self.n_size):
            batches = self.poetrys_vector[i * self.batch_size: (i + 1) * self.batch_size]
            length = max(map(len, batches))
            for row in range(self.batch_size):
                if len(batches[row]) < length:
                    r = length - len(batches[row])
                    batches[row][len(batches[row]): length] = [self.unknow_char] * r
            xdata = np.array(batches)
            ydata = np.copy(xdata)
            ydata[:, :-1] = xdata[:, 1:]
            self.x_batches.append(xdata)
            self.y_batches.append(ydata)


class Model:
    def __init__(self, data, model='lstm', infer=False):
        self.rnn_size = 128
        self.n_layers = 2

        if infer:
            self.batch_size = 1
        else:
            self.batch_size = data.batch_size

        if model == 'rnn':
            cell_rnn = rnn.BasicRNNCell
        elif model == 'gru':
            cell_rnn = rnn.GRUCell
        elif model == 'lstm':
            cell_rnn = rnn.BasicLSTMCell

        cell = cell_rnn(self.rnn_size, state_is_tuple=False)
        self.cell = rnn.MultiRNNCell([cell] * self.n_layers, state_is_tuple=False)

        self.x_tf = tf.placeholder(tf.int32, [self.batch_size, None])
        self.y_tf = tf.placeholder(tf.int32, [self.batch_size, None])

        self.initial_state = self.cell.zero_state(self.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [self.rnn_size, data.words_size])
            softmax_b = tf.get_variable("softmax_b", [data.words_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                    "embedding", [data.words_size, self.rnn_size])
                inputs = tf.nn.embedding_lookup(embedding, self.x_tf)

        outputs, final_state = tf.nn.dynamic_rnn(
            self.cell, inputs, initial_state=self.initial_state, scope='rnnlm')

        self.output = tf.reshape(outputs, [-1, self.rnn_size])
        self.logits = tf.matmul(self.output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        self.final_state = final_state
        pred = tf.reshape(self.y_tf, [-1])
        # seq2seq
        loss = seq2seq.sequence_loss_by_example([self.logits],
                                                [pred],
                                                [tf.ones_like(pred, dtype=tf.float32)],)

        self.cost = tf.reduce_mean(loss)
        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


def train(data, model):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(save_dir)
        saver.restore(sess, model_file)
        n = 0
        for epoch in range(epochs):
            sess.run(tf.assign(model.learning_rate, 0.002 * (0.97 ** epoch)))
            pointer = 0
            for batche in range(data.n_size):
                n += 1
                feed_dict = {model.x_tf: data.x_batches[pointer], model.y_tf: data.y_batches[pointer]}
                pointer += 1
                train_loss, _, _ = sess.run([model.cost, model.final_state, model.train_op], feed_dict=feed_dict)
                sys.stdout.write('\r')
                info = "{}/{} (epoch {}) | train_loss {:.3f}" \
                    .format(epoch * data.n_size + batche,
                            epochs * data.n_size, epoch, train_loss)
                sys.stdout.write(info)
                sys.stdout.flush()
                # save
                if (epoch * data.n_size + batche) % 1000 == 0 \
                        or (epoch == epochs-1 and batche == data.n_size-1):
                    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=n)
                    sys.stdout.write('\n')
                    print("model saved to {}".format(checkpoint_path))
            sys.stdout.write('\n')


def sample(data, model, head=u''):
    def to_word(weights):
        t = np.cumsum(weights)
        s = np.sum(weights)
        sa = int(np.searchsorted(t, np.random.rand(1) * s))
        return data.id2char(sa)

    for word in head:
        if word not in data.words:
            return u'{} 不在字典中'.format(word)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables())
        model_file = tf.train.latest_checkpoint(save_dir)
        # print(model_file)
        saver.restore(sess, model_file)

        if head:
            print('生成藏头诗 ---> ', head)
            poem = BEGIN_CHAR
            for head_word in head:
                poem += head_word
                x = np.array([list(map(data.char2id, poem))])
                state = sess.run(model.cell.zero_state(1, tf.float32))
                feed_dict = {model.x_tf: x, model.initial_state: state}
                [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
                word = to_word(probs[-1])
                while word != u'，' and word != u'。':
                    poem += word
                    x = np.zeros((1, 1))
                    x[0, 0] = data.char2id(word)
                    [probs, state] = sess.run([model.probs, model.final_state],
                                              {model.x_tf: x, model.initial_state: state})
                    word = to_word(probs[-1])
                poem += word
            return poem[1:]
        else:
            poem = ''
            head = BEGIN_CHAR
            x = np.array([list(map(data.char2id, head))])
            state = sess.run(model.cell.zero_state(1, tf.float32))
            feed_dict = {model.x_tf: x, model.initial_state: state}
            [probs, state] = sess.run([model.probs, model.final_state], feed_dict)
            word = to_word(probs[-1])
            while word != END_CHAR:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = data.char2id(word)
                [probs, state] = sess.run([model.probs, model.final_state],
                                          {model.x_tf: x, model.initial_state: state})
                word = to_word(probs[-1])
            return poem


def main():
    msg = """
            Usage:
            Training: 
                python poetry_gen.py --mode train
            Sampling:
                python poetry_gen.py --mode sample --head 明月别枝惊鹊
            """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='sample',
                        help=u'usage: train or sample, sample is default')
    parser.add_argument('--head', type=str, default='',
                        help='生成藏头诗')

    args = parser.parse_args()

    if args.mode == 'sample':
        infer = True  # True
        data = Data()
        model = Model(data=data, infer=infer)
        print(sample(data, model, head=args.head))
    elif args.mode == 'train':
        infer = False
        data = Data()
        model = Model(data=data, infer=infer)
        print(train(data, model))
    else:
        print(msg)


if __name__ == '__main__':
    main()
