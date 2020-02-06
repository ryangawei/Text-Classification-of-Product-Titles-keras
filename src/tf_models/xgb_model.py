# coding=utf-8
import tensorflow as tf
from tf.cnn_model import TextCNN
from tf.cnn_model import CNNConfig
from data import preprocess
import os
import numpy as np
import xgboost as xgb


def get_score(model_path, checkpoint_file, meta_file, titles    ):
    '''
    根据给定模型和标题，返回模型softmax层得分[None, 1258]
    '''
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        checkpoint_dir = os.path.abspath(model_path)
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, meta_file))
        saver.restore(sess, checkpoint_file)
        graph = tf.get_default_graph()

        config = CNNConfig()
        model = cnn(config)
        # 读取测试集及词汇表数据
        dataset, next_element = model.prepare_test_data()

        # 从图中读取变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # prediction = graph.get_operation_by_name("output/prediction").outputs[0]
        score = graph.get_operation_by_name("output/score").outputs[0]
        training = graph.get_operation_by_name("training").outputs[0]

        label = preprocess.read_label(os.path.join('data', preprocess.LABEL_ID_PATH))
        batch_x = []
        if model.train_mode == 'CHAR-RANDOM' or model.train_mode == 'WORD-NON-STATIC':
            # 1.id
            for title in titles:
                batch_x.append(preprocess.to_id(title, model.vocab, model.train_mode))
        batch_x = np.stack(batch_x)

        feed_dict = {
            input_x: batch_x,
            dropout_keep_prob: 1.0,
            training: False
        }
        scr = sess.run(score, feed_dict)
        # pre = sess.run(prediction, feed_dict)
        return scr

def write_socres(model_path, checkpoint_file, meta_file):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        config = CNNConfig()
        cnn = TextCNN(config)
        cnn.prepare_data()

        # 读取模型
        checkpoint_dir = os.path.abspath(model_path)
        checkpoint_file = os.path.join(checkpoint_dir, checkpoint_file)
        saver = tf.train.import_meta_graph(os.path.join(checkpoint_dir, meta_file))
        saver.restore(sess, checkpoint_file)
        graph = tf.get_default_graph()

        # 从图中读取变量
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        # prediction = graph.get_operation_by_name("output/prediction").outputs[0]
        score = graph.get_operation_by_name("output/score").outputs[0]
        training = graph.get_operation_by_name("training").outputs[0]

        train_init_op, valid_init_op, next_train_element, next_valid_element = cnn.shuffle_datset()
        sess.run(train_init_op)

        label = preprocess.read_label(os.path.join('data', preprocess.LABEL_ID_PATH))
        while True:
            try:
                lines = sess.run(next_train_element)
                titles, batch_y = cnn.convert_input(lines)
                batch_x = []
                if cnn.train_mode == 'CHAR-RANDOM' or cnn.train_mode == 'WORD-NON-STATIC':
                    for title in titles:
                        batch_x.append(preprocess.to_id(title, cnn.vocab, cnn.train_mode))
                batch_x = np.stack(batch_x)

                feed_dict = {
                    input_x: batch_x,
                    dropout_keep_prob: 1.0,
                    training: False
                }
                scr = sess.run(score, feed_dict)
                print(scr)
                break

            except tf.errors.OutOfRangeError:
                # 初始化验证集迭代器
                sess.run(valid_init_op)
                # 计算验证集准确率
                # valid_step(next_valid_element)
                break

class xgb_model():
    def __init__(self):
        self.params = {
            'max_depth': 2,
            'eta': 1,
            'silent': 1,
            'objective': 'multi:softmax',
            'num_class': 10,
            'nthread': 5,
            'eval_metric': 'mlogloss'
        }
        # GPU support
        # params['gpu_id'] = 0
        # params['max_bin'] = 16
        # params['tree_method'] = 'gpu_hist'

        self.num_round = 500
        self.early_stop = 300

        self.cnn_path = "checkpoints\\textcnn"
        self.cnn_checkpoint = "CHAR-RANDOM-2735"
        self.cnn_meta = "CHAR-RANDOM-2735.meta"

    def fit(self, title, label):
        cnn_score = get_score(self.cnn_path, self.cnn_checkpoint, self.cnn_meta, title)
        stack_scores = np.hstack(cnn_score)
        
        dtrain = xgb.DMatrix(stack_scores, label=label)
        xgb.cv(self.params, dtrain, self.num_round, nfold=5,
            metrics={'error'}, seed=0,
            callbacks=[xgb.callback.print_evaluation(show_stdv=False),
                        xgb.callback.early_stop(self.early_stop)])


if __name__ == '__main__':
    # data = np.empty(shape=[0, 2])
    # with open('./data/train_with_id.csv', newline='') as f:
    #     reader = csv.reader(f)
    #     # for row in reader:
    #     #     data = np.append(data, [row], axis=0)
    #     #     print(data.size)
    #     print(list(reader))

    # bgt = xgb_model()
    # bgt.fit(data[:,0], data[:,1])
    write_socres("checkpoints\\textcnn", "CHAR-RANDOM-54700", "CHAR-RANDOM-54700.meta")