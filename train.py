# coding: utf-8
import os

import paddle

from model_dygraph import *
from data_iterator import *
import numpy as np
import paddtf as tf
import sys
import random
from datetime import timedelta, datetime

num_epochs = 100
batch_size = 4  # 256
window_size = 50
starter_learning_rate = 0.001
learning_rate_decay = 1.0

today = datetime.today() + timedelta(0)
today_format = today.strftime('%Y%m%d')
ckpt_dir = 'ckpt/dmr_' + today_format
os.makedirs(ckpt_dir,exist_ok=True)

in_features = 111
def train():
    train_data = DataIterator('alimama_sampled.txt', batch_size, 20)


    lr_scheduler=paddle.optimizer.lr.ExponentialDecay(starter_learning_rate,gamma=learning_rate_decay,last_epoch=2000000)

    # construct the model structure
    model = Model_DMR(in_features)
    adam_optim = paddle.optimizer.Adam(learning_rate=lr_scheduler,parameters=model.parameters())

    iter = 0
    test_iter = 1
    loss_sum = 0.0
    accuracy_sum = 0.
    aux_loss_sum = 0.
    stored_arr = []


    for features, targets in train_data:
        loss, acc, aux_loss, prob = model.train_batch(None, features, targets)
        loss.backward()
        adam_optim.minimize(loss)
        loss_sum += loss.numpy()
        accuracy_sum += acc.numpy()
        aux_loss_sum += aux_loss.numpy()
        prob_1 = prob.numpy()[:, 0].tolist()
        target_1 = targets.numpy().tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        iter += 1
        if (iter % test_iter) == 0:
            print(datetime.now().ctime())
            print(
                'iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f ---- train_auc: %.4f' % \
                (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter,
                 calc_auc(stored_arr)))
            loss_sum = 0.0
            accuracy_sum = 0.0
            aux_loss_sum = 0.0
            stored_arr = []
            # paddle.save(model.state_dict(),ckpt_dir+"/iter_"+str(iter)) ##see if model size very big
            paddle.save(model.state_dict(), ckpt_dir + "/current" )

    print("session finished.")


def eval():
    test_data = DataIterator('alimama_test.txt', batch_size, 20)
    model = Model_DMR(in_features)
    iter = 0
    test_iter = 100
    loss_sum = 0.0
    accuracy_sum = 0.
    aux_loss_sum = 0.
    stored_arr = []


    model.load_dict(paddle.load(ckpt_dir + "/current"))
    print("model loaded from:",ckpt_dir + "/current")

    for features, targets in test_data:
        loss, acc, aux_loss, prob = model.calculate(None, features, targets)
        loss_sum += loss.numpy()
        accuracy_sum += acc.numpy()
        aux_loss_sum += aux_loss.numpy()
        prob_1 = prob[:, 0].numpy().tolist()
        target_1 = targets.numpy().tolist()
        for p, t in zip(prob_1, target_1):
            stored_arr.append([p, t])
        iter += 1
        if (iter % test_iter) == 0:
            print(datetime.now().ctime())

            print(
                'iter: %d ----> test_loss: %.4f ---- test_accuracy: %.4f ---- test_aux_loss: %.4f ---- test_auc: %.4f' % \
                (iter, loss_sum / iter, accuracy_sum / iter, aux_loss_sum / iter, calc_auc(stored_arr)))
    print("session finished.")


if __name__ == "__main__":
    SEED = 3
    #
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    print(sys.argv)
    if sys.argv[1] == 'train':
        train()
    elif sys.argv[1] == 'test':
        eval()
