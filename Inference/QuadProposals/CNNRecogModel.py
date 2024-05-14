import  numpy as np
import cv2
import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt
import time
import glob

class CNNRecogModel:
    def __init__(self, type = 'BN'):
        self.imgSet = None
        tf.reset_default_graph()
        rseed2 = 123
        np.random.seed(rseed2)
        tf.set_random_seed(rseed2)
        self.imgs_ph = tf.placeholder(tf.uint8, [None, 104, 104, 1])
        self.labels1_ph = tf.placeholder(tf.int32, [None])
        self.labels2_ph = tf.placeholder(tf.int32, [None])
        self.learnrate_ph = tf.placeholder(tf.float32, None)  # probably can remove the None
        self.training_ph = tf.placeholder(tf.bool)
        self.pkeep_ph = tf.placeholder(tf.float32)
        self.type = type
        self.softMaxCutval = 0.8

        if (type == 'BN'):
            self.defineCNN_BN()
        else:
            self.defineCNN()
        self.sess = tf.Session()
        self.dict = np.array(['1', '2', '3', '4', '5', '6', '7', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'J', 'K', 'L',
                               'M', 'P', 'Q', 'R', 'T', 'U', 'V', 'Y'])

    def defineCNN_BN(self):
        num_classes = 26

        num_filters1 = 12
        num_filters2 = 25
        num_filters3 = 50
        W1 = tf.get_variable("W1", shape=[5, 5, 1, num_filters1], initializer=tf.initializers.glorot_normal())
        b1 = tf.get_variable("b1", shape=[num_filters1], initializer=tf.initializers.zeros())
        W2 = tf.get_variable("W2", shape=[5, 5, num_filters1, num_filters2],
                             initializer=tf.initializers.glorot_normal())
        b2 = tf.get_variable("b2", shape=[num_filters2], initializer=tf.initializers.zeros())
        W3 = tf.get_variable("W3", shape=[5, 5, num_filters2, num_filters3],
                             initializer=tf.initializers.glorot_normal())
        b3 = tf.get_variable("b3", shape=[num_filters3], initializer=tf.initializers.zeros())

        num_fc1 = 4050
        num_fc2 = 120
        num_fc3 = 80
        num_fc4 = 50
        # ... 2) dense layers: fork 1 (first character prediction)
        W4a = tf.get_variable("W4a", shape=[num_fc1, num_fc2], initializer=tf.initializers.glorot_normal())
        b4a = tf.get_variable("b4a", shape=[num_fc2], initializer=tf.initializers.zeros())
        W5a = tf.get_variable("W5a", shape=[num_fc2, num_fc3], initializer=tf.initializers.glorot_normal())
        b5a = tf.get_variable("b5a", shape=[num_fc3], initializer=tf.initializers.zeros())
        W6a = tf.get_variable("W6a", shape=[num_fc3, num_fc4], initializer=tf.initializers.glorot_normal())
        b6a = tf.get_variable("b6a", shape=[num_fc4], initializer=tf.initializers.zeros())
        W7a = tf.get_variable("W7a", shape=[num_fc4, num_classes], initializer=tf.initializers.glorot_normal())
        b7a = tf.get_variable("b7a", shape=[num_classes], initializer=tf.initializers.zeros())
        # ... 3) dense layers: fork 2 (second character prediction)
        W4b = tf.get_variable("W4b", shape=[num_fc1, num_fc2], initializer=tf.initializers.glorot_normal())
        b4b = tf.get_variable("b4b", shape=[num_fc2], initializer=tf.initializers.zeros())
        W5b = tf.get_variable("W5b", shape=[num_fc2, num_fc3], initializer=tf.initializers.glorot_normal())
        b5b = tf.get_variable("b5b", shape=[num_fc3], initializer=tf.initializers.zeros())
        W6b = tf.get_variable("W6b", shape=[num_fc3, num_fc4], initializer=tf.initializers.glorot_normal())
        b6b = tf.get_variable("b6b", shape=[num_fc4], initializer=tf.initializers.zeros())
        # note the num_classes - 1 below:
        W7b = tf.get_variable("W7b", shape=[num_fc4, num_classes - 1], initializer=tf.initializers.glorot_normal())
        b7b = tf.get_variable("b7b", shape=[num_classes - 1], initializer=tf.initializers.zeros())

        # build compute graph evaluating the CNN
        conv1 = tf.nn.relu(
            tf.layers.batch_normalization(tf.nn.conv2d(self.imgs_ph/255, W1, strides=[1, 1, 1, 1], padding='VALID'),
                                          axis=[1, 2, 3], training=self.training_ph))
        # conv1 = tf.nn.relu(tf.nn.conv2d(imgs_ph, W1, strides = [1,1,1,1], padding = 'VALID'))
        pool1 = tf.nn.dropout(tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
                              self.pkeep_ph)
        conv2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(pool1, W2, strides=[1, 1, 1, 1], padding='VALID'),
                                                         axis=[1, 2, 3], training=self.training_ph))
        # conv2 = tf.nn.relu(tf.nn.conv2d(pool1, W2, strides = [1,1,1,1], padding = 'VALID'))
        pool2 = tf.nn.dropout(tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
                              self.pkeep_ph)
        conv3 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.conv2d(pool2, W3, strides=[1, 1, 1, 1], padding='VALID'),
                                                         axis=[1, 2, 3], training=self.training_ph))
        # conv3 = tf.nn.relu(tf.nn.conv2d(pool2, W3, strides = [1,1,1,1], padding = 'VALID'))
        pool3 = tf.nn.dropout(tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'),
                              self.pkeep_ph)
        flat1 = tf.reshape(pool3, [-1, num_fc1])
        bnorm_axis = -1
        dens1a = tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(
            tf.matmul(flat1, W4a), axis=bnorm_axis, training=self.training_ph)), self.pkeep_ph)
        dens2a = tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(
            tf.matmul(dens1a, W5a), axis=bnorm_axis, training=self.training_ph)), self.pkeep_ph)
        dens3a = tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(
            tf.matmul(dens2a, W6a), axis=bnorm_axis, training=self.training_ph)), 1.0)
        dens4a = tf.layers.batch_normalization(tf.matmul(dens3a, W7a), axis=bnorm_axis, training=self.training_ph)

        dens1b = tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(
            tf.matmul(flat1, W4b), axis=bnorm_axis, training=self.training_ph)), self.pkeep_ph)
        dens2b = tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(
            tf.matmul(dens1b, W5b), axis=bnorm_axis, training=self.training_ph)), self.pkeep_ph)
        dens3b = tf.nn.dropout(tf.nn.relu(tf.layers.batch_normalization(
            tf.matmul(dens2b, W6b), axis=bnorm_axis, training=self.training_ph)), 1.0)
        dens4b_partial = tf.layers.batch_normalization(tf.matmul(dens3b, W7b), axis=bnorm_axis, training=self.training_ph)
        # hack: copy the first column of dens4a into dens4b
        d4a_s1, d4a_s2 = tf.split(dens4a, [1, 25], 1)
        dens4b = tf.concat([d4a_s1, dens4b_partial], axis=1)

        self.dens4a = dens4a
        self.dens4b = dens4b

        self.softMaxA = tf.nn.softmax(self.dens4a)
        self.softMaxB = tf.nn.softmax(self.dens4b)

    def loadWeights_BN(self, weightsFile='./CNN_2char_bn_unified.ckpt'):
        saver = tf.train.Saver()
        saver.restore(self.sess, weightsFile)

    def setInputImgSet(self, imgSet):
        self.imgSet = imgSet.reshape(imgSet.shape[0], imgSet.shape[1], imgSet.shape[2], 1)
        self.imgSet = self.imgSet

    def predict(self):
        if self.type == 'BN':
            t_dens4a = self.sess.run(self.dens4a, {self.imgs_ph: self.imgSet, self.training_ph:False, self.pkeep_ph: 1.0})
            t_dens4b = self.sess.run(self.dens4b, {self.imgs_ph: self.imgSet, self.training_ph:False, self.pkeep_ph: 1.0})
        else:
            t_dens4a = self.sess.run(self.dens4a, {self.imgs_ph: self.imgSet, self.pkeep_ph: 1.0})
            t_dens4b = self.sess.run(self.dens4b, {self.imgs_ph: self.imgSet, self.pkeep_ph: 1.0})

        t_predictions1 = np.argmax(t_dens4a, 1)
        t_predictions2 = np.argmax(t_dens4b, 1)

        outputs = []
        for i in range(self.imgSet.shape[0]):
            if t_predictions1[i] and t_predictions2[i]:
                outputs.append(self.dict[t_predictions1[i]-1]+self.dict[t_predictions2[i]-1])
            else:
                outputs.append('\0\0')
        return outputs

    def predictSoftMaxCut(self):
        if self.type == 'BN':
            t_softMaxA = self.sess.run(self.softMaxA,
                                       {self.imgs_ph: self.imgSet, self.training_ph: False, self.pkeep_ph: 1.0})
            t_softMaxB = self.sess.run(self.softMaxB,
                                       {self.imgs_ph: self.imgSet, self.training_ph: False, self.pkeep_ph: 1.0})
        else:
            t_softMaxA = self.sess.run(self.softMaxA, {self.imgs_ph: self.imgSet, self.pkeep_ph: 1.0})
            t_softMaxB = self.sess.run(self.softMaxB, {self.imgs_ph: self.imgSet, self.pkeep_ph: 1.0})

        t_predictions1 = np.argmax(t_softMaxA, 1)
        t_predictions2 = np.argmax(t_softMaxB, 1)

        outputs = []
        for i in range(self.imgSet.shape[0]):
            peakA = np.max(t_softMaxA[i, :])
            peakB = np.max(t_softMaxB[i, :])
            if peakA < self.softMaxCutval or peakB < self.softMaxCutval:
                t_predictions1[i] = 0
                t_predictions2[i] = 0
            if t_predictions1[i] and t_predictions2[i]:
                outputs.append(self.dict[t_predictions1[i]-1]+self.dict[t_predictions2[i]-1])
            else:
                outputs.append('\0\0')
        return outputs

    def softMaxToCode(self, softMaxA, softMaxB):
        outputs = []
        t_predictions1 = np.argmax(softMaxA, 1)
        t_predictions2 = np.argmax(softMaxB, 1)
        for i in range(softMaxA.shape[0]):
            if t_predictions1[i] and t_predictions2[i]:
                outputs.append(self.dict[t_predictions1[i]-1]+self.dict[t_predictions2[i]-1])
            else:
                outputs.append('\0\0')
        return outputs


    def predictSoftMaxOutput(self):
        if self.type == 'BN':
            t_softMaxA = self.sess.run(self.softMaxA, {self.imgs_ph: self.imgSet, self.training_ph:False, self.pkeep_ph: 1.0})
            t_softMaxB = self.sess.run(self.softMaxB, {self.imgs_ph: self.imgSet, self.training_ph:False, self.pkeep_ph: 1.0})
            return t_softMaxA, t_softMaxB
        else:
            t_softMaxA = self.sess.run(self.softMaxA, {self.imgs_ph: self.imgSet, self.pkeep_ph: 1.0})
            t_softMaxB = self.sess.run(self.softMaxB, {self.imgs_ph: self.imgSet, self.pkeep_ph: 1.0})
            return t_softMaxA, t_softMaxB



