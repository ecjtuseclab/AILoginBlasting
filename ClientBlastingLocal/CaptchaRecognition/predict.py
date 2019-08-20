import numpy as np
import tensorflow as tf
import cv2
import os
import random
import time
# 验证码中的字符
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
 'V', 'W', 'X', 'Y', 'Z']
class Predict_Model(object):
    def __init__(self,model_path,MAX_CAPTCHA,Is_alphabet,IMAGE_HEIGHT,IMAGE_WIDTH):
        self.model_path = model_path
        self.MAX_CAPTCHA=MAX_CAPTCHA
        self.char_set=self.get_char_set(Is_alphabet)
        self.IMAGE_WIDTH=IMAGE_WIDTH
        self.IMAGE_HEIGHT=IMAGE_HEIGHT
        self.CHAR_SET_LEN=len(self.char_set)
        self.X = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None,  self.CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout
        self.output=self.crack_captcha_cnn();
            #根据是否包含英文字母获取字符集.
    def get_char_set(self,Is_alphabet):
        char_set=[]
        if(Is_alphabet==0):
            char_set=number
        else:
            char_set=number+alphabet+ALPHABET
        return char_set
    # 构建网络，定义ｃｎｎ网络
    def crack_captcha_cnn(self,w_alpha=0.01, b_alpha=0.1):
        x = tf.reshape(self.X, shape=[-1, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1])
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
        b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv1 = tf.nn.dropout(conv1, self.keep_prob)
        w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
        b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)
        w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
        b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
        conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)
        w_d = tf.Variable(w_alpha * tf.random_normal([5 * 15 * 64, 1024]))
        b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
        dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
        dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
        dense = tf.nn.dropout(dense, self.keep_prob)
        w_out = tf.Variable(w_alpha * tf.random_normal([1024, self.CHAR_SET_LEN]))
        b_out = tf.Variable(b_alpha * tf.random_normal([self.CHAR_SET_LEN]))
        out = tf.add(tf.matmul(dense, w_out), b_out)
        return out
    def gen_image_for_test(self,image_Path):
        img = cv2.imread(image_Path,0)
        img = cv2.resize(img, (self.IMAGE_WIDTH*self.MAX_CAPTCHA,self.IMAGE_HEIGHT))
        img = np.float32(img)
        return img
    def get_image(self,image_Path):
        batch_x = np.zeros([self.MAX_CAPTCHA, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        def wrap_test_captcha_image(image_Path):
            while True:
                image = self.gen_image_for_test(image_Path)
                if image.shape == (self.IMAGE_HEIGHT, self.IMAGE_WIDTH*self.MAX_CAPTCHA):
                    return image
        for listNum in os.walk(image_Path):
            pass
        image = wrap_test_captcha_image(image_Path)
        cut_image=np.hsplit(image, self.MAX_CAPTCHA)
        for j in range(self.MAX_CAPTCHA):
             single_image=cut_image[j]
             batch_x[j, :] = single_image.flatten() / 255  # (image.flatten()-128)/128  meanÎª0
        return batch_x
    def predict_image(self,predict_image_path):
        captcha_image=self.get_image(predict_image_path)
        saver = tf.train.Saver()
        predict = tf.reshape(self.output, [-1, 1, self.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        result=""
        with tf.Session() as sess:
            for i in range(self.MAX_CAPTCHA):
                saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                batch_x_predict=captcha_image[i]
                text_list = sess.run(max_idx_p, feed_dict={self.X: [batch_x_predict], self.keep_prob: 1})
                text = text_list[0].tolist()
                vector = np.zeros(self.CHAR_SET_LEN)
                j = 0
                for n in text:
                    vector[j * self.CHAR_SET_LEN + n] = 1
                    j += 1
                predict_text= self.vec2text(vector)
                result=result+predict_text
        return result

    def predict_folder(self,predict_image_folder_path):
        predict_image_list=[]
        true_list=[]
        result_list=[]
        for filePath in os.listdir(predict_image_folder_path):
            true_list.append(filePath.split(".")[0])
            predict_image_list.append(os.path.join(predict_image_folder_path,filePath))
        saver = tf.train.Saver()
        predict = tf.reshape(self.output, [-1, 1, self.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        with tf.Session() as sess:
            for image_path in predict_image_list:
                result=""
                captcha_image=self.get_image(image_path)
                for i in range(self.MAX_CAPTCHA):
                    saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
                    batch_x_predict=captcha_image[i]
                    text_list = sess.run(max_idx_p, feed_dict={self.X: [batch_x_predict], self.keep_prob: 1})
                    text = text_list[0].tolist()
                    vector = np.zeros(self.CHAR_SET_LEN)
                    j = 0
                    for n in text:
                        vector[j * self.CHAR_SET_LEN + n] = 1
                        j += 1
                    predict_text= self.vec2text(vector)
                    result=result+predict_text
                result_list.append(result)
        return true_list,result_list
    def vec2text(self,vec):
        char_pos = vec.nonzero()[0]
        text = []
        for i, c in enumerate(char_pos):
            char_at_pos = i  # c/63
            char_idx = c % self.CHAR_SET_LEN
            if char_idx < 10:
                char_code = char_idx + ord('0')
            elif char_idx < 36:
                char_code = char_idx - 10 + ord('A')
            elif char_idx < 62:
                char_code = char_idx - 36 + ord('a')
            elif char_idx == 62:
                char_code = ord('_')
            else:
                raise ValueError('error')
            text.append(chr(char_code))
        return "".join(text)

