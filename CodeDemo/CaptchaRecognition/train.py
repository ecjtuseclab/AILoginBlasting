import numpy as np
import sys
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

class Model(object):
    def __init__(self,MAX_CAPTCHA,Is_alphabet,IMAGE_HEIGHT,IMAGE_WIDTH,model_path, train_image_path=None,valid_image_path=None):
        self.model_path = model_path
        self.train_path = train_image_path
        self.valid_path = valid_image_path
        self.MAX_CAPTCHA=MAX_CAPTCHA
        self.char_set=self.get_char_set(Is_alphabet)
        self.IMAGE_WIDTH=IMAGE_WIDTH
        self.IMAGE_HEIGHT=IMAGE_HEIGHT
        self.CHAR_SET_LEN=len(self.char_set)
        self.X = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        self.Y = tf.placeholder(tf.float32, [None,  self.CHAR_SET_LEN])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout
        self.image_filename_list, self.total = self.get_image_file_name(self.train_path)
        self.image_filename_list_valid, self.valid_total = self.get_image_file_name(self.valid_path)
        self.output = self.crack_captcha_cnn()
    #根据是否包含英文字母获取字符集.
    def get_char_set(self,Is_alphabet):
        char_set=[]
        if(Is_alphabet==0):
            char_set=number
        else:
            char_set=number+alphabet+ALPHABET
        return char_set
    def get_image_file_name(self,imgFilePath):
        fileName = []
        total = 0
        for filePath in os.listdir(imgFilePath):
            captcha_name = filePath.split('/')[-1]
            fileName.append(captcha_name)
            total += 1
        random.seed(time.time())
        random.shuffle(fileName)
        return fileName, total
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
        
    def predict_builder(self):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(self.model_path))
        predict = tf.reshape(self.output, [-1, 1, self.CHAR_SET_LEN])
        self.max_idx_p = tf.argmax(predict, 2)
  
        pass
        
    def predict_image_num(self, predict_image_path):
        #print("predict_image_path"+predict_image_path)
        result=""
        if not os.path.isfile(predict_image_path):
            print('file %s is not exist' % predict_image_path)
            return
        else:
            result = self.get_result(self.sess, predict_image_path, self.max_idx_p)
            return result
        return
                    
    def predict_image(self, predict_image_path):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            predict = tf.reshape(self.output, [-1, 1, self.CHAR_SET_LEN])
            max_idx_p = tf.argmax(predict, 2)

            if not os.path.isdir(predict_image_path):
                print("test_path is not dir")
                return
            while True:
                image_name = input("input image name:")
                image_path = os.path.join(predict_image_path, image_name)
                image_path = image_path.replace('\\', '/')
                print("file path: %s" % image_path)
                if not os.path.isfile(image_path):
                    print('file %s is not exist' % image_path)
                else:
                    result = self.get_result(sess, image_path, max_idx_p)
                    print(result)

    def get_result(self, sess, predict_image_path, max_idx_p):
        captcha_image = self.get_image(predict_image_path)
        result = ""
        for i in range(self.MAX_CAPTCHA):
            batch_x_predict = captcha_image[i]
            text_list = sess.run(max_idx_p, feed_dict={self.X: [batch_x_predict], self.keep_prob: 1})
            text = text_list[0].tolist()
            vector = np.zeros(self.CHAR_SET_LEN)
            j = 0
            for n in text:
                vector[j * self.CHAR_SET_LEN + n] = 1
                j += 1
            predict_text = self.vec2text(vector)
            result = result + predict_text
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

    def train_crack_captcha_cnn(self):

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels=self.Y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        # predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
        predict = tf.reshape(self.output, [-1, 1, self.CHAR_SET_LEN])
        max_idx_p = tf.argmax(predict, 2)
        # max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
        max_idx_l = tf.argmax(tf.reshape(self.Y, [-1, 1, self.CHAR_SET_LEN]), 2)
        correct_pred = tf.equal(max_idx_p, max_idx_l)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            step = 0
            while True:
                batch_x, batch_y = self.get_next_batch(self.train_path, self.image_filename_list, 5)
                _, loss_ = sess.run([optimizer, loss], feed_dict={self.X: batch_x, self.Y: batch_y, self.keep_prob: 0.75})
                step += 1
                print(step, loss_)
                if step % 100 == 0:
                    batch_x_test, batch_y_test = self.get_next_batch(self.valid_path, self.image_filename_list_valid,50)
                    acc = sess.run(accuracy, feed_dict={self.X: batch_x_test, self.Y: batch_y_test, self.keep_prob: 1.})
                    print(step,"准确率", acc)
                    if step>3000:
                    # if acc ==1 or step>20000:
                        '''''
                        right_num=0
                        print(batch_x_test[0].shape)
                        for i in range(200):
                        # for i in range(1):
                            batch_x_predict=batch_x_test[i]
                            batch_y_predict=batch_y_test[i]
                            text_list = sess.run(max_idx_p, feed_dict={X: [batch_x_predict], self.keep_prob: 1})
                            # print(vec2text(batch_y_predict))
                            text = text_list[0].tolist()
                            vector = np.zeros(self.CHAR_SET_LEN)
                            i = 0
                            for n in text:
                                vector[i * self.CHAR_SET_LEN + n] = 1
                                i += 1
                            # print(vec2text(vector))
                            if(vec2text(vector)==vec2text(batch_y_predict)):
                                right_num=right_num+1
                            else:
                                print(vec2text(vector))
                                print(vec2text(batch_y_predict))
                        print("测试准确率", right_num/200)
                        '''
                        saver.save(sess, os.path.join(self.model_path,"crack_capcha.model"), global_step=step)
                        break
    def get_next_batch(self,imageFilePath, image_filename_list, batch_size=128):
        batch_x = np.zeros([batch_size*self.MAX_CAPTCHA, self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        batch_y = np.zeros([batch_size*self.MAX_CAPTCHA,   self.CHAR_SET_LEN])
        for listNum in os.walk(imageFilePath):
            pass
        imageAmount = len(listNum[2])
        for i in range(batch_size):
            text, image = self.wrap_gen_captcha_text_and_image(imageFilePath,image_filename_list, imageAmount)
            cut_image=np.hsplit(image, self.MAX_CAPTCHA)
            for j in range(self.MAX_CAPTCHA):
                single_image=cut_image[j]
                batch_x[i*4+j, :] = single_image.flatten() / 255  # (image.flatten()-128)/128  meanÎª0
                cut_text=text[j:j+1]
                batch_y[i*4+j, :] = self.text2vec(cut_text)
        return batch_x, batch_y
    def wrap_gen_captcha_text_and_image(self,imageFilePath,image_filename_list, imageAmount):
        while True:
            text, image = self.gen_captcha_text_and_image(imageFilePath, image_filename_list, imageAmount)
            if image.shape == (self.IMAGE_HEIGHT, self.IMAGE_WIDTH*self.MAX_CAPTCHA):
                return text, image
    def gen_captcha_text_and_image(self,imageFilePath, image_filename_list, imageAmount):
        num = random.randint(0, imageAmount - 1)
        img = cv2.imread(os.path.join(imageFilePath, image_filename_list[num]), 0)
        img = cv2.resize(img, (self.IMAGE_WIDTH*self.MAX_CAPTCHA,self.IMAGE_HEIGHT ))
        img = np.float32(img)
        text = image_filename_list[num].split('.')[0]
        return text, img
    def text2vec(self,text):
        text_len = len(text)
        if text_len > self.MAX_CAPTCHA:
            raise ValueError('验证码最长为四个字符')
        vector = np.zeros(self.CHAR_SET_LEN)
        def char2pos(c):
            if c == '_':
                k = 62
                return k
            k = ord(c) - 48
            if k > 9:
                k = ord(c) - 55
                if k > 35:
                    k = ord(c) - 61
                    if k > 61:
                        raise ValueError('No Map')
            return k
        for i, c in enumerate(text):
            idx = i * self.CHAR_SET_LEN + char2pos(c)
            vector[idx] = 1
        return vector
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


