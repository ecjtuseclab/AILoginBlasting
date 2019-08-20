from CaptchaRecognition import train
#import train
import argparse

class CRMain(object):
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-is_training', help='train or test', type=bool, default=False)
    parser.add_argument('-model_check', help='the path of pretrained vgg model', type=str,
                        default='./model/crack_capcha.model-3100')
    parser.add_argument('-train_data_path', help='the path of train data', type=str,
                        default='./train_code')
    parser.add_argument('-valid_data_path', help='the path of train data', type=str,
                        default='./test_code')
    parser.add_argument('-test_data', help='the path of test image', type=str, default='./test_code/1311.jpg')
    args = parser.parse_args()
    '''
    
    MAX_CAPTCHA=4
    Is_alphabet=0
    model_dir = './CaptchaRecognition/model'
    train_image_path = './CaptchaRecognition/train_code'
    valid_image_path = './CaptchaRecognition/test_code'
    is_training = False

    def __init__(self):
        self.model = train.Model(self.MAX_CAPTCHA, self.Is_alphabet, 35, 120, model_path=self.model_dir, train_image_path=self.train_image_path, valid_image_path=self.valid_image_path)
        self.model.predict_builder()

    def getCodeNum(self,imgPath):
        #test_image = './test_code/test.jpg'
        #test_path = './test_code'
        num = 0
        
        # 训练
        if self.is_training:
            #print("training")
            self.model.train_crack_captcha_cnn()
        # 预测
        else:
            #print("predict")
            num = self.model.predict_image_num(imgPath)
        return num
