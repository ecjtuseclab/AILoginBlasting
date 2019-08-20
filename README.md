# AILoginBlasting
基于机器学习的登录验证码爆破工具

开发人员：康文洋、魏超、张梦丽、陈兰兰、艾美珍、夏萍萍、易传佳、张泓、王松

    基于机器学习的登录验证码爆破工具提供了两种工作模式：本地运行模式和远程服务运行模式。
    不管是本地运行模式还是远程服务运行模式都需要安装服务端和客户端。
    服务端为机器学习模块，该模块主要在TensorFlow深度学习框架基础上构建CNN卷积神经网络算法，
    对目标验证码进行训练与识别。
    客户端为验证码爆破管理模块。
    对于远程服务运行模式，还需要运行一个基于Django的Web程序，提供对外访问的API接口。

    机器学习模块
    |--CaptchaRecognition：核心模块
    |----model：保存的模型
    |----test_code：测试验证码
    |----train_code：训练验证码
    |----CRMain.py：核心模块主程序类
    |----predict.py：核心模块预测程序
    |----test.py：核心模块测试代码
    |----train.py：核心模块训练程序

    验证码爆破管理模块
    |--ClientBlastingLocal（本地）/ClientBlastingRemote（服务器端）
    |----img：验证码保存文件夹
    |----Config.py：配置文件
    |----Main.py：主程序
    |----password.txt：密码字典
    |----username.txt：用户字典


    运行环境：
    python3.5以上
    python安装：pip install requests,bs4,numpy,tensorflow,opencv-python
    python安装：pip isnatll django   #(远程服务模式需要)


    各程序介绍：
    ClientBlastingLocal：本地的客户端程序，里面加入CaptchaRecognition机器学习模块。
        运行：python Main.py
    ClientBlastingRemote：远程服务的客户端程序
        运行：python Main.py
    CodeDemo：远程服务的服务端程序，基于Django的web程序，里面加入CaptchaRecognition机器学习模块。
        运行：python manage.py runserver 0.0.0.0:8000

演示Demo目标界面如下：
![image](https://raw.githubusercontent.com/ecjtuseclab/AILoginBlasting/master/demologin.png)

演示Demo结果如下：
![image](https://raw.githubusercontent.com/ecjtuseclab/AILoginBlasting/master/success.png)

# 声明：弱口令检查工具，请勿用于非法用途

