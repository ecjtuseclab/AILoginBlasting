# AILoginBlasting
基于机器学习的登录验证码爆破工具

说在前面<br>
喜欢我们的项目请Fork、Star、Watching支持我们。<br>
如果有宝贵的意见，请在Issue中回复，你们宝贵的意见是我们完善的动力。<br>

开发人员：[康文洋](https://github.com/m0w3n?tab=repositories)、魏超、张梦丽、陈兰兰、艾美珍、夏萍萍、易传佳、张泓、王松

基于机器学习的登录验证码爆破工具提供了两种工作模式：本地运行模式和远程服务运行模式。<br>
不管是本地运行模式还是远程服务运行模式都需要安装服务端和客户端。<br>
服务端为机器学习模块，该模块主要在TensorFlow深度学习框架基础上构建CNN卷积神经网络算法，对目标验证码进行训练与识别。<br>
客户端为验证码爆破管理模块。<br>
对于远程服务运行模式，还需要运行一个基于Django的Web程序，提供对外访问的API接口。<br>

* 机器学习模块<br>
    |--CaptchaRecognition：核心模块<br>
    |----model：保存的模型<br>
    |----test_code：测试验证码<br>
    |----train_code：训练验证码<br>
    |----CRMain.py：核心模块主程序类<br>
    |----predict.py：核心模块预测程序<br>
    |----test.py：核心模块测试代码<br>
    |----train.py：核心模块训练程序<br>

* 验证码爆破管理模块<br>
    |--ClientBlastingLocal（本地）/ClientBlastingRemote（服务器端）<br>
    |----img：验证码保存文件夹<br>
    |----Config.py：配置文件<br>
    |----Main.py：主程序<br>
    |----password.txt：密码字典<br>
    |----username.txt：用户字典<br>

* 运行环境：
    python3.5以上<br>
    python安装：pip install requests,bs4,numpy,tensorflow,opencv-python<br>
    python安装：pip isnatll django   #(远程服务模式需要)<br>


* 各程序介绍：
    ClientBlastingLocal：本地的客户端程序，里面加入CaptchaRecognition机器学习模块。<br>
        运行：python Main.py<br>
    ClientBlastingRemote：远程服务的客户端程序<br>
        运行：python Main.py<br>
    CodeDemo：远程服务的服务端程序，基于Django的web程序，里面加入CaptchaRecognition机器学习模块。<br>
        运行：python manage.py runserver 0.0.0.0:8000<br>

演示Demo目标界面如下：<br>
![image](https://raw.githubusercontent.com/ecjtuseclab/AILoginBlasting/master/demologin.png)

演示Demo结果如下：<br>
结果采用类似Burp的爆破思路，通过获取返回数据的长度判断是否爆破成功（假设成功的数据长度明显区别于失败的数据长度）。<br>
![image](https://raw.githubusercontent.com/ecjtuseclab/AILoginBlasting/master/success.png)

# 声明：弱口令检查工具，请勿用于非法用途

