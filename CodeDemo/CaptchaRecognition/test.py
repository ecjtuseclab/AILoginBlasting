from CRMain import CRMain
import os,re

def getCurrentFilePath():
    #获得当前工作目录
    path = os.getcwd()
    path = re.sub('\\\\',"/",path)
    #print(path)
    return path

if __name__ == "__main__":
    print(11111111111111111111111111111)
    Demo = CRMain()
    imgPath=getCurrentFilePath()+'/'+"test_code"+'/'+'1311.jpg'
    num = Demo.getCodeNum(imgPath)
    print(num)
    num = Demo.getCodeNum(imgPath)
    print(num)

