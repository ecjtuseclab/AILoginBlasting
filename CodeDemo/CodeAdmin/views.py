from django.shortcuts import render
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.urls import reverse
#from django.core import serializers
from django.forms.models import model_to_dict
from datetime import date,datetime
import json
import math
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import re
import urllib.request
import os
import filetype,hashlib
from CaptchaRecognition import CRMain
CR = CRMain.CRMain()
# Create your views here.

def index(request):
    #return HttpResponse(str("sdsdsdsd"))
	return render(request, 'index.html')

def home(request):
    #return HttpResponse(str("sdsdsdsd"))
	return render(request, 'home.html')

def uploadimg(request):
    return render(request, 'uploadimg.html')

def fileupload(request):
    return HttpResponse(str('ssssss'))

#对外提供获取验证码API
def GetCodeNum(request):
    try:
        print('GetCodeNum startttttttttttttttttttttttttttttttttttttttt')
        # 从请求表单中获取文件对象
        file = request.FILES.get("img", None)
        if not file:  # 文件对象不存在， 返回400请求错误
            return HttpResponse("need img.")

        # 图片大小限制
        if not pIsAllowedFileSize(file.size):
            return HttpResponse("文件太大")

        # 计算文件md5
        md5 = pCalculateMd5(file)

        # 获取扩展类型 并 判断
        ext = pGetFileExtension(file)
        if not pIsAllowedImageType(ext):
            return HttpResponse("文件类型错误")

        # 检测通过 创建新的image对象
        # 保存 文件到磁盘
        imgpath = getCurrentFilePath() + '/img/' + md5 + '.' + ext
        print(imgpath)
        with open(imgpath, "wb+") as f:
            # 分块写入
            for chunk in file.chunks():
                f.write(chunk)

        #调用识别程序进行识别
        image_num = CR.getCodeNum(imgpath)

        # 返回图片的url以供访问
        return HttpResponse('{"status":"succses","num":'+image_num+'}')
    except  Exception as e:
        print(str(e))
        return HttpResponse('{"status":"failed","num":0}')

# 上传文件的视图
#@require_http_methods(["POST"])
@csrf_exempt
def uploadImage(request):
    print('uploadImage startttttttttttttttttttttttttttttttttttttttt')
    # 从请求表单中获取文件对象
    file = request.FILES.get("file", None)
    if not file:  # 文件对象不存在， 返回400请求错误
        return HttpResponse("need file.")

    # 图片大小限制
    if not pIsAllowedFileSize(file.size):
        return HttpResponse("文件太大")

    #计算文件md5
    md5 = pCalculateMd5(file)

    # 获取扩展类型 并 判断
    ext = pGetFileExtension(file)
    if not pIsAllowedImageType(ext):
        return HttpResponse("文件类型错误")

    # 检测通过 创建新的image对象
    # 保存 文件到磁盘
    imgpath = getCurrentFilePath()+'/img/'+ md5 + '.' + ext
    print(imgpath)
    with open(imgpath, "wb+") as f:
        # 分块写入
        for chunk in file.chunks():
            f.write(chunk)

    # 返回图片的url以供访问
    return HttpResponse({"url": imgpath})


# 检测文件类型
# 我们使用第三方的库filetype进行检测，而不是通过文件名进行判断
# pip install filetype 即可安装该库
def pGetFileExtension(file):
    rawData = bytearray()
    for c in file.chunks():
        rawData += c
    try:
        ext = filetype.guess_extension(rawData)
        return ext
    except Exception as e:
        # todo log
        return None

# 计算文件的md5
def pCalculateMd5(file):
    md5Obj = hashlib.md5()
    for chunk in file.chunks():
        md5Obj.update(chunk)
    return md5Obj.hexdigest()

# 文件类型过滤 我们只允许上传常用的图片文件
def pIsAllowedImageType(ext):
    if ext in ["png", "jpeg", "jpg"]:
        return True
    return False

# 文件大小限制
# settings.IMAGE_SIZE_LIMIT是常量配置，我设置为10M
def pIsAllowedFileSize(size):
    limit = 10*1024*1024    #M
    if size < limit:
        return True
    return False


@csrf_exempt
def api(request):
    imgUrl = request.GET['imgurl']
    imgName = hash(imgUrl)
    filePath = getCurrentFilePath()+'/img/'+ str(imgName)+".jpg"
    flag=SaveImage(filePath,imgName,imgUrl)
    num = 4778#Demo.getCodeNum(filePath)
    print(num)

    return HttpResponse(str(num))
 
def getCurrentFilePath():
    #获得当前工作目录
    path = os.getcwd()
    path = re.sub('\\\\',"/",path)
    #print(path)
    return path

def SaveImage(filePath,imgName,imgUrl):
    print(1111111)
    flag = True
    # ------ 这里最好使用异常处理及多线程编程方式 ------
    try:
        print("filePath==="+filePath)
        f = open(filePath, 'wb')
        print("imgUrl==="+imgUrl)
        f.write((urllib.request.urlopen(imgUrl)).read())
        #print(imgUrl)
        f.close()
        
    except Exception as e:
        print("e===="+str(e))
        #print(imgUrl+" error")
        flag = False
        
    return flag