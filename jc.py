
import cv2
import numpy as np
import os

sumn = 0#记录合格工件
count = 0#记录瑕疵工件
min_weight =10
min_high = 10

#提取中文路径下的图片
base = os.path.dirname(__file__)
defest = os.path.join(base,'IMAGES')#图片文件夹路径
e=0
for name in os.listdir(defest):
   
    img =  os.path.join(defest,name)#文件夹路径+图片名字
    #转码
    data = np.fromfile(img,dtype = np.uint8)
    picture = cv2.imdecode(data,cv2.IMREAD_COLOR)

    kl = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))#卷积核

    img1 = cv2.cvtColor(picture,cv2.COLOR_BGR2GRAY)#灰度
    img2 = cv2.GaussianBlur(img1,(3,3),1)#高斯滤波
    img3 = cv2.morphologyEx(img2, cv2.MORPH_TOPHAT, kl)#顶帽
    img4 = cv2.adaptiveThreshold(img3,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,5,1)   #自适应二值化处理

    #闭运算
    img5 = cv2.morphologyEx(img4,cv2.MORPH_CLOSE,kl)
    
    #轮廓查找
    crs , hhy = cv2.findContours(img5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    #检测是否为缺陷
    for i,c in enumerate(crs):
        x,y,w,h = cv2.boundingRect(c)
        #进行判断验证是否为有效缺陷画轮廓
        isValid = (w >= min_weight) and (h >= min_high)
        if (not isValid):
            continue
        cv2.drawContours(picture, [c] , -1, (0, 0,255),1)#绘制轮廓

        are = cv2.arcLength(c ,True)
        print(are )
    #     if((are<=1934)and(are>=1534) ):
    #         sumn = sumn+1
    #         cv2.putText(picture,'合格:'+str(sumn),(20,550),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3 )
    #         cv2.putText(picture,'瑕疵:'+str(count),(20,500),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3 )

    #     else:
    #         count = count + 1 
    #         cv2.putText(picture,'瑕疵:'+str(count),(20,500),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3 )
    #         cv2.putText(picture,'合格:'+str(sumn),(20,550),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,0),3 )

    # if picture is None:
    #     print('图片读取失败')
    # else:
    cv2.imshow('picture',picture)

    #cv2.imshow('name',img5)
        
    

    key = cv2.waitKey(0)
    if(key == ord('q')):
        break

   

cv2.destroyAllWindows()





