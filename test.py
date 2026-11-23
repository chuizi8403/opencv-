import cv2
import numpy as np
import os

base = os.path.dirname(__file__)
defest = os.path.join(base,'KolektorSDD2','train')#图片文件夹路径
for name in os.listdir(defest):

    if name.endswith('_GT.png'):
        continue
    else:
        img =  os.path.join(defest,name)#文件夹路径+图片名字
                #转码
        data = np.fromfile(img,dtype = np.uint8)
        picture = cv2.imdecode(data,cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

        # 双边：平滑纹理，尽量保住弱边缘
        blur = cv2.bilateralFilter(gray, d=5, sigmaColor=12, sigmaSpace=30)

        # Scharr 梯度，抓取微弱划痕
        scharr_x = cv2.Scharr(blur, cv2.CV_64F, dx=1, dy=0)
        scharr_y = cv2.Scharr(blur, cv2.CV_64F, dx=0, dy=1)
        grad = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
        grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # 自适应阈值，适配局部光照、反光
        binary = cv2.adaptiveThreshold(
            grad, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=11,
            C=2
        )

        # 长条核：优先保留细长划痕，压制圆点脏点
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 1))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = picture.copy()

        # 特征筛选：划痕是细长条
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            long_side = max(w, h)
            short_side = min(w, h)
            ratio = long_side / (short_side + 1e-6)
            # 长宽比大，就是细长划痕
            if ratio > 3.0:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("gray", gray)
        cv2.imshow("gradient", grad)
        cv2.imshow("binary", binary)
        cv2.imshow("result", out)
        cv2.waitKey(0)
cv2.destroyAllWindows()