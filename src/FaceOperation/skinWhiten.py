import cv2
import numpy as np

"""
skinWhiten.py - 皮肤美白与磨皮算法实现

功能概述：
    - 提供基于 YCbCr 颜色空间的肤色检测与美白增强算法
    - 支持局部亮度增强 + RGB 增益调整
    - 提供双边滤波 + 高斯模糊结合的磨皮效果

依赖库：
    - OpenCV (cv2)
    - NumPy

使用说明：
    - 主要调用入口：SkinWhiten(), skinSmoothen()
"""
def SkinWhiten(img,value):
    """
       实现图像的肤色美白增强功能。

       参数:
           img (numpy.ndarray): 输入 BGR 图像
           value (float): 美白强度系数，范围通常为 [0.1, 1.0]，
                          数值越大，美白效果越明显

       返回:
           numpy.ndarray: 经过美白处理后的图像

       处理流程:
           1. 转换到 YCbCr 颜色空间
           2. 分析 Cb/Cr 通道统计信息以识别肤色区域
           3. 计算亮度 Y 通道中的高亮区域作为参考
           4. 对 RGB 三通道进行增益调整
           5. 限制输出值在 [0, 255] 范围内

       """
    img_YCbCr=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)

    # 将图片的RGB值转换成YCbCr值
    Y, Cb, Cr = cv2.split(img_YCbCr)

    # 计算Cb、Cr的均值
    Mb = np.mean(Cb)
    Mr = np.mean(Cr)

    # 计算Cb、Cr的方差
    Db = np.var(Cb)
    Dr = np.var(Cr)

    b1 = Cb - (Mb + Db * np.sign(Mb))
    b2 = Cr - (1.5 * Mr + Dr * np.sign(Mr))

    mask = (b1 < np.abs(1.5 * np.sqrt(Db))) & (b2 < np.abs(1.5 * np.sqrt(Dr)))

    Ciny = Y[mask].flatten()

    Ciny_sorted = np.sort(Ciny)[::-1]  # 从大到小排序
    nn = int(np.round(len(Ciny) / 10))
    Ciny2 = Ciny_sorted[:nn]


    mn = np.min(Ciny2)

    # 创建一个二值化掩码，用于选择near-white区域
    binary_mask = (Y >= mn).astype(np.int_)

    # 计算出图片的亮度的最大值的15%
    Ymax = np.max(Y)

    # 计算RGB通道的平均值
    B,G,R = cv2.split(img)
    # 使用掩码调整RGB通道
    R = cv2.convertScaleAbs(R)
    G = cv2.convertScaleAbs(G)
    B = cv2.convertScaleAbs(B)

    Rav = np.mean(R[binary_mask == 1])
    Gav = np.mean(G[binary_mask == 1])
    Bav = np.mean(B[binary_mask == 1])

    # 计算RGB通道的增益（避免除以零）
    Rgain = Ymax / Rav if Rav != 0 else 1
    Ggain = Ymax / Gav if Gav != 0 else 1
    Bgain = Ymax / Bav if Bav != 0 else 1

    # 通过增益调整图片的RGB三信道
    im_adjusted = img.copy().astype(np.float32)
    im_adjusted[:, :, 0] = im_adjusted[:, :, 0] * (1+Bgain)*value
    im_adjusted[:, :, 1] = im_adjusted[:, :, 1] * (1+Ggain)*value
    im_adjusted[:, :, 2] = im_adjusted[:, :, 2] * (1+Rgain)*value

    # 确保值在0-255范围内，并转换为uint8类型以保存或显示
    im_adjusted = np.clip(im_adjusted, 0, 255).astype(np.uint8)

    return  im_adjusted

def skinSmoothen(img,v1,v2):
    """
       实现图像的磨皮平滑处理，保留边缘细节。

       参数:
           img (numpy.ndarray): 输入 BGR 图像
           v1 (int): 控制双边滤波参数（影响滤波半径）
           v2 (int): 控制高斯模糊核大小

       返回:
           numpy.ndarray: 经过磨皮处理后的图像

       处理流程:
           1. 使用双边滤波保留边缘的同时平滑纹理
           2. 提取轮廓信息并叠加到原图
           3. 使用高斯模糊柔和过渡
           4. 混合原始图像与增强图像，提升肤质质感

       """
    try:
        dst = np.zeros_like(img)
        dx = v1 * 5
        fc = v1 * 12.5
        p = 0.1

        # 双边滤波
        temp1 = cv2.bilateralFilter(img, dx, fc, fc)
        # 图像矩阵相减,得到人脸轮廓
        temp2 = cv2.subtract(temp1, img)
        temp2 = cv2.add(temp2, (10, 10, 10, 128))
        # 高斯模糊
        temp3 = cv2.GaussianBlur(temp2, (2 * v2 - 1, 2 * v2 - 1), 0)
        # 原图叠加
        temp4 = cv2.add(img, temp3)
        # 按权重混合
        dst1 = cv2.addWeighted(img, p, temp4, 1 - p, 0.0)
        dst = cv2.add(dst1, (10, 10, 10, 255))

        return dst
    except:
        Exception("未检测到人脸")


if __name__ == '__main__':
    """
       测试用例：加载本地图片并应用美白+磨皮算法

       注意：
           - 替换路径为你自己的测试图片路径
           - 可调整参数观察不同美白/磨皮效果
       """
    img = cv2.imread('D:\\PythonWork\\digital-image-final\\src\\1.png',1)
    src = skinSmoothen(img,5,10)

    cv2.imshow('OTSU', src)
    ket = cv2.waitKey(0)

