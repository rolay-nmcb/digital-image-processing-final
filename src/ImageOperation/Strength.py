import numpy as np
import cv2

"""
Strength.py - 图像增强与色彩调整模块

提供多种图像增强方法：
- 亮度/对比度调整（BorC_Convert）
- 直方图均衡化（hisEqulColor1）
- 自适应直方图均衡（hisEqulColor2）
- Gamma 曝光调整（Exposure_by_Gamma）
- 色调、饱和度、明度调整（Hue_by_LUT / Saturation_by_LUT / Value_by_LUT）
- 灰度线性映射、对数增强、分层显示等

适用于图像质量提升、细节增强、视觉效果优化等任务。
"""

def BorC_Convert(img,value,type):
    """
        调整图像的亮度或对比度。

        参数:
            img (numpy.ndarray): 输入图像（BGR 格式）
            value (float): 调整值：亮度用绝对值，对比度用百分比
            type (str): 调整类型，可选 'bright'（亮度）或 'contrast'（对比度）

        返回:
            numpy.ndarray: 调整后的图像
        """
    result = None
    if type == 'bright':
        result = cv2.convertScaleAbs(img, beta=value)
    elif type == 'contrast':
        result = cv2.convertScaleAbs(img, alpha=value/100)
    return result

# 彩色图像全局直方图均衡化
def hisEqulColor1(img):
    """
       对彩色图像进行全局直方图均衡化处理（增强整体对比度）。

       参数:
           img (numpy.ndarray): 输入图像（BGR 格式）

       返回:
           numpy.ndarray: 均衡化后的图像（BGR 格式）
       """
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


# 彩色图像进行自适应直方图均衡化
def hisEqulColor2(img):
    """
       对彩色图像进行自适应直方图均衡化（CLAHE），增强局部对比度。

       参数:
           img (numpy.ndarray): 输入图像（BGR 格式）

       返回:
           numpy.ndarray: 增强后的图像（BGR 格式）
       """
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

# 修改曝光
def Exposure_by_Gamma(img, gamma):
    """
       使用 Gamma 变换调整图像曝光（非线性亮度增强）。

       参数:
           img (numpy.ndarray): 输入图像（BGR 格式）
           gamma (float): Gamma 值，小于 1 提亮，大于 1 变暗

       返回:
           numpy.ndarray: 曝光调整后的图像
       """
    # gamma函数处理
    # 建立映射表
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    # 颜色值为整数
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 对输入的src执行查找表lut转换
    result = cv2.LUT(img, gamma_table)
    return result

# def Hue_by_LUT(src,limit,tunnel):
#     """
#        通过查找表（LUT）修改指定通道的色调。
#
#        参数:
#            src (numpy.ndarray): 输入图像（BGR 格式）
#            limit (int): 修改强度（0-255）
#            tunnel (str): 指定通道，可选 'B', 'G', 'R'
#
#        返回:
#            numpy.ndarray: 修改色调后的图像（BGR 格式）
#        """
#     lutHalf = np.array([int(i * limit / 255) for i in range(256)]).astype("uint8")
#     lutEqual = np.array([i for i in range(256)]).astype("uint8")
#     if tunnel == 'B':
#         lut3HalfB = np.dstack((lutHalf, lutEqual, lutEqual))
#         return cv2.LUT(src, lut3HalfB)
#     elif tunnel=='G':
#         lut3HalfG = np.dstack((lutEqual, lutHalf, lutEqual))
#         return cv2.LUT(src, lut3HalfG)
#     else :
#         lut3HalfR = np.dstack((lutEqual, lutEqual, lutHalf))
#         return  cv2.LUT(src, lut3HalfR)

def Saturation_by_LUT(src,percentage):
    """
       通过 LUT 增强或减弱图像饱和度。

       参数:
           src (numpy.ndarray): 输入图像（BGR 格式）
           percentage (float): 饱和度增强比例（>1 增强，<1 减弱）

       返回:
           numpy.ndarray: 饱和度调整后的图像（BGR 格式）
       """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lutEqual = np.array([i for i in range(256)]).astype("uint8")
    if percentage > 1:
        lutNew = np.array([int(percentage * i) for i in range(256)]).astype("uint8")
    else :
        lutNew = np.array([int(100+(percentage-1) * i) for i in range(256)]).astype("uint8")
    lutSResult = np.dstack((lutEqual, lutNew, lutEqual))
    result = cv2.LUT(hsv, lutSResult )

    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

def Value_by_LUT(src,percentage):
    """
       通过 LUT 增强或减弱图像明度（HSV 中 V 通道）。

       参数:
           src (numpy.ndarray): 输入图像（BGR 格式）
           percentage (float): 明度增强比例（>1 增强，<1 减弱）

       返回:
           numpy.ndarray: 明度调整后的图像（BGR 格式）
       """
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lutEqual = np.array([i for i in range(256)]).astype("uint8")
    if percentage > 1:
        lutNew = np.array([int(percentage * i) for i in range(256)]).astype("uint8")
    else:
        lutNew = np.array([int(100 + (percentage - 1) * i) for i in range(256)]).astype("uint8")
    lutVResult = np.dstack((lutEqual, lutEqual, lutNew))
    result = cv2.LUT(hsv, lutVResult)
    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

def is_gray_image(src):
    """
       判断输入图像是否为灰度图像。

       参数:
           src (numpy.ndarray): 输入图像

       返回:
           bool: 是否为灰度图像
       """
    if len(src.shape) == 2:
        return True
    return False

def linear_enhancement(src,a,b,c,d):
    """
       灰度线性变换增强（y = ((d-c)/(b-a))x + (b*c - a*d)/(b-a)）。

       参数:
           src (numpy.ndarray): 输入图像（BGR 或 GRAY）
           a (float): 输入最小灰度值
           b (float): 输入最大灰度值
           c (float): 输出最小灰度值
           d (float): 输出最大灰度值

       返回:
           numpy.ndarray: 线性增强后的图像（BGR 格式）
       """
    if is_gray_image(src) == False:
        src= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    result = (d - c) / (b - a) * src + (b * c - a * d) / (b - a)

    return cv2.cvtColor(np.array(result, np.uint8), cv2.COLOR_GRAY2BGR)

def log_enhancement(src):
    """
       对图像应用对数变换以增强暗部细节。

       参数:
           src (numpy.ndarray): 输入图像（BGR 或 GRAY）

       返回:
           numpy.ndarray: 对数增强后的图像（BGR 格式）
       """
    if is_gray_image(src) == False:
        src= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    C = 255 / np.log(1 + np.max(src))
    src_clipped = np.clip(src, -1 + np.finfo(float).eps, None)
    result = C * np.log(1 + src_clipped)
    return cv2.cvtColor(np.array(result, np.uint8),cv2.COLOR_GRAY2BGR)

def gray_hist(src):
    """
       对灰度图像执行全局直方图均衡化（增强整体对比度）。

       参数:
           src (numpy.ndarray): 输入图像（BGR 或 GRAY）

       返回:
           numpy.ndarray: 均衡化后的图像（BGR 格式）
       """
    if is_gray_image(src) == False:
        src= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    result = cv2.equalizeHist(src)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def gray_layered(src,code,value1,value2):
    """
       灰度分层增强：突出指定灰度区间（二值化显示）。

       参数:
           src (numpy.ndarray): 输入图像（BGR 或 GRAY）
           code (str): 处理方式，目前仅支持 'binary'
           value1 (int): 灰度下限
           value2 (int): 灰度上限

       返回:
           numpy.ndarray: 分层增强后的图像（BGR 格式）
       """
    if is_gray_image(src) == False:
        src= cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    result = src.copy()
    if code == 'binary':
        result[(result[:, :] < value1) | (result[:, :] > value2)] = 0  # 二值处理其它区域：黑色
    result[(result[:, :] >= value1) & (result[:, :] <= value2)] = 255  # 灰度级窗口：白色
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

def gray_gamma(src,value):
    """
       应用 Gamma 校正增强图像对比度。

       参数:
           src (numpy.ndarray): 输入图像（BGR 或 GRAY）
           value (float): Gamma 值，用于非线性对比度调整

       返回:
           numpy.ndarray: Gamma 增强后的图像（BGR 格式）
       """
    result = np.power(src,value)
    imgRebuild = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255
    imgRebuild = np.uint8(np.clip(imgRebuild, 0, 255))
    return cv2.cvtColor(imgRebuild, cv2.COLOR_GRAY2BGR)


def Hue_by_LUT(bgr_img, hue_shift):
    """
    调整BGR图像的色相（Hue），自动兼容灰度图输入

    参数:
        bgr_img: 输入图像（OpenCV BGR格式或灰度图，uint8类型）
        hue_shift: 色相偏移量（0-180，OpenCV的Hue范围）

    返回:
        色相调整后的BGR图像（灰度图直接返回原图）
    """
    # 检查是否为灰度图（单通道）
    if len(bgr_img.shape) == 2:
        return bgr_img.copy()

    # 检查是否为3通道BGR图像
    elif len(bgr_img.shape) == 3 and bgr_img.shape[2] == 3:
        # 转换到HSV色彩空间
        hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

        # 调整Hue通道
        hsv_img = hsv_img.astype(np.int32)
        hsv_img[..., 0] = (hsv_img[..., 0] + hue_shift) % 180
        hsv_img = np.clip(hsv_img, 0, 255).astype(np.uint8)

        # 转回BGR
        return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    else:
        raise ValueError("输入图像必须是灰度图（2D）或3通道BGR图像（3D）")

