import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

"""
Smooth.py - 图像平滑与降噪模块

提供多种图像平滑方法：
- 空间域滤波（中值滤波、双边滤波、均值漂移滤波）
- 频率域低通滤波（理想、巴特沃斯、高斯）

适用于去除噪声、模糊细节、保留整体结构等任务。
"""


def bgr_to_gray(src):
    """
        将输入图像转换为灰度图（如果还不是）。

        参数:
            src (numpy.ndarray): 输入图像（BGR 或 灰度）

        返回:
            numpy.ndarray: 灰度图像
        """
    if len(src.shape) != 2:
        src = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
    return src


def Space_Fliter(src, code):
    """
    空间滤波处理

    参数:
    src (numpy.ndarray): 输入图像，灰度图或彩色图
    code (str): 滤波类型，可选 'median'、'bilateral' 或 'mean'

    返回:
    numpy.ndarray: 滤波处理后的图像
    """
    result = src.copy()

    # 检查输入图像是否为灰度图
    is_gray = len(result.shape) == 2

    if code == 'median':
        result = cv2.medianBlur(result, 5)

    elif code == 'bilateral':
        result = cv2.bilateralFilter(result, d=0, sigmaColor=100, sigmaSpace=10)

    elif code == 'mean':
        # 均值漂移滤波需要彩色图像
        if is_gray:
            # 将灰度图转为伪彩色图（3通道）
            result_color = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            # 应用均值漂移滤波
            result_color = cv2.pyrMeanShiftFiltering(result_color, sp=15, sr=20)
            # 转回灰度图
            result = cv2.cvtColor(result_color, cv2.COLOR_BGR2GRAY)
        else:
            result = cv2.pyrMeanShiftFiltering(result, sp=15, sr=20)

    return result


def Low_Pass(img, value, kernel='ideal'):
    """
    应用低通滤波器到输入图像

    参数:
        img: 输入图像 (灰度或彩色)
        value: 滤波器控制参数 (截止频率)
        kernel: 滤波器类型 ('ideal', 'butterworth', 'gauss')

    返回:
        滤波后的图像
    """
    # 检查是否为彩色图像
    if len(img.shape) == 3:
        # 分离通道
        b, g, r = cv2.split(img)
        # 对各通道分别处理
        b_filtered = _apply_filter(b, value, kernel)
        g_filtered = _apply_filter(g, value, kernel)
        r_filtered = _apply_filter(r, value, kernel)
        # 合并通道
        filtered_img = cv2.merge((b_filtered, g_filtered, r_filtered))
    else:
        filtered_img = _apply_filter(img, value, kernel)

    return filtered_img


def _apply_filter(channel, D0, kernel_type):
    """
       对单个通道应用指定类型的低通滤波器（理想/巴特沃斯/高斯）

       参数:
           channel (numpy.ndarray): 单通道图像（灰度）
           D0 (float): 截止频率
           kernel_type (str): 滤波器类型，可选 'ideal', 'butterworth', 'gauss'

       返回:
           numpy.ndarray: 滤波后的单通道图像（uint8 格式）
       """
    # 获取图像尺寸
    rows, cols = channel.shape
    crow, ccol = rows // 2, cols // 2

    # 傅里叶变换
    dft = cv2.dft(np.float32(channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 创建滤波器
    mask = np.zeros((rows, cols, 2), np.float32)

    for i in range(rows):
        for j in range(cols):
            # 计算距离中心点的距离
            D = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)

            if kernel_type == 'ideal':
                # 理想低通滤波器
                if D <= D0:
                    mask[i, j] = 1
                else:
                    mask[i, j] = 0
            elif kernel_type == 'butterworth':
                # 巴特沃斯低通滤波器 (n=2)
                n = 2
                mask[i, j] = 1 / (1 + (D / D0) ** (2 * n))
            elif kernel_type == 'gauss':
                # 高斯低通滤波器
                mask[i, j] = np.exp(-(D ** 2) / (2 * (D0 ** 2)))

    # 应用滤波器
    fshift = dft_shift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(img_back)
if __name__ == '__main__':
    img = cv2.imread('../1.png', 1)
    result = Low_Pass(img,30,'gauss')

    cv2.imshow('test3', result)
    ket = cv2.waitKey(0)