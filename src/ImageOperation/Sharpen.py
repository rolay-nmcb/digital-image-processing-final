import cv2
import numpy as np

"""
Sharpen.py - 图像锐化与高频增强模块

提供多种图像锐化方法：
- 频率域高通滤波（High_Pass）
- 空间域边缘检测（Laplacian, Sobel, Roberts, Scharr）
- 自定义加权图像增强（sharpen_by_space）

适用于图像细节增强、轮廓提取等任务。
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


def High_Pass(img, value, kernel='ideal'):
    """
    应用高通滤波器到输入图像

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
        b_filtered = _apply_highpass_filter(b, value, kernel)
        g_filtered = _apply_highpass_filter(g, value, kernel)
        r_filtered = _apply_highpass_filter(r, value, kernel)
        # 合并通道
        filtered_img = cv2.merge((b_filtered, g_filtered, r_filtered))
    else:
        filtered_img = _apply_highpass_filter(img, value, kernel)

    return filtered_img


def _apply_highpass_filter(channel, D0, kernel_type):
    """
       对单个通道应用指定类型的高通滤波器（理想/巴特沃斯/高斯）

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
                # 理想高通滤波器
                if D <= D0:
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1
            elif kernel_type == 'butterworth':
                # 巴特沃斯高通滤波器 (n=2)
                n = 2
                if D == 0:
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1 / (1 + (D0 / D) ** (2 * n))
            elif kernel_type == 'gauss':
                # 高斯高通滤波器
                mask[i, j] = 1 - np.exp(-(D ** 2) / (2 * (D0 ** 2)))

    # 应用滤波器
    fshift = dft_shift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 归一化
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    return np.uint8(img_back)
def shapren_kernel(src,code):
    """
       使用不同算子提取图像边缘（锐化核）

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           code (str): 锐化方法，可选 'laplacian', 'robels', 'sobel', 'scharr'

       返回:
           numpy.ndarray: 提取的边缘图像
       """
    img = src.copy()
    if code == 'laplacian':
        return cv2.Laplacian(img,-1,ksize=3)
    elif code == 'robels':
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(img, cv2.CV_16S, kernelx)
        y = cv2.filter2D(img, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
    elif code == 'sobel':
        SobelX = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        SobelY = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(SobelX)
        absY = cv2.convertScaleAbs(SobelY)
    elif code == 'scharr':
        ScharrX = cv2.Scharr(img, cv2.CV_16S, 1, 0)  # 计算 x 轴方向
        ScharrY = cv2.Scharr(img, cv2.CV_16S, 0, 1)  # 计算 y 轴方向
        absX = cv2.convertScaleAbs(ScharrX)  # 转回 uint8
        absY = cv2.convertScaleAbs(ScharrY)  # 转回 uint8
    kernel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 用绝对值近似平方根
    return kernel

def sharpen_by_space(src,code):
    """
       使用空域卷积核进行图像锐化处理

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           code (str): 锐化方法，可选 'laplacian', 'robels', 'sobel', 'scharr'

       返回:
           numpy.ndarray: 锐化后的图像
       """

    kernel = shapren_kernel(src,code)

    return cv2.addWeighted(src,1,kernel,-0.4,0)


def image_rebuild(imgRebuild):
    """
        对图像进行归一化重建，使其像素值落在 [0, 255] 范围内

        参数:
            imgRebuild (numpy.ndarray): 待重建的图像（任意 float 类型）

        返回:
            numpy.ndarray: 归一化并裁剪后的 uint8 图像
        """
    imgRebuild = (imgRebuild - np.min(imgRebuild)) / (np.max(imgRebuild) - np.min(imgRebuild)) * 255
    imgRebuild = np.uint8(np.clip(imgRebuild, 0, 255))
    return imgRebuild



if __name__ == '__main__':
    img = cv2.imread('../2.png', 1)
    result1 = High_Pass(img,10,'gauss')

    cv2.imshow('test3', result1)
    ket = cv2.waitKey(0)
