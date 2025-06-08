import cv2
import numpy as np
import pywt


"""
Restoration.py - 图像恢复与去噪处理模块

包含多种噪声添加方法和对应的滤波器去噪方法，支持：
- 高斯噪声（Gauss_Noise）
- 瑞利噪声（Rayleigh_Noise）
- 伽马噪声（Ireland_Noise）
- 指数噪声（Exponential_Noise）
- 均匀噪声（Uniform_Noise）
- 椒盐噪声（SaltPepper_Noise）
- 算术/几何/调和均值滤波器
- 统计排序滤波器（中值、最大值、最小值等）

所有滤波器均兼容灰度图和彩色图。
"""

def Gauss_Noise(src, mu=0, sigma=20):
    """
       向图像添加高斯噪声。

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           mu (float): 噪声均值，默认为 0
           sigma (float): 噪声标准差，默认为 20

       返回:
           numpy.ndarray: 添加高斯噪声后的图像（uint8 格式）
       """
    noiseGause = np.random.normal(mu, sigma, src.shape)
    imgGaussNoise = src + noiseGause
    imgGaussNoise = np.uint8(cv2.normalize(imgGaussNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgGaussNoise

def Rayleigh_Noise(src,a=30):
    """
       向图像添加瑞利噪声。

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           a (float): 瑞利分布的尺度参数，默认为 30

       返回:
           numpy.ndarray: 添加瑞利噪声后的图像（uint8 格式）
       """
    noiseRayleigh = np.random.rayleigh(a, size=src.shape)
    imgRayleighNoise = src + noiseRayleigh
    imgRayleighNoise = np.uint8(cv2.normalize(imgRayleighNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgRayleighNoise

def Ireland_Noise(src,a=10,b=2.5):
    """
       向图像添加伽马噪声（爱尔兰噪声）。

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           a (float): 尺度参数，默认为 10
           b (float): 形状参数，默认为 2.5

       返回:
           numpy.ndarray: 添加伽马噪声后的图像（uint8 格式）
       """
    noiseGamma = np.random.gamma(shape=b, scale=a, size=src.shape)
    imgGammaNoise = src + noiseGamma
    imgGammaNoise = np.uint8(cv2.normalize(imgGammaNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgGammaNoise


def Exponential_Noise(src,a=10):
    """
       向图像添加指数噪声。

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           a (float): 指数分布的尺度参数，默认为 10

       返回:
           numpy.ndarray: 添加指数噪声后的图像（uint8 格式）
       """
    noiseExponent = np.random.exponential(scale=a, size=src.shape)
    imgExponentNoise = src + noiseExponent
    imgExponentNoise = np.uint8(cv2.normalize(imgExponentNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgExponentNoise

def Uniform_Noise(src, mean=10, sigma=100):
    """
        向图像添加均匀分布噪声。

        参数:
            src (numpy.ndarray): 输入图像（灰度或BGR格式）
            mean (float): 均匀分布的均值，默认为 10
            sigma (float): 均匀分布的标准差，默认为 100

        返回:
            numpy.ndarray: 添加均匀噪声后的图像（uint8 格式）
        """
    a = 2 * mean - np.sqrt(12 * sigma)
    b = 2 * mean + np.sqrt(12 * sigma)
    noiseUniform = np.random.uniform(a, b, src.shape)
    imgUniformNoise = src + noiseUniform
    imgUniformNoise = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))
    return imgUniformNoise

def SaltPepper_Noise(src,salt=0.02, pepper=0.02):
    """
       向图像添加椒盐噪声。

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           salt (float): 盐噪声比例（白点），默认为 0.02
           pepper (float): 椒噪声比例（黑点），默认为 0.02

       返回:
           numpy.ndarray: 添加椒盐噪声后的图像（uint8 格式）
       """
    mask = np.random.choice((0, 0.5, 1), size=src.shape[:2], p=[pepper, (1 - salt - pepper), salt])
    imgChoiceNoise = src.copy()
    imgChoiceNoise[mask == 1] = 255
    imgChoiceNoise[mask == 0] = 0
    return imgChoiceNoise


def Arithmetic_Mean_Filter(src,size=5):
    """
       使用算术均值滤波器去除图像噪声。

       参数:
           src (numpy.ndarray): 输入图像（灰度或BGR格式）
           size (int): 滤波器窗口大小（奇数，默认为5）

       返回:
           numpy.ndarray: 滤波后的图像（与输入类型一致）
       """
    kSize = (size, size)
    kernalMean = np.ones(kSize, np.float32) / (kSize[0] * kSize[1])  # 生成归一化盒式核
    result = cv2.filter2D(src, -1, kernalMean)
    return result


def Geometric_Mean_Filter(src, size=5):
    """
    几何均值滤波器（兼容灰度图和彩色图）

    参数:
        src: 输入图像（灰度或彩色）
        size: 滤波器大小（奇数，默认5）

    返回:
        滤波后的图像
    """
    if size % 2 == 0:
        raise ValueError("滤波器大小必须为奇数")

    # 灰度图处理（单通道）
    if len(src.shape) == 2:
        img = src.copy()
        height, width = src.shape
        order = 1.0 / (size * size)
        hPad = size // 2

        # 边缘填充（避免边界问题）
        imgPad = np.pad(img, ((hPad, hPad), (hPad, hPad)), mode="edge")

        # 初始化输出
        result = np.zeros_like(img, dtype=np.float32)

        # 滑动窗口计算几何均值
        for i in range(hPad, height + hPad):
            for j in range(hPad, width + hPad):
                window = imgPad[i - hPad:i + hPad + 1, j - hPad:j + hPad + 1]
                prod = np.prod(window.astype(np.float64) + 1e-10)  # 避免零值导致log爆炸
                result[i - hPad, j - hPad] = np.power(prod, order)

        return np.clip(result, 0, 255).astype(src.dtype)

    # 彩色图处理（多通道）
    elif len(src.shape) == 3:
        img = src.copy()
        height, width, channels = src.shape
        order = 1.0 / (size * size)
        hPad = size // 2

        # 初始化输出
        result = np.zeros_like(img, dtype=np.float32)

        # 分通道处理
        for k in range(channels):
            imgPad = np.pad(img[:, :, k], ((hPad, hPad), (hPad, hPad)), mode="edge")

            for i in range(hPad, height + hPad):
                for j in range(hPad, width + hPad):
                    window = imgPad[i - hPad:i + hPad + 1, j - hPad:j + hPad + 1]
                    prod = np.prod(window.astype(np.float64) + 1e-10)
                    result[i - hPad, j - hPad, k] = np.power(prod, order)

        return np.clip(result, 0, 255).astype(src.dtype)

    else:
        raise ValueError("输入图像必须是灰度图（2D）或彩色图（3D）")


def Harmonic_Mean_Filter(src, size=5):
    """
    调和均值滤波器（兼容灰度图和彩色图）

    参数:
        src: 输入图像（灰度或彩色）
        size: 滤波器大小（奇数，默认5）

    返回:
        滤波后的图像
    """
    if size % 2 == 0:
        raise ValueError("滤波器大小必须为奇数")

    # 灰度图处理（单通道）
    if len(src.shape) == 2:
        img = src.copy()
        height, width = src.shape
        order = size * size
        hPad = size // 2
        epsilon = 1e-8  # 避免除以零

        # 边缘填充（避免边界问题）
        imgPad = np.pad(img, ((hPad, hPad), (hPad, hPad)), mode="edge")

        # 初始化输出
        result = np.zeros_like(img, dtype=np.float32)

        # 滑动窗口计算调和均值
        for i in range(hPad, height + hPad):
            for j in range(hPad, width + hPad):
                window = imgPad[i - hPad:i + hPad + 1, j - hPad:j + hPad + 1]
                sum_temp = np.sum(1.0 / (window.astype(np.float64) + epsilon))
                result[i - hPad, j - hPad] = order / sum_temp

        return np.clip(result, 0, 255).astype(src.dtype)

    # 彩色图处理（多通道）
    elif len(src.shape) == 3:
        img = src.copy()
        height, width, channels = src.shape
        order = size * size
        hPad = size // 2
        epsilon = 1e-8

        # 初始化输出
        result = np.zeros_like(img, dtype=np.float32)

        # 分通道处理
        for k in range(channels):
            imgPad = np.pad(img[:, :, k], ((hPad, hPad), (hPad, hPad)), mode="edge")

            for i in range(hPad, height + hPad):
                for j in range(hPad, width + hPad):
                    window = imgPad[i - hPad:i + hPad + 1, j - hPad:j + hPad + 1]
                    sum_temp = np.sum(1.0 / (window.astype(np.float64) + epsilon))
                    result[i - hPad, j - hPad, k] = order / sum_temp

        return np.clip(result, 0, 255).astype(src.dtype)

    else:
        raise ValueError("输入图像必须是灰度图（2D）或彩色图（3D）")


def Statistical_Sorting_Filter(src, code, size=3):
    """
    统计排序滤波器（兼容灰度图和彩色图）

    参数:
        src: 输入图像（灰度或彩色）
        code: 滤波类型（'median', 'max', 'min', 'middle'）
        size: 滤波器大小（奇数，默认3）

    返回:
        滤波后的图像
    """
    if size % 2 == 0:
        raise ValueError("滤波器大小必须为奇数")

    if code not in ['median', 'max', 'min', 'middle']:
        raise ValueError("code 必须是 'median', 'max', 'min' 或 'middle'")

    # 灰度图处理（单通道）
    if len(src.shape) == 2:
        img = src.copy()
        height, width = src.shape
        hPad = size // 2

        # 边缘填充（避免边界问题）
        imgPad = np.pad(img, ((hPad, hPad), (hPad, hPad)), mode="edge")

        # 初始化输出
        result = np.zeros_like(img, dtype=np.uint8)

        # 滑动窗口计算统计排序
        for i in range(height):
            for j in range(width):
                window = imgPad[i:i + size, j:j + size]
                if code == 'median':
                    result[i, j] = np.median(window)
                elif code == 'max':
                    result[i, j] = np.max(window)
                elif code == 'min':
                    result[i, j] = np.min(window)
                elif code == 'middle':
                    result[i, j] = int(window.max() / 2 + window.min() / 2)

        return result

    # 彩色图处理（多通道）
    elif len(src.shape) == 3:
        img = src.copy()
        height, width, channels = src.shape
        hPad = size // 2

        # 初始化输出
        result = np.zeros_like(img, dtype=np.uint8)

        # 分通道处理
        for k in range(channels):
            imgPad = np.pad(img[:, :, k], ((hPad, hPad), (hPad, hPad)), mode="edge")

            for i in range(height):
                for j in range(width):
                    window = imgPad[i:i + size, j:j + size]
                    if code == 'median':
                        result[i, j, k] = np.median(window)
                    elif code == 'max':
                        result[i, j, k] = np.max(window)
                    elif code == 'min':
                        result[i, j, k] = np.min(window)
                    elif code == 'middle':
                        result[i, j, k] = int(window.max() / 2 + window.min() / 2)

        return result

    else:
        raise ValueError("输入图像必须是灰度图（2D）或彩色图（3D）")


def wavelet_compress(image, wavelet='db1', level=1, threshold=0.1, mode='soft'):
    """
    使用小波变换对图像进行压缩

    参数:
    image: numpy数组，原图像（灰度或彩色）
    wavelet: 小波函数名称，默认为'Daubechies 1'
    level: 分解级别，默认为1
    threshold: 阈值比例，用于决定保留多少系数，范围(0,1)
    mode: 阈值处理方式，默认为'soft'

    返回:
    numpy数组，压缩后的图像，格式为np.uint8
    """
    # 检查输入图像是否为彩色（三维数组）
    is_color = len(image.shape) == 3

    # 对彩色图像的每个通道分别处理
    if is_color:
        compressed_channels = []
        for channel in range(image.shape[2]):
            compressed = _process_channel(image[:, :, channel], wavelet, level, threshold, mode)
            compressed_channels.append(compressed)
        # 合并通道
        result = np.stack(compressed_channels, axis=2)
    else:
        # 灰度图像直接处理
        result = _process_channel(image, wavelet, level, threshold, mode)

    # 确保结果在uint8范围内
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def _process_channel(channel, wavelet, level, threshold, mode):
    """处理单个通道的小波变换和压缩"""
    # 执行小波分解
    coeffs = pywt.wavedec2(channel, wavelet, level=level)

    # 计算阈值
    threshold_value = threshold * np.max(np.abs(coeffs[0]))

    # 对细节系数应用阈值处理
    processed_coeffs = [coeffs[0]]  # 近似系数保持不变
    for detail_level in coeffs[1:]:
        # 对每个方向的细节系数应用阈值
        processed_detail = [pywt.threshold(d, threshold_value, mode=mode) for d in detail_level]
        processed_coeffs.append(processed_detail)

    # 执行小波重构
    reconstructed = pywt.waverec2(processed_coeffs, wavelet)

    # 处理尺寸可能的微小差异（由于小波变换的离散性质）
    if reconstructed.shape != channel.shape:
        # 裁剪或填充以匹配原始尺寸
        reconstructed = _match_size(reconstructed, channel.shape)

    return reconstructed


# def wavelet_denoise(image, wavelet='db4', level=1, mode='soft', sigma=None):
#     """
#     使用小波变换对图像进行去噪
#
#     参数:
#     image: numpy数组，原图像（灰度或彩色）
#     wavelet: 小波函数名称，默认为'Daubechies 4'
#     level: 分解级别，默认为1
#     mode: 阈值处理方式，默认为'soft'
#     sigma: 噪声标准差，若为None则自动估计
#
#     返回:
#     numpy数组，去噪后的图像，格式为np.uint8
#     """
#     # 检查输入图像是否为彩色（三维数组）
#     is_color = len(image.shape) == 3
#
#     # 对彩色图像的每个通道分别处理
#     if is_color:
#         denoised_channels = []
#         for channel in range(image.shape[2]):
#             denoised = _process_channel(image[:, :, channel], wavelet, level, mode, sigma)
#             denoised_channels.append(denoised)
#         # 合并通道
#         result = np.stack(denoised_channels, axis=2)
#     else:
#         # 灰度图像直接处理
#         result = _process_channel(image, wavelet, level, mode, sigma)
#
#     # 确保结果在uint8范围内
#     result = np.clip(result, 0, 255).astype(np.uint8)
#     return result
#
#
# def _process_channel(channel, wavelet, level, mode, sigma):
#     """处理单个通道的小波变换和去噪"""
#     # 执行小波分解
#     coeffs = pywt.wavedec2(channel, wavelet, level=level)
#
#     # 估计噪声标准差（若未提供）
#     if sigma is None:
#         # 使用细节系数的中位数估计噪声标准差
#         sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
#
#     # 计算阈值（通用阈值：sigma * sqrt(2 * log(n))）
#     n = channel.size
#     threshold = sigma * np.sqrt(2 * np.log(n))
#
#     # 对细节系数应用阈值处理
#     processed_coeffs = [coeffs[0]]  # 近似系数保持不变
#     for detail_level in coeffs[1:]:
#         # 对每个方向的细节系数应用阈值
#         processed_detail = [pywt.threshold(d, threshold, mode=mode) for d in detail_level]
#         processed_coeffs.append(processed_detail)
#
#     # 执行小波重构
#     reconstructed = pywt.waverec2(processed_coeffs, wavelet)
#
#     # 处理尺寸可能的微小差异
#     if reconstructed.shape != channel.shape:
#         reconstructed = _match_size(reconstructed, channel.shape)
#
#     return reconstructed

def wavelet_denoise(image, wavelet='db1', level=3, mode='soft', threshold_multiplier=0.1):
    """
    使用小波变换对图像进行去噪

    参数:
        image: 输入图像 (灰度或彩色), NumPy数组
        wavelet: 使用的小波基, 默认为'db1'(Daubechies小波)
        level: 小波分解的层数, 默认为3
        mode: 阈值处理模式, 'soft'或'hard', 默认为'soft'
        threshold_multiplier: 阈值乘数, 用于调整去噪强度, 默认为1.0

    返回:
        去噪后的图像, uint8类型的NumPy数组
    """
    # 确保图像是float32类型(0-1范围或0-255范围都可以，下面会处理)
    if image.dtype != np.float32:
        image = image.astype(np.float32)

    # 如果是彩色图像(3通道)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # 分别对每个通道进行处理
        denoised_channels = []
        for c in range(3):
            channel = image[:, :, c]
            # 将通道值归一化到0-1范围
            channel_normalized = channel / 255.0 if channel.max() > 1 else channel
            # 小波变换去噪
            denoised_channel = _wavelet_denoise_channel(channel_normalized, wavelet, level, mode, threshold_multiplier)
            # 恢复原始范围
            denoised_channel = denoised_channel * 255.0 if channel.max() > 1 else denoised_channel
            denoised_channels.append(denoised_channel)

        # 合并通道
        denoised_image = np.stack(denoised_channels, axis=2)
    else:
        # 灰度图像处理
        image_normalized = image / 255.0 if image.max() > 1 else image
        denoised_image = _wavelet_denoise_channel(image_normalized, wavelet, level, mode, threshold_multiplier)
        denoised_image = denoised_image * 255.0 if image.max() > 1 else denoised_image

    # 确保结果在0-255范围内并转换为uint8
    denoised_image = np.clip(denoised_image, 0, 255).astype(np.uint8)
    return denoised_image


def _wavelet_denoise_channel(channel, wavelet, level, mode, threshold_multiplier):
    """
    对单通道图像进行小波去噪
    """
    # 小波分解
    coeffs = pywt.wavedec2(channel, wavelet, level=level)

    # 计算阈值(使用第一层细节系数的标准差作为基础)
    sigma = np.median(np.abs(coeffs[1][0])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(channel.size)) * threshold_multiplier

    # 对细节系数进行阈值处理
    new_coeffs = [coeffs[0]]  # 保留近似系数

    for i in range(1, len(coeffs)):
        new_detail = list(coeffs[i])
        for j in range(len(new_detail)):
            if mode == 'soft':
                new_detail[j] = pywt.threshold(new_detail[j], threshold, mode='soft')
            else:
                new_detail[j] = pywt.threshold(new_detail[j], threshold, mode='hard')
        new_coeffs.append(tuple(new_detail))

    # 小波重构
    denoised_channel = pywt.waverec2(new_coeffs, wavelet)

    # 确保重构后的图像与输入尺寸一致(有时会因为小波变换而差1个像素)
    if denoised_channel.shape != channel.shape:
        denoised_channel = denoised_channel[:channel.shape[0], :channel.shape[1]]

    return denoised_channel

def _match_size(arr, target_shape):
    """调整数组大小以匹配目标形状"""
    result = np.zeros(target_shape, dtype=arr.dtype)
    min_rows = min(arr.shape[0], target_shape[0])
    min_cols = min(arr.shape[1], target_shape[1])
    result[:min_rows, :min_cols] = arr[:min_rows, :min_cols]
    return result


if __name__ == '__main__':
    img = cv2.imread('../1.png', 1)
    result1 = wavelet_denoise(img)
    # result = cv2.subtract(img,result1)
    cv2.imshow('OTSU', result1)
    ket = cv2.waitKey(0)
