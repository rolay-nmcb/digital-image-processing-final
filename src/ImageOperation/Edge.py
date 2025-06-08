import cv2

import numpy as np

def threshold(img,code):
    """
       使用 OpenCV 的全局阈值方法对图像进行二值化处理。

       参数:
           img (numpy.ndarray): 输入图像（灰度或BGR格式）
           code (str): 阈值方法类型，可选 'TRIANGLE'、'OTSU'、'TRUNC'、'TOZERO'

       返回:
           numpy.ndarray: 二值化后的图像（单通道）
       """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if code=='TRIANGLE':
        ret, result = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
    elif code == 'OTSU':
        ret, result = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    elif code == 'TRUNC':
        ret, result = cv2.threshold(gray, 200, 255, cv2.THRESH_TRUNC)
    elif code == 'TOZERO':
        ret, result = cv2.threshold(gray, 100, 255, cv2.THRESH_TOZERO)
    return result


def roberts_kernel(img):
    """
       使用 Roberts 算子进行边缘检测（基于2x2梯度核）。

       参数:
           img (numpy.ndarray): 输入图像（灰度或BGR格式）

       返回:
           numpy.ndarray: 边缘检测结果（与输入图像格式一致）
       """
    # 检查图像是否为彩色（3通道）
    if len(img.shape) == 3 and img.shape[2] == 3:
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayImage = img  # 已经是灰度图

    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 根据输入类型决定是否转回BGR
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        return result


def sobel_kernel(img):
    """
       使用 Sobel 算子进行边缘检测（支持 x 和 y 方向梯度计算）。

       参数:
           img (numpy.ndarray): 输入图像（灰度或BGR格式）

       返回:
           numpy.ndarray: 边缘检测结果（与输入图像格式一致）
       """
    # 判断输入图像是否为彩色
    if len(img.shape) == 3 and img.shape[2] == 3:
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayImage = img  # 输入已经是灰度图

    # 计算Sobel梯度
    x = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)

    # 转换为8位无符号整数
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)

    # 合并梯度
    result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 根据输入类型决定输出格式
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 彩色输入返回BGR
    else:
        return result  # 灰度输入直接返回灰度结果


def laplacian_kernel(img):
    """
       使用拉普拉斯算子进行边缘检测，并使用高斯模糊降噪。

       参数:
           img (numpy.ndarray): 输入图像（灰度或BGR格式）

       返回:
           numpy.ndarray: 边缘检测结果（与输入图像格式一致）
       """
    # 判断输入图像是否为彩色
    if len(img.shape) == 3 and img.shape[2] == 3:
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        grayImage = img  # 输入已经是灰度图

    # 高斯模糊降噪
    grayImage = cv2.GaussianBlur(grayImage, (5, 5), 0, 0)

    # 计算拉普拉斯算子
    laplacian = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)

    # 转换为8位无符号整数
    result = cv2.convertScaleAbs(laplacian)

    # 根据输入类型决定输出格式
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  # 彩色输入返回BGR
    else:
        return result  # 灰度输入直接返回灰度结果


def log_kernel(img):
    """
       使用 LoG（Laplacian of Gaussian）算子进行边缘检测。

       参数:
           img (numpy.ndarray): 输入图像（灰度或BGR格式）

       返回:
           numpy.ndarray: 边缘检测结果（与输入图像格式一致）
       """
    # 输入检查：确保是单通道或三通道图像
    if len(img.shape) not in [2, 3]:
        raise ValueError("Input must be a grayscale or BGR image.")

    # 边界填充（填充2像素，使用BORDER_REPLICATE模式）
    image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)

    # 高斯模糊（3x3核，标准差自动计算）
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # 5x5 LoG算子
    log = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ])

    # 初始化输出图像
    rows, cols = image.shape[:2]
    if len(image.shape) == 3:  # 多通道（BGR）
        image1 = np.zeros_like(image, dtype=np.float32)
        for k in range(3):  # 处理所有通道
            for i in range(2, rows - 2):
                for j in range(2, cols - 2):
                    image1[i, j, k] = np.sum(log * image[i - 2:i + 3, j - 2:j + 3, k])
    else:  # 单通道（灰度）
        image1 = np.zeros((rows, cols), dtype=np.float32)
        for i in range(2, rows - 2):
            for j in range(2, cols - 2):
                image1[i, j] = np.sum(log * image[i - 2:i + 3, j - 2:j + 3])

    # 转换为8位无符号整型（绝对值+截断）
    kernel = cv2.convertScaleAbs(image1)
    return kernel


def canny_kernel(img, low_threshold=50, high_threshold=150):
    """
    Canny边缘检测（兼容灰度和彩色输入）

    参数:
        img: 输入图像（灰度或BGR）
        low_threshold: Canny低阈值
        high_threshold: Canny高阈值

    返回:
        edges: 边缘检测结果（BGR格式，与输入一致）
    """
    # 检查输入是否为单通道（灰度）
    if len(img.shape) == 2:
        gray = img  # 直接使用灰度图像
        is_gray = True
    else:
        # 彩色图像转灰度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_gray = False

    # 高斯模糊去噪
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # 计算Sobel梯度
    grad_x = cv2.Sobel(blur, cv2.CV_16SC1, 1, 0)  # x方向梯度
    grad_y = cv2.Sobel(blur, cv2.CV_16SC1, 0, 1)  # y方向梯度

    # Canny边缘检测
    edges = cv2.Canny(grad_x, grad_y, low_threshold, high_threshold)

    # 根据输入决定输出格式
    if is_gray:
        return edges  # 返回单通道灰度结果
    else:
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # 转BGR三通道

def HoughLines(img):
    """
       使用霍夫变换检测图像中的直线（HoughLines）。

       参数:
           img (numpy.ndarray): 输入图像（通常为边缘图）

       返回:
           numpy.ndarray: 绘制了检测到直线的图像
       """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 200, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 30, 100)
    result = img.copy()

    for i_line in lines:
        for line in i_line:
            rho = line[0]
            theta = line[1]
            a = np.cos(theta)
            # 存储sin(theta)的值
            b = np.sin(theta)
            # 存储rcos(theta)的值
            x0 = a * rho
            # 存储rsin(theta)的值
            y0 = b * rho
            # 存储(rcos(theta)-1000sin(theta))的值
            x1 = int(x0 + 1000 * (-b))
            # 存储(rsin(theta)+1000cos(theta))的值
            y1 = int(y0 + 1000 * (a))
            # 存储(rcos(theta)+1000sin(theta))的值
            x2 = int(x0 - 1000 * (-b))
            # 存储(rsin(theta)-1000cos(theta))的值
            y2 = int(y0 - 1000 * (a))
            # 绘制直线结果
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 1)

    return result

def HoughLinesP(img,threshold=80,minLineLength=30,maxLineGap=10):
    """
       使用概率霍夫变换检测图像中的线段（HoughLinesP）。

       参数:
           img (numpy.ndarray): 输入图像（通常为边缘图）
           threshold (int): 累加器阈值，用于识别线段
           minLineLength (int): 可接受线段的最小长度
           maxLineGap (int): 同一线段上点之间的最大间隙

       返回:
           numpy.ndarray: 绘制了检测到线段的图像
       """
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edges = cv2.Canny(img, 50, 200, apertureSize=3)

    linesP = cv2.HoughLinesP(edges, rho = 1, theta = np.pi/180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap)
    result = img.copy()
    for i_P in linesP:
        for x1, y1, x2, y2 in i_P:
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return result

def LSD_Line(src):
    """
       使用 LSD（Line Segment Detector）算法检测图像中的线段。

       参数:
           src (numpy.ndarray): 输入图像（BGR格式）

       返回:
           numpy.ndarray: 检测并绘制线段后的图像（BGR格式）
       """

    img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    # 执行检测结果
    dlines = lsd.detect(img)
    # 绘制检测结果
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        cv2.line(src, (x0, y0), (x1, y1), (0, 0, 0), 1, cv2.LINE_AA)
    return src

if __name__ == '__main__':
    img = cv2.imread('../2.png', 1)
    src = canny_kernel(img);
    cv2.imshow('OTSU', src)
    ket = cv2.waitKey(0)