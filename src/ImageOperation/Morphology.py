# 导入必要的库
import cv2  # OpenCV 库，用于图像处理
import numpy as np  # 数值计算库，支持多维数组操作


def get_kernel(size, type='rect'):
    """
    根据指定类型和大小生成结构元素（卷积核）

    参数:
        size (int): 结构元素的尺寸（例如 3 表示 3x3 的核）
        type (str): 结构元素类型，可选 'rect'（矩形）、'cross'（十字形）、'ellipse'（椭圆形）

    返回:
        kernel (numpy.ndarray): 生成的结构元素
    """
    if type == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif type == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
    elif type == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return kernel


def Erosion_Or_Dilation(src, type, code):
    """
    执行膨胀或腐蚀操作

    参数:
        src (numpy.ndarray): 输入图像
        type (str): 结构元素类型，可选 'rect'、'cross'、'ellipse'
        code (str): 操作类型，'erode' 表示腐蚀，其他则执行膨胀

    返回:
        result (numpy.ndarray): 膨胀或腐蚀后的图像
    """
    kernel = get_kernel(3, type)  # 默认使用 3x3 的结构元素
    result = src.copy()  # 避免修改原始图像
    if code == 'erode':
        return cv2.erode(result, kernel=kernel)  # 腐蚀操作
    else:
        return cv2.dilate(result, kernel=kernel)  # 膨胀操作


def Open_Or_Close(src, type, code):
    """
    执行开运算或闭运算操作

    参数:
        src (numpy.ndarray): 输入图像
        type (str): 结构元素类型，可选 'rect'、'cross'、'ellipse'
        code (str): 操作类型，'open' 表示开运算，其他则执行闭运算

    返回:
        result (numpy.ndarray): 开运算或闭运算后的图像
    """
    # 确定结构元素类型对应的OpenCV常量
    if type == 'rect':
        kernel_type = cv2.MORPH_RECT
    elif type == 'cross':
        kernel_type = cv2.MORPH_CROSS
    elif type == 'ellipse':
        kernel_type = cv2.MORPH_ELLIPSE
    else:
        # 默认使用矩形结构元素
        kernel_type = cv2.MORPH_RECT

    # 创建3×3的结构元素
    kernel = cv2.getStructuringElement(kernel_type, (3, 3))

    # 根据code参数选择操作类型
    if code == 'open':
        # 开运算：先腐蚀后膨胀
        result = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel)
    else:
        # 闭运算：先膨胀后腐蚀
        result = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel)

    return result


if __name__ == '__main__':
    """
    主程序入口，用于测试形态学操作函数
    """
    img = cv2.imread('../1.png', 1)  # 加载彩色图像
    result1 = Open_Or_Close(img, 'rect', 'close')  # 对图像进行运算

    cv2.imshow('test3', result1)  # 显示结果图像
    ket = cv2.waitKey(0)  # 等待按键按下，关闭窗口
