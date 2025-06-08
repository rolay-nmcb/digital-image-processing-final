# 导入必要的库
import cv2  # OpenCV 图像处理库
import string  # 字符串操作（虽然未使用，但可以保留）
import numpy as np  # 数值计算库
from PyQt5.QtCore import QPoint  # 用于坐标点处理
from PyQt5.QtGui import QImage  # 用于图像显示转换


def cv2_img_to_qimg(cv2_img, rgb_mode=True):
    """
    将 OpenCV 的图像格式转换为 PyQt 可用的 QImage 格式。

    参数:
        cv2_img (numpy.ndarray): 输入的 OpenCV 图像（BGR 格式）
        rgb_mode (bool): 是否将图像从 BGR 转换为 RGB

    返回:
        QImage: 转换后的 QImage 对象
    """
    try:
        if len(cv2_img.shape) == 3:  # 判断是否为彩色图像
            height, width, _ = cv2_img.shape
            bytes_per_line = 3 * width

            if rgb_mode:
                cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

            qImg = QImage(cv2_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:  # 灰度图
            if cv2_img.dtype != np.uint8:
                cv2_img = cv2_img.astype(np.uint8)
            if len(cv2_img.shape) != 2:
                cv2_img = cv2_img.squeeze()
            cv2_img = np.ascontiguousarray(cv2_img)
            height, width = cv2_img.shape
            bytes_per_line = width
            qImg = QImage(cv2_img.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

        # 如果图像为空，尝试修复数据格式
        if qImg.isNull():
            qImg = QImage(cv2_img.data, width, height,
                          bytes_per_line, QImage.Format_RGB888).rgbSwapped().mirrored(False, True)
        return qImg
    except:
        raise Exception("转化图像时出错")


def cv2_Chinese_imread(filepath, type):
    """
    支持中文路径的图像读取函数。

    参数:
        filepath (str): 图像文件路径
        type (str): 'gray' 表示灰度图，'color' 表示彩色图

    返回:
        numpy.ndarray: 读取后的图像数据
    """
    if type == 'gray':
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        return img
    elif type == 'color':
        img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def rotate_angle(src, angle):
    """
    按指定角度旋转图像，并自动裁剪去除黑边。

    参数:
        src (numpy.ndarray): 输入图像
        angle (int): 旋转角度（单位：度）（-180到180）

    返回:
        numpy.ndarray: 旋转并裁剪后的图像
    """
    height, width = src.shape[:2]

    # 计算旋转矩阵并应用旋转
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale=1)
    rotated = cv2.warpAffine(src, M, (width, height), borderValue=(255, 255, 255))

    # 如果不是90度倍数，需要裁剪去除黑边
    if angle % 90 != 0:
        # 确保角度在0-180度之间
        angle_mod = abs(angle) % 180
        if angle_mod > 90:
            angle_mod = 180 - angle_mod

        theta = angle_mod * np.pi / 180
        hw_ratio = float(height) / float(width)
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        r = hw_ratio if height > width else 1 / hw_ratio
        denominator = r * tan_theta + 1
        crop_mult = numerator / denominator

        # 计算裁剪尺寸
        w_crop = int(crop_mult * width)
        h_crop = int(crop_mult * height)

        # 确保裁剪尺寸有效
        if w_crop <= 0 or h_crop <= 0:
            return rotated  # 如果裁剪尺寸无效，返回旋转后的图像

        # 计算裁剪区域
        x0 = max(0, (width - w_crop) // 2)
        y0 = max(0, (height - h_crop) // 2)
        x1 = min(width, x0 + w_crop)
        y1 = min(height, y0 + h_crop)

        # 执行裁剪
        cropped = rotated[y0:y1, x0:x1]

        # 如果裁剪后的图像有效，则调整大小
        if cropped.size > 0:
            result = cv2.resize(cropped, (width, height))
            return result

    return rotated


def scale_by_equal_position(src, size):
    """
    按比例缩放图像。

    参数:
        src (numpy.ndarray): 输入图像
        size (int): 缩放百分比（如 50 表示缩小为原图的一半）

    返回:
        numpy.ndarray: 缩放后的图像
    """
    ratio = size / 100
    result = cv2.resize(src, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
    return result


def Scale_Or_Translate_by_input(src, type, new_width, new_height):
    """
    执行缩放或平移操作。

    参数:
        src (numpy.ndarray): 输入图像
        type (str): 操作类型，'scale' 表示缩放，其他表示平移
        new_width (int): 新宽度或水平平移量
        new_height (int): 新高度或垂直平移量

    返回:
        numpy.ndarray: 处理后的图像
    """
    shape = src.shape
    height = shape[0]
    width = shape[1] if len(shape) > 1 else 1

    if type == 'scale':
        result = cv2.resize(src, (new_width, new_height))
    else:
        M = np.float32([[1, 0, new_width], [0, 1, new_height]])
        result = cv2.warpAffine(src, M, (width, height))
    return result


def filp_by_type(src, code):
    """
    按照指定方式翻转图像。

    参数:
        src (numpy.ndarray): 输入图像
        code (int): 翻转方式：
                    1 水平翻转，
                    0 垂直翻转，
                   -1 水平+垂直翻转

    返回:
        numpy.ndarray: 翻转后的图像
    """
    result = cv2.flip(src, code)
    return result


def crop_by_pos(src, pos1, pos2):
    """
    根据两个坐标点裁剪图像（兼容灰度图和彩色图）。

    参数:
        src (numpy.ndarray): 输入图像
        pos1 (QPoint): 起始点坐标
        pos2 (QPoint): 结束点坐标

    返回:
        numpy.ndarray: 裁剪后的图像区域
    """
    # 确保坐标顺序正确（避免反向切片）
    x1, y1 = min(pos1.x(), pos2.x()), min(pos1.y(), pos2.y())
    x2, y2 = max(pos1.x(), pos2.x()), max(pos1.y(), pos2.y())

    # #打印坐标信息
    # print("裁剪坐标：", x1, y1, x2, y2)

    # 灰度图（2D数组）和彩色图（3D数组）的兼容处理
    if len(src.shape) == 2:
        cropped = src[y1:y2, x1:x2]
    else:
        cropped = src[y1:y2, x1:x2, :]
    return cropped


if __name__ == '__main__':
    """
    主程序入口，用于测试 crop_by_pos 函数
    """
    img = cv2.imread('../1.png', 1)  # 加载测试图像
    src = rotate_angle(img, -43)
    # 测试裁剪功能
    cv2.imshow('OTSU', src)
    ket = cv2.waitKey(0)
