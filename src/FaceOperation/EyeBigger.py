import dlib
import cv2
import numpy as np
import math
import os
def face_Pointer_detect(img_src):
    """
    使用 dlib 检测人脸并提取 68 个面部关键点坐标。

    参数:
        img_src (np.ndarray): 输入的彩色图像 (BGR)

    返回:
        land_marks (list): 包含所有检测到人脸的关键点坐标的列表，
                           每个元素是 68 个关键点的 NumPy 矩阵
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    predictor_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    land_marks = []

    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])
        land_marks.append(land_marks_node)

    return land_marks



def getEllipseCross(p1x, p1y, p2x, p2y, a, b, centerX, centerY):
    """
    计算直线与指定椭圆的交点之一（离参考点较近的那个）。

    参数:
        p1x, p1y: 直线上的第一个点
        p2x, p2y: 直线上的第二个点
        a, b: 椭圆长轴和短轴
        centerX, centerY: 椭圆中心点

    返回:
        list: 交点坐标 (x, y)
    """
    k = (p1y - p2y) / (p1x - p2x)
    m = p1y - k * p1x
    A = b * b + a * a * k * k
    B = 2 * a * a * k * m
    C = a * a * (m * m - b * b)

    X1 = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
    X2 = (-B - math.sqrt(B * B - 4 * A * C)) / (2 * A)

    Y1 = k * X1 + m
    Y2 = k * X2 + m

    if getDis(p2x, p2y, X1, Y1) < getDis(p2x, p2y, X2, Y2):
        resx = X1
        resy = Y1
    else:
        resx = X2
        resy = Y2

    return [resx + centerX, resy + centerY]



def getLinearEquation(p1x, p1y, p2x, p2y):
    """
    计算两点所决定的直线的一般式方程 Ax + By + C = 0

    参数:
        p1x, p1y: 第一个点
        p2x, p2y: 第二个点

    返回:
        list: [A, B, C]
    """
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]


def getDis(p1x, p1y, p2x, p2y):
    """
    计算两个点之间的欧氏距离

    参数:
        p1x, p1y: 第一个点
        p2x, p2y: 第二个点

    返回:
        float: 两点之间距离
    """
    return math.sqrt((p1x - p2x) ** 2 + (p1y - p2y) ** 2)


def get_line_cross_point(p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y):
    """
    计算两条直线的交点

    参数:
        p1x, p1y: 第一条直线端点1
        p2x, p2y: 第一条直线端点2
        p3x, p3y: 第二条直线端点1
        p4x, p4y: 第二条直线端点2

    返回:
        tuple: 交点坐标 (x, y)，若平行则返回 None
    """
    a0, b0, c0 = getLinearEquation(p1x, p1y, p2x, p2y)
    a1, b1, c1 = getLinearEquation(p3x, p3y, p4x, p4y)

    D = a0 * b1 - a1 * b0
    if D == 0:
        return None
    x = (b0 * c1 - b1 * c0) / D
    y = (a1 * c0 - a0 * c1) / D
    return x, y


def localTranslationWarp(srcImg, startIndex, endIndex, Strength, landmarks_node):
    """
    对图像进行局部变形，模拟眼球放大效果

    参数:
        srcImg: 输入图像
        startIndex, endIndex: 眼睛关键点索引范围（如左眼为 36~41）
        Strength: 变形强度（百分比）
        landmarks_node: 当前人脸的 68 个关键点数据

    返回:
        np.ndarray: 处理后的图像
    """
    midIndex = (startIndex + endIndex + 1) >> 1

    Eye = []
    for i in range(startIndex, endIndex+1):
        Eye.append([landmarks_node[i][0, 0], landmarks_node[i][0, 1]])
    ellipseEye = cv2.fitEllipse(np.array(Eye))

    list = []

    for i in range(0,3):

        tmplist = get_line_cross_point(
                        landmarks_node[startIndex + i][0, 0], landmarks_node[startIndex + i][0, 1],
                        landmarks_node[midIndex + i][0, 0], landmarks_node[midIndex + i][0, 1],
                        landmarks_node[startIndex + ((i + 1) % 3)][0, 0], landmarks_node[startIndex + ((i + 1) % 3)][0, 1],
                        landmarks_node[midIndex + ((i + 1) % 3)][0, 0], landmarks_node[midIndex + ((i + 1) % 3)][0, 1]
                  )
        list.append(tmplist)

    width, height, cou = srcImg.shape
    copyImg = srcImg.copy()

    K0 = Strength / 100.0

    centerX = ellipseEye[0][0]
    centerY = ellipseEye[0][1]
    ellipseA = ellipseEye[1][1]
    ellipseB = ellipseEye[1][0]
    ellipseC = math.sqrt(ellipseA * ellipseA - ellipseB * ellipseB)

    for i in range(width):
        for j in range(height):
            if getDis(i, j, centerX - ellipseC, centerY) + getDis(i, j, centerX + ellipseC, centerY) > 2 * ellipseA:
                continue

            [crossX, crossY] = getEllipseCross(0, 0, i - ellipseEye[0][0], j - ellipseEye[0][1], ellipseEye[1][1],
                                               ellipseEye[1][0], ellipseEye[0][0], ellipseEye[0][1])

            radius = getDis(centerX, centerY, crossX, crossY)
            ddradius = radius * radius
            distance = (i - centerX) * (i - centerX) + (j - centerY) * (j - centerY)
            K1 = 1.0 - (1.0 - distance / ddradius) * K0

            UX = (i - centerX) * K1 + centerX
            UY = (j - centerY) * K1 + centerY

            value = BilinearInsert(srcImg, UX, UY)
            copyImg[j, i] = value

    return copyImg



# 双线性插值法
def BilinearInsert(src, ux, uy):
    """
    实现双线性插值法，获取非整数坐标处的颜色值

    参数:
        src: 原始图像
        ux, uy: 插值位置（可以是浮点）

    返回:
        np.ndarray: 插值得到的颜色值
    """
    w, h, c = src.shape
    if c == 3:
        x1 = int(ux)
        x2 = x1 + 1
        y1 = int(uy)
        y2 = y1 + 1

        part1 = src[y1, x1].astype(np.float32) * (float(x2) - ux) * (float(y2) - uy)
        part2 = src[y1, x2].astype(np.float32) * (ux - float(x1)) * (float(y2) - uy)
        part3 = src[y2, x1].astype(np.float32) * (float(x2) - ux) * (uy - float(y1))
        part4 = src[y2, x2].astype(np.float32) * (ux - float(x1)) * (uy - float(y1))

        insertValue = part1 + part2 + part3 + part4

        return insertValue.astype(np.int8)



def eye_bigger_auto(src, LStrength=5, RStrength=5):
    """
    主函数，执行双眼放大操作

    参数:
        src: 输入图像
        LStrength: 左眼放大强度
        RStrength: 右眼放大强度

    返回:
        np.ndarray: 放大后的眼睛图像
    """
    try:
        landmarks = face_Pointer_detect(src)

        if len(landmarks) == 0:
            return

        for landmarks_node in landmarks:
            bigEyeImage = localTranslationWarp(src,36,41,LStrength,landmarks_node)
            bigEyeImage = localTranslationWarp(bigEyeImage,42,47,RStrength,landmarks_node)

        return bigEyeImage
    except:
        raise Exception("未检测到人脸")



if __name__ == '__main__':
    img = cv2.imread('../1.png', 1)


    result = eye_bigger_auto(img)
    cv2.imshow('OTSU', result)
    ket = cv2.waitKey(0)



