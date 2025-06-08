import cv2
import numpy as np
import dlib
import math
import os
"""
FaceThinner.py - 基于 dlib 人脸关键点检测的瘦脸算法实现

功能概述：
    - 使用 dlib 的 68 个面部关键点进行人脸定位
    - 实现局部仿射变换瘦脸算法
    - 支持左右脸分别控制强度、范围、中心点等参数

依赖库：
    - OpenCV (cv2)
    - NumPy
    - dlib（需额外安装）
    - shape_predictor_68_face_landmarks.dat（dlib 预训练模型）

使用说明：
    - 请确保 predictor_path 路径正确指向 shape_predictor_68_face_landmarks.dat 文件
    - 主要调用入口：face_thin_auto()
"""

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
predictor_path = os.path.join(current_dir, 'shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def face_Pointer_detect(img):
    """
       使用 dlib 检测图像中的人脸关键点。

       参数:
           img (numpy.ndarray): 输入 BGR 图像

       返回:
           list of numpy.matrix: 检测到的每个人脸的 68 个关键点坐标列表，
                                 每个元素为一个 68x2 的矩阵，表示关键点坐标。
       """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    land_marks = []
    rects = detector(img_gray, 0)

    for i in range(len(rects)):
        land_marks_node = np.matrix([[p.x, p.y] for p in predictor(img_gray, rects[i]).parts()])

        land_marks.append(land_marks_node)

    return land_marks


def BilinearInsert(src, ux, uy):
    """
        双线性插值法获取浮点坐标位置的像素值。

        参数:
            src (numpy.ndarray): 输入图像
            ux (float): 待插值的 x 坐标（浮点数）
            uy (float): 待插值的 y 坐标（浮点数）

        返回:
            numpy.ndarray: 插值得到的像素值（BGR 格式）
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

def localTranslationWarp(srcImg, startX, startY, endX, endY, radius, Strength):
    """
       局部平移形变算法，用于实现瘦脸效果。

       参数:
           srcImg (numpy.ndarray): 输入图像
           startX (int): 形变起始点 x 坐标
           startY (int): 形变起始点 y 坐标
           endX (int): 目标点 x 坐标（瘦脸目标方向）
           endY (int): 目标点 y 坐标
           radius (int): 形变影响半径
           Strength (int): 瘦脸强度（数值越大，效果越明显）

       返回:
           numpy.ndarray: 经过局部形变后的图像
       """
    ddradius = float(radius * radius)
    copyImg = np.zeros(srcImg.shape, np.uint8)
    copyImg = srcImg.copy()

    K0 = 100 / Strength

    ddmc = (endX - startX) * (endX - startX) + (endY - startY) * (endY - startY)  # (m-c)^2
    H, W, C = srcImg.shape
    for i in range(W):
        for j in range(H):
            # 计算该点是否在形变圆的范围之内
            # 优化，第一步，直接判断是会在（startX,startY)的矩阵框中
            if math.fabs(i - startX) > radius and math.fabs(j - startY) > radius:
                continue

            distance = (i - startX) * (i - startX) + (j - startY) * (j - startY)
            K1 = math.sqrt(distance)
            if (distance < ddradius):
                # 计算出（i,j）坐标的原坐标
                # 计算公式中右边平方号里的部分
                ratio = (ddradius - distance) / (ddradius - distance + K0 * ddmc)
                ratio = ratio * ratio

                # 映射原位置
                UX = i - (ratio * (endX - startX) * (1.0 - (K1 / radius)))
                UY = j - (ratio * (endY - startY) * (1.0 - (K1 / radius)))

                # 根据双线性插值法得到UX，UY的值
                value = BilinearInsert(srcImg, UX, UY)
                # 改变当前 i ，j的值
                copyImg[j, i] = value

    return copyImg

def face_thin_auto(src, LStrength=200, RStrength=200, Lcen=5, Rcen=5, Lrad=5, Rrad=5, Center=5):
    """
       自动瘦脸主函数，基于人脸关键点进行局部形变处理。

       参数:
           src (numpy.ndarray): 输入图像
           LStrength (int): 左脸瘦脸强度，默认 100
           RStrength (int): 右脸瘦脸强度，默认 100
           Lcen (int): 左脸瘦脸起始关键点索引（0~67），默认 5
           Rcen (int): 右脸瘦脸起始关键点索引（0~67），默认 5
           Lrad (int): 左脸瘦脸影响范围半径的关键点偏移量，默认 5
           Rrad (int): 右脸瘦脸影响范围半径的关键点偏移量，默认 5
           Center (int): 瘦脸目标点关键点索引（通常为中心点），默认 5

       返回:
           numpy.ndarray: 处理后的瘦脸图像；
                          如果未检测到人脸，返回原图副本或 None（视具体输入而定）

       """
    # src为原图像
    # LStrength，RStrength为左右脸形变强度
    # Lcen,Rcen为左右脸形变中心
    # Lrad,Rrad为形变范围半径
    # Center为形变重点
    try:
        thin_image = np.array(src)
        landmarks = face_Pointer_detect(src)
        # 如果未检测到人脸关键点，就不进行瘦脸
        if len(landmarks) == 0:
            return

        for landmarks_node in landmarks:
            # print(landmarks_node)
            left_landmark = landmarks_node[Lcen]
            left_landmark_down = landmarks_node[Lcen + Lrad]

            right_landmark = landmarks_node[Rcen]
            right_landmark_down = landmarks_node[Rcen + Rrad]

            endPt = landmarks_node[Center]

            # 计算第Lcen个点到第Lcen+Lrad个点的距离作为瘦脸距离
            r_left = math.sqrt(
                (left_landmark[0, 0] - left_landmark_down[0, 0]) * (left_landmark[0, 0] - left_landmark_down[0, 0]) +
                (left_landmark[0, 1] - left_landmark_down[0, 1]) * (left_landmark[0, 1] - left_landmark_down[0, 1]))

            # 计算第Rcen个点到第Rcen+Rrad个点的距离作为瘦脸距离
            r_right = math.sqrt(
                (right_landmark[0, 0] - right_landmark_down[0, 0]) * (
                                right_landmark[0, 0] - right_landmark_down[0, 0]) +
                (right_landmark[0, 1] - right_landmark_down[0, 1]) * (right_landmark[0, 1] - right_landmark_down[0, 1]))

            # 瘦左边脸
            thin_image = localTranslationWarp(src, left_landmark[0, 0], left_landmark[0, 1], endPt[0, 0], endPt[0, 1],
                                                  r_left, LStrength)
            # 瘦右边脸
            thin_image = localTranslationWarp(thin_image, right_landmark[0, 0], right_landmark[0, 1], endPt[0, 0],
                                                  endPt[0, 1], r_right, RStrength)

        return thin_image
    except:
        raise Exception("未检测到人脸")


def face_thin_auto_optimized(src):
    """
    优化后的自动瘦脸主函数。
    """
    try:
        thin_image = np.array(src)
        landmarks = face_Pointer_detect(src)

        if len(landmarks) == 0:
            return src

        for landmarks_node in landmarks:
            # 选择关键点
            left_landmark = landmarks_node[2]  # 左脸颊外侧点
            right_landmark = landmarks_node[14]  # 右脸颊外侧点
            endPt = landmarks_node[8]  # 下巴中心点

            # 设置瘦脸强度和范围
            LStrength = 60
            RStrength = 60
            Lrad = 10
            Rrad = 10

            # 计算瘦脸距离
            left_landmark_down = landmarks_node[2 + Lrad]
            right_landmark_down = landmarks_node[14 + Rrad]

            r_left = math.sqrt(
                (left_landmark[0, 0] - left_landmark_down[0, 0]) ** 2 +
                (left_landmark[0, 1] - left_landmark_down[0, 1]) ** 2
            )

            r_right = math.sqrt(
                (right_landmark[0, 0] - right_landmark_down[0, 0]) ** 2 +
                (right_landmark[0, 1] - right_landmark_down[0, 1]) ** 2
            )

            # 瘦左边脸
            thin_image = localTranslationWarp(src,
                                              left_landmark[0, 0], left_landmark[0, 1],
                                              endPt[0, 0], endPt[0, 1],
                                              r_left, LStrength)

            # 瘦右边脸
            thin_image = localTranslationWarp(thin_image,
                                              right_landmark[0, 0], right_landmark[0, 1],
                                              endPt[0, 0], endPt[0, 1],
                                              r_right, RStrength)

        return thin_image
    except Exception as e:
        print("Error during face thinning:", e)
        return src


if __name__ == '__main__':
    import os

    print("Current working directory:", os.getcwd())

    # 替换为你自己的图片路径
    img_path = r'..\1.png'
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        exit()

    try:
        result = face_thin_auto_optimized(img)
        if result is not None:
            cv2.imshow('Thin Face', result)
            cv2.waitKey(0)
        else:
            print("No face detected or processing failed.")
    except Exception as e:
        print("An error occurred during processing:", e)

# if __name__ == '__main__':
#     import os
#
#     print("Current working directory:", os.getcwd())
#
#     # 替换为你自己的图片路径
#     img_path = r'..\1.png'
#     img = cv2.imread(img_path)
#
#     if img is None:
#         print(f"Error: Unable to load image at {img_path}")
#         exit()
#
#     try:
#         result = face_thin_auto(img)
#         if result is not None:
#             cv2.imshow('Thin Face', result)
#             cv2.waitKey(0)
#         else:
#             print("No face detected or processing failed.")
#     except Exception as e:
#         print("An error occurred during processing:", e)