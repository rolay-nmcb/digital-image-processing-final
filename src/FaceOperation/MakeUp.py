import cv2
import numpy as np
import dlib
import face_recognition
from PIL import Image,ImageDraw
import os

"""
MakeUp.py - 基于人脸关键点的自动美妆实现

功能概述：
    - 使用 face_recognition 获取人脸关键点（68 点）
    - 对眉毛、嘴唇、眼睛等区域进行美妆绘制
    - 支持透明度叠加效果（RGBA），避免破坏原图纹理
    - 返回 BGR 图像以便 OpenCV 显示或保存

依赖库：
    - OpenCV (cv2)
    - NumPy
    - face_recognition（底层使用 dlib）
    - PIL（用于图像绘图）

使用说明：
    - 请确保已安装 face_recognition 和 Pillow 库
    - 主要调用入口：Make_Up()
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


def Make_Up(img):
    """
       自动为人脸添加虚拟美妆效果，包括眉毛、唇彩、眼线等。

       参数:
           img (numpy.ndarray): 输入 BGR 格式的图像

       返回:
           numpy.ndarray: 添加美妆后的图像，BGR 格式

       绘制内容:
           - 眉毛：深棕色填充
           - 嘴唇：红色填充 + 较浅红色描边
           - 眼睛：白色轮廓线
           - 眼线：黑色描边，增强眼部立体感

       """
    try:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        face_landmarks_list = face_recognition.face_landmarks(img)
        for face_landmarks in face_landmarks_list:
            pil_image = Image.fromarray(img)
            drawer = ImageDraw.Draw(pil_image,'RGBA')
            # 绘制眉毛
            drawer.polygon(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 128))
            drawer.polygon(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 128))
            drawer.line(face_landmarks['left_eyebrow'], fill=(68, 54, 39, 150), width=5)
            drawer.line(face_landmarks['right_eyebrow'], fill=(68, 54, 39, 150), width=5)

            # 绘制嘴唇（深红色填充 + 浅红色描边）
            drawer.polygon(face_landmarks['top_lip'], fill=(0, 0, 120, 128))  # 深红色填充
            drawer.polygon(face_landmarks['bottom_lip'], fill=(0, 0, 120, 128))
            drawer.line(face_landmarks['top_lip'], fill=(50, 50, 150, 100), width=6)  # 浅红色描边
            drawer.line(face_landmarks['bottom_lip'], fill=(50, 50, 150, 100), width=6)

            # 绘制眼睛
            drawer.line(face_landmarks['left_eye'], fill=(255, 255, 255, 30))
            drawer.line(face_landmarks['right_eye'], fill=(255, 255, 255, 30))

            # 绘制眼线
            drawer.line(
                face_landmarks['left_eye'] + [face_landmarks['left_eye'][0]],
                fill=(0, 0, 0, 110),
                width=4)
            drawer.line(
                face_landmarks['right_eye'] + [face_landmarks['right_eye'][0]],
                fill=(0, 0, 0, 110),
                width=4)
        result = np.array(pil_image)
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # return result
    except:
        raise Exception("未检测到人脸")


if __name__ == '__main__':
    """
       测试用例：加载本地图片并应用美妆算法

       注意：
           - 替换路径为你自己的测试图片路径
           - 可扩展支持参数调整、多图处理等功能
       """
    img = cv2.imread('D:\\PythonWork\\digital-image-final\\src\\1.png', 1)

    result = Make_Up(img)

    cv2.imshow('OTSU', result)
    ket = cv2.waitKey(0)