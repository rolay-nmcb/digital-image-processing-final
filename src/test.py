import cv2
import dlib

img = cv2.imread('./1.png', 0)  # 直接读为灰度图
print(f"Shape: {img.shape}, dtype: {img.dtype}")

detector = dlib.get_frontal_face_detector()
rects = detector(img, 0)  # 如果这里出错，说明是 dlib 本身兼容性问题 测试说明与numpy版本冲突应降低numpy版本
print(rects)
