# 导入系统模块
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import *

# 导入图像处理相关操作模块
from FaceOperation.EyeBigger import *
from FaceOperation.skinWhiten import *
from FaceOperation.FaceThinner import *
from FaceOperation.MakeUp import *
from ImageOperation.BasicOperation import *
from ImageOperation.Edge import *
from ImageOperation.Morphology import *
from ImageOperation.Restoration import *
from ImageOperation.Sharpen import *
from ImageOperation.Smooth import *
from ImageOperation.Split import *
from ImageOperation.Strength import *
from neural_style.start import *

import sys
import os

# 导入 UI 模块（由 Designer 生成）
from ui import Ui_ImageProcess

"""
main.py - 图像处理软件主界面与控制中心

提供完整的数字图像处理流程：
- 支持图像基本操作（旋转、翻转、裁剪、缩放、平移）
- 图像增强（亮度、对比度、伽马校正、直方图均衡,色相）
- 频域滤波（理想/巴特沃斯/高斯低通/高通滤波）
- 锐化、边缘检测、霍夫变换
- 形态学操作（腐蚀、膨胀、开闭运算）
- 分割算法（区域生长、K-Means、分水岭）
- 人像美化（大眼、瘦脸、美白、磨皮、上妆）
- 图像的加噪与去噪
- 图像压缩(基于小波变换)
- 风格迁移
- 支持中文路径图像读写

依赖模块：
- PyQt5: GUI 构建
- OpenCV: 图像处理核心
- 自定义模块：FaceOperation.*, ImageOperation.*,neural_style.*
"""

# 定义主界面类 myUI，继承自 QMainWindow 和 Ui_ImageProcess
class myUI(QMainWindow, Ui_ImageProcess):
    """
       主程序界面类，集成 PyQt5 界面与图像处理逻辑。

       功能概述：
           - 支持多种图像操作：旋转、翻转、裁剪、缩放、平移
           - 图像增强：亮度/对比度/饱和度调整、Gamma 校正、直方图均衡化
           - 滤波器：低通、高通、中值滤波等
           - 边缘检测、形态学操作、分割算法（区域生长、K-means、分水岭）
           - 人像处理：大眼、瘦脸、美白、磨皮、上妆
           - 支持中文路径的图像加载与保存

       成员变量:
           img_original (numpy.ndarray): 原始图像数据
           img_process (numpy.ndarray): 当前处理中的图像数据
           slider (QSlider): 参数调节滑块控件
           original_width / height (int): 原始图像尺寸
       """

    def __init__(self, parent=None):
        """
               初始化主窗口与图像处理环境。
               """
        super(myUI, self).__init__(parent)
        # 初始化 UI 界面
        self.setupUi(self)
        # 初始化信号与槽连接
        self.trigger_slot_init()

        # 初始化图像变量
        self.img_original = None  # 原始图像
        self.img_process = None   # 当前正在处理的图像
        self.slider = None        # 滑块控件，用于交互式调整参数
        self.update_button=None
        self.value1=None
        self.value2=None
        self.value3=None
        self.value4=None

        # 保存原始图像尺寸
        self.original_width = None
        self.original_height = None

        #是否连续操作
        self.is_continuous = False

    # 初始化所有按钮点击事件对应的函数
    def trigger_slot_init(self):
        """
        绑定所有菜单项与对应图像处理函数。
        """
        # 文件操作
        self.OpenGrayImg.triggered.connect(lambda: self.loadOriginalImage('gray'))     # 打开灰度图
        self.OpenColorImg.triggered.connect(lambda: self.loadOriginalImage('color'))   # 打开彩色图
        self.CloseImage.triggered.connect(self.loadModifiedImage)    # 保存修改后的图像
        self.setContinuousFlag.triggered.connect(self.set_continuous_flag) #是否连续操作

        # 图像旋转
        self.Rotate90.triggered.connect(lambda: self.simple_operation('rotate', 90))
        self.Rotate180.triggered.connect(lambda: self.simple_operation('rotate', 180))
        self.Rotate270.triggered.connect(lambda: self.simple_operation('rotate', 270))
        self.RotateFree.triggered.connect(lambda: self.add_slider('rotate'))

        # 图像翻转
        self.HorizontalFlip.triggered.connect(lambda: self.simple_operation('flip', 1))  # 水平翻转
        self.VerticalFlip.triggered.connect(lambda: self.simple_operation('flip', 0))    # 垂直翻转
        self.DualFlip.triggered.connect(lambda: self.simple_operation('flip', -1))       # 双向翻转

        # 裁剪
        self.Crop.triggered.connect(self.my_crop)
        self.setFlag.triggered.connect(self.showInformation)

        # 放缩
        self.scaleByEqualPosition.triggered.connect(lambda: self.add_slider('scale'))
        self.scaleByInput.triggered.connect(lambda: self.Operation_by_Input('scale'))

        # 平移
        self.Translate.triggered.connect(lambda: self.Operation_by_Input('translate'))

        # 亮度、对比度、饱和度等滑动调节
        self.Brightness.triggered.connect(lambda: self.add_slider('bright'))
        self.Contrast.triggered.connect(lambda: self.add_slider('contrast'))
        self.Saturation.triggered.connect(lambda: self.add_slider('saturation'))
        self.Exposure.triggered.connect(lambda: self.add_slider('exposure'))
        self.Value.triggered.connect(lambda: self.add_slider('value'))
        self.Hue.triggered.connect(lambda: self.add_slider('hue'))

        # 直方图均衡化
        self.globalHist.triggered.connect(lambda: self.simple_operation('globalHist', 0))         # 全局直方图均衡化
        self.adjustiveHist.triggered.connect(lambda: self.simple_operation('adjustiveHist', 0))  # 自适应直方图均衡化

        # 图像增强
        self.logStrength.triggered.connect(lambda: self.simple_operation('log', 0))               # 对数变换
        self.linearStrength.triggered.connect(lambda: self.Operation_by_Input('linear'))          # 线性增强
        self.gammaStrength.triggered.connect(lambda: self.add_slider('gamma'))                    # Gamma 校正
        self.actiongrayHist.triggered.connect(lambda: self.simple_operation('hist', 0))           # 直方图显示
        self.binaryLayered.triggered.connect(lambda: self.Operation_by_Input('binary_layered'))   # 二值化
        self.scaleLayered.triggered.connect(lambda: self.Operation_by_Input('scale_layered'))     # 分层缩放

        # 滤波器
        self.idealLowPass.triggered.connect(lambda: self.Operation_by_Input('idealLowPass'))      # 理想低通滤波
        self.gaussLowPass.triggered.connect(lambda: self.Operation_by_Input('gaussLowPass'))      # 高斯低通滤波
        self.butterworthLowPass.triggered.connect(lambda: self.Operation_by_Input('butterworthLowPass'))  # 巴特沃斯低通滤波
        self.MedianLowPass.triggered.connect(lambda: self.simple_operation('MedianLowPass', 0))   # 中值滤波
        self.BilaLowPass.triggered.connect(lambda: self.simple_operation('BilaLowPass', 0))       # 双边滤波
        self.MeanShiftLowPass.triggered.connect(lambda: self.simple_operation('MeanShiftLowPass', 0))  # 均值漂移滤波

        # 高通滤波器
        self.idealHighPass.triggered.connect(lambda: self.Operation_by_Input('idealHighPass'))
        self.gaussHighPass.triggered.connect(lambda: self.Operation_by_Input('gaussHighPass'))
        self.butterworthHighPass.triggered.connect(lambda: self.Operation_by_Input('butterworthHighPass'))

        # 锐化
        self.roberts_sharpen.triggered.connect(lambda: self.simple_operation('robels_sharpen', 0))
        self.sobel_sharpen.triggered.connect(lambda: self.simple_operation('sobel_sharpen', 0))
        self.laplacian_sharpen.triggered.connect(lambda: self.simple_operation('laplacian_sharpen', 0))
        self.scharr_sharpen.triggered.connect(lambda: self.simple_operation('scharr_sharpen', 0))

        # 噪声与恢复
        self.Gauss_Noise.triggered.connect(lambda: self.simple_operation('Gauss_Noise', 0))
        self.Rayleigh_Noise.triggered.connect(lambda: self.simple_operation('Rayleigh_Noise', 0))
        self.Ireland_Noise.triggered.connect(lambda: self.simple_operation('Ireland_Noise', 0))
        self.Exponential_Noise.triggered.connect(lambda: self.simple_operation('Exponential_Noise', 0))
        self.Uniform_Noise.triggered.connect(lambda: self.simple_operation('Uniform_Noise', 0))
        self.SaltPepper_Noise.triggered.connect(lambda: self.simple_operation('SaltPepper_Noise', 0))

        # 统计滤波器
        self.Arithmentic_Mean_Filter.triggered.connect(lambda: self.simple_operation('Arithmetic_Mean_Filter', 0))
        self.Geometric_Mean_Filter.triggered.connect(lambda: self.simple_operation('Geometric_Mean_Filter', 0))
        self.Harmonic_Mean_Filter.triggered.connect(lambda: self.simple_operation('Harmonic_Mean_Filter', 0))
        self.Max_Filter.triggered.connect(lambda: self.simple_operation('Max_Filter', 0))
        self.Min_Filter.triggered.connect(lambda: self.simple_operation('Min_Filter', 0))
        self.Middle_Filter.triggered.connect(lambda: self.simple_operation('Middle_Filter', 0))
        self.Midian_Filter.triggered.connect(lambda: self.simple_operation('Median_Filter', 0))
        #小波恢复
        self.wavelet_denoise.triggered.connect(lambda: self.simple_operation('wavelet_denoise',  0))


        #图像压缩
        self.wavelet_compress.triggered.connect(lambda: self.Operation_by_Input('wavelet_compress'))

        # 边缘检测
        self.roberts_split.triggered.connect(lambda: self.simple_operation('roberts_split', 0))
        self.sobel_split.triggered.connect(lambda: self.simple_operation('sobel_split', 0))
        self.laplacian_split.triggered.connect(lambda: self.simple_operation('laplacian_split', 0))
        self.log_split.triggered.connect(lambda: self.simple_operation('log_split', 0))
        self.canny_split.triggered.connect(lambda: self.simple_operation('canny_split', 0))
        self.HoughLines.triggered.connect(lambda: self.simple_operation('HoughLines', 0))
        self.HoughLinesP.triggered.connect(lambda: self.simple_operation('HoughLinesP', 0))

        # 区域分割
        self.regionalGrowth.triggered.connect(lambda: self.Operation_by_Input('regionalGrowth'))
        self.regionalSplit.triggered.connect(lambda: self.simple_operation('regionalSplit', 0))
        self.kmeans.triggered.connect(lambda: self.simple_operation('kmeans', 0))
        self.disWaterHold.triggered.connect(lambda: self.simple_operation('disWaterHold', 0))
        self.sobelWaterHold.triggered.connect(lambda: self.simple_operation('sobelWaterHold', 0))


        #阈值处理
        self.globaThreshold.triggered.connect(lambda: self.Operation_by_Input('globalThreshold'))
        self.localThreshold.triggered.connect(lambda: self.simple_operation('localThreshold', 0))
        self.multiThreshold.triggered.connect(lambda: self.Operation_by_Input('multiThreshold'))

        # 形态学操作
        self.Erodision_Rect.triggered.connect(lambda: self.simple_operation('Erosion_Rect', 0))
        self.Erodision_Cross.triggered.connect(lambda: self.simple_operation('Erosion_Cross', 0))
        self.Erodision_Ellipse.triggered.connect(lambda: self.simple_operation('Erosion_Ellipse', 0))
        self.Dilation_Rect.triggered.connect(lambda: self.simple_operation('Dilation_Rect', 0))
        self.Dilation_Cross.triggered.connect(lambda: self.simple_operation('Dilation_Cross', 0))
        self.Dilation_Ellipse.triggered.connect(lambda: self.simple_operation('Dilation_Ellipse', 0))
        self.Open_Rect.triggered.connect(lambda: self.simple_operation('Open_Rect', 0))
        self.Open_Cross.triggered.connect(lambda: self.simple_operation('Open_Cross', 0))
        self.Open_Ellipse.triggered.connect(lambda: self.simple_operation('Open_Ellipse', 0))
        self.Close_Rect.triggered.connect(lambda: self.simple_operation('Close_Rect', 0))
        self.Close_Cross.triggered.connect(lambda: self.simple_operation('Close_Cross', 0))
        self.Close_Ellipse.triggered.connect(lambda: self.simple_operation('Close_Ellipse', 0))

        # 人像处理
        self.skin_whiten.triggered.connect(lambda: self.add_slider('whiten'))            # 美白
        self.skin_smooth.triggered.connect(lambda: self.simple_operation('smooth', 0))  # 磨皮
        self.eye_bigger.triggered.connect(lambda: self.Operation_by_Input('bigger'))     # 大眼
        self.face_thinner.triggered.connect(lambda: self.simple_operation('thinner', 0)) # 瘦脸
        self.make_up.triggered.connect(lambda: self.simple_operation('make_up', 0))     # 上妆


        #风格迁移
        self.candy.triggered.connect(lambda: self.simple_operation('candy', 0))
        self.mosaic.triggered.connect(lambda: self.simple_operation('mosaic', 0))
        self.rain_princess.triggered.connect(lambda: self.simple_operation('rain_princess', 0))
        self.udnie.triggered.connect(lambda: self.simple_operation('udnie', 0))
        self.starry_night.triggered.connect(lambda: self.simple_operation('starry_night', 0))
        self.picasso.triggered.connect(lambda: self.simple_operation('picasso', 0))
        self.cuphead.triggered.connect(lambda: self.simple_operation('cuphead', 0))
        self.JoJo.triggered.connect(lambda: self.simple_operation('JoJo', 0))
        self.fu_shi_hui.triggered.connect(lambda: self.simple_operation('fu_shi_hui', 0))
        self.anime.triggered.connect(lambda: self.simple_operation('anime', 0))
        self.mc.triggered.connect(lambda: self.simple_operation('mc', 0))

        # 重置图像
        self.ResetAction.triggered.connect(self.reset_image)

    def my_crop(self):
        """
        启动图像裁剪功能。

        功能步骤：
            1. 显示或隐藏图像辅助信息（如坐标提示）
            2. 绑定 QLabel 的 cropPointsSelected 信号到 crop_requested 函数，
               当用户选择完矩形区域后会调用该回调函数进行裁剪
        """
        self.clear_controls()

        if self.img_original is None:
            QMessageBox.critical(self, "错误", f"你没有选择图片")
            return

            # 先断开可能存在的旧连接，避免重复绑定
        try:
            self.label_original.cropPointsSelected.disconnect(self.crop_requested)
        except TypeError:
            pass  # 如果没有绑定过，disconnect 会抛出异常，忽略即可

        self.showInformation()  # 切换显示/隐藏图像上的辅助信息
        self.label_original.cropPointsSelected.connect(self.crop_requested)  # 连接裁剪信号

    def crop_requested(self, start_point, end_point):
        """
        图像裁剪执行函数。在用户选中一个矩形区域后被触发。

        参数:
            start_point (tuple): 裁剪起始点坐标 (x, y)
            end_point (tuple): 裁剪结束点坐标 (x, y)
        """
        try:
            #每次裁剪之前先对剪的图像重置
            self.reset_image()

            # 调用图像处理函数对图像进行裁剪
            self.img_process = crop_by_pos(self.img_process, start_point, end_point)

            self.update_img(self.img_process)  # 更新图像显示

            self.showInformation()  # 再次切换辅助信息显示状态（关闭）

            # 隐藏 tooltip 提示框
            if hasattr(self.label_original, 'tooltip_widget'):
                self.label_original.tooltip_widget.hide_tooltip()

            # 恢复 QLabel 中 pixmap 的原始显示内容
            self.label_original.setPixmap(self.label_original.storage)
        except Exception as e:
            # 使用 QMessageBox 显示错误信息
            QMessageBox.critical(self, "错误", f"图像处理操作失败：\n{str(e)}")


    def loadOriginalImage(self, type):
        """
        打开文件对话框让用户选择并加载原始图像。

        参数:
            type (str): 图像读取类型，'gray' 表示灰度图，'color' 表示彩色图
        """
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "选择原始图片",
            "",
            "Images (*.png *.xpm *.jpg)",
            options=options
        )

        if fileName:
            # 加载图像到两个 QLabel 控件中（原图和处理图）
            self.loadImage(fileName, self.label_original)
            self.loadImage(fileName, self.label_modified)

            # 使用自定义函数读取支持中文路径的图像
            self.img_original = cv2_Chinese_imread(fileName, type)
            self.img_process = cv2_Chinese_imread(fileName, type)

            # 记录原始图像尺寸，用于后续重置操作
            height, width = self.img_original.shape[:2]
            self.original_width = width
            self.original_height = height

    def showInformation(self):
        """
        控制是否显示图像上的辅助信息（如坐标、提示等）。

        原理：
            - flag 是 QLabel 自定义属性，表示当前信息是否处于显示状态
            - 如果为 True，则关闭信息；如果为 False，则开启信息
        """
        if self.label_original.flag:
            self.label_original.off_show()
            self.label_modified.off_show()
        else:
            self.label_original.open_show()
            self.label_modified.open_show()

    def loadModifiedImage(self):
        """
        弹出“另存为”对话框，保存当前处理后的图像。
        支持中文路径的图像保存。
        """
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(
            self,
            "保存修改后的图片",
            "",
            "Images (*.png *.xpm *.jpg)",
            options=options
        )

        if fileName:
            if self.img_process is None:
                # 弹窗提示错误
                QMessageBox.critical(self, "错误", "当前没有可保存的图像")
                return

            print(f"保存路径: {fileName}")
            # 将 BGR 格式转为 RGB 格式以避免 PyQt 显示异常
            img = cv2.cvtColor(self.img_process, cv2.COLOR_BGR2RGB)
            # 保存图像，使用 imencode 支持中文路径
            cv2.imencode('.jpg', img)[1].tofile(fileName)

    def set_continuous_flag(self):
        if self.is_continuous:
            self.is_continuous = False
            self.setWindowTitle("图像处理软件 - 关闭连续操作")
        else:
            reply = QMessageBox.question(None, '警告', '开启图像连续操作可能会导致一些异常！是否确认开启？',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.is_continuous = True
                self.setWindowTitle("图像处理软件 - 已开启连续操作")
            else:
                print("用户点击了取消，操作已取消")
                self.setWindowTitle("图像处理软件 - 关闭连续操作")

    # 简单图像操作（无需用户输入参数）
    def simple_operation(self, code, value):
        """
        执行无需额外参数输入的图像处理操作。

        参数:
            code (str): 操作命令字符串（如 'rotate', 'flip' 等）
            value (int): 辅助参数（如旋转角度、翻转方向等）
        """
        self.clear_controls()
        if self.img_original is None:
            QMessageBox.critical(self, "错误", f"你没有选择图片")
            return
        if not self.is_continuous:
            self.reset_image()
        try:
            # 根据 code 调用对应的图像处理函数
            if code == 'rotate':
                self.img_process = rotate_angle(self.img_process, value)
            elif code == 'globalHist':
                self.img_process = hisEqulColor1(self.img_process)
            elif code == 'adjustiveHist':
                self.img_process = hisEqulColor2(self.img_process)
            elif code == 'flip':
                self.img_process = filp_by_type(self.img_process, value)
            elif code == 'log':
                self.img_process = log_enhancement(self.img_process)
            elif code == 'hist':
                self.img_process = gray_hist(self.img_process)
            elif code == 'MedianLowPass':
                self.img_process = Space_Fliter(self.img_process, 'median')
            elif code == 'BilaLowPass':
                self.img_process = Space_Fliter(self.img_process, 'bilateral')
            elif code == 'MeanShiftLowPass':
                self.img_process = Space_Fliter(self.img_process, 'mean')
            elif code == 'robels_sharpen':
                self.img_process = sharpen_by_space(self.img_process, 'robels')
            elif code == 'sobel_sharpen':
                self.img_process = sharpen_by_space(self.img_process, 'sobel')
            elif code == 'laplacian_sharpen':
                self.img_process = sharpen_by_space(self.img_process, 'laplacian')
            elif code == 'scharr_sharpen':
                self.img_process = sharpen_by_space(self.img_process, 'scharr')
            elif code == 'Gauss_Noise':
                self.img_process = Gauss_Noise(self.img_process)
            elif code == 'Rayleigh_Noise':
                self.img_process = Rayleigh_Noise(self.img_process)
            elif code == 'Ireland_Noise':
                self.img_process = Ireland_Noise(self.img_process)
            elif code == 'Exponential_Noise':
                self.img_process = Exponential_Noise(self.img_process)
            elif code == 'Uniform_Noise':
                self.img_process = Uniform_Noise(self.img_process)
            elif code == 'SaltPepper_Noise':
                self.img_process = SaltPepper_Noise(self.img_process)
            elif code == 'Arithmetic_Mean_Filter':
                self.img_process = Arithmetic_Mean_Filter(self.img_process)
            elif code == 'Geometric_Mean_Filter':
                self.img_process = Geometric_Mean_Filter(self.img_process)
            elif code == 'Harmonic_Mean_Filter':
                self.img_process = Harmonic_Mean_Filter(self.img_process)
            elif code == 'Max_Filter':
                self.img_process = Statistical_Sorting_Filter(self.img_process, 'max')
            elif code == 'Min_Filter':
                self.img_process = Statistical_Sorting_Filter(self.img_process, 'min')
            elif code == 'Middle_Filter':
                self.img_process = Statistical_Sorting_Filter(self.img_process, 'middle')
            elif code == 'Median_Filter':
                self.img_process = Statistical_Sorting_Filter(self.img_process, 'median')
            elif code =='wavelet_denoise':
                self.img_process = wavelet_denoise(self.img_process)
            elif code == 'roberts_split':
                self.img_process = roberts_kernel(self.img_process)
            elif code == 'sobel_split':
                self.img_process = sobel_kernel(self.img_process)
            elif code == 'laplacian_split':
                self.img_process = laplacian_kernel(self.img_process)
            elif code == 'log_split':
                self.img_process = log_kernel(self.img_process)
            elif code == 'canny_split':
                self.img_process = canny_kernel(self.img_process)
            elif code == 'HoughLines':
                self.img_process = HoughLines(self.img_process)
            elif code == 'HoughLinesP':
                self.img_process = HoughLinesP(self.img_process)
            elif code == 'regionalSplit':
                self.img_process = regional_split_algorithm(self.img_process)
            elif code == 'kmeans':
                self.img_process = kmeans_algorithm(self.img_process)
            elif code == 'disWaterHold':
                self.img_process = dis_waterhold_algorithm(self.img_process)
            elif code == 'sobelWaterHold':
                self.img_process = sobel_waterhold_algorithm(self.img_process)
            elif code == 'localThreshold':
                self.img_process = local_threshold(self.img_process)
            elif code == 'Erosion_Rect':
                self.img_process = Erosion_Or_Dilation(self.img_process, 'rect', 'erode')
            elif code == 'Erosion_Cross':
                self.img_process = Erosion_Or_Dilation(self.img_process, 'cross', 'erode')
            elif code == 'Erosion_Ellipse':
                self.img_process = Erosion_Or_Dilation(self.img_process, 'ellipse', 'erode')
            elif code == 'Dilation_Rect':
                self.img_process = Erosion_Or_Dilation(self.img_process, 'rect', 'dilate')
            elif code == 'Dilation_Cross':
                self.img_process = Erosion_Or_Dilation(self.img_process, 'cross', 'dilate')
            elif code == 'Dilation_Ellipse':
                self.img_process = Erosion_Or_Dilation(self.img_process, 'ellipse', 'dilate')
            elif code == 'Open_Rect':
                self.img_process = Open_Or_Close(self.img_process, 'rect', 'open')
            elif code == 'Open_Cross':
                self.img_process = Open_Or_Close(self.img_process, 'cross', 'open')
            elif code == 'Open_Ellipse':
                self.img_process = Open_Or_Close(self.img_process, 'ellipse', 'open')
            elif code == 'Close_Rect':
                self.img_process = Open_Or_Close(self.img_process, 'rect', 'close')
            elif code == 'Close_Cross':
                self.img_process = Open_Or_Close(self.img_process, 'cross', 'close')
            elif code == 'Close_Ellipse':
                self.img_process = Open_Or_Close(self.img_process, 'ellipse', 'close')
            elif code == 'smooth':
                self.img_process = skinSmoothen(self.img_process, 10, 10)
            elif code == 'thinner':
                self.img_process = face_thin_auto_optimized(self.img_process)
            elif code == 'make_up':
                self.img_process = Make_Up(self.img_process)
            elif code == 'candy':
                self.img_process = process_image_with_style(self.img_process, 'candy')
            elif code == 'mosaic':
                self.img_process = process_image_with_style(self.img_process, 'mosaic')
            elif code == 'rain_princess':
                self.img_process = process_image_with_style(self.img_process, 'rain_princess')
            elif code == 'udnie':
                self.img_process = process_image_with_style(self.img_process, 'udnie')
            elif code == 'starry_night':
                self.img_process = process_image_with_style(self.img_process, 'starry_night')
            elif code == 'picasso':
                self.img_process = process_image_with_style(self.img_process, 'picasso')
            elif code == 'cuphead':
                self.img_process = process_image_with_style(self.img_process, 'cuphead')
            elif code == 'JoJo':
                self.img_process = process_image_with_style(self.img_process, 'JoJo')
            elif code == 'fu_shi_hui':
                self.img_process = process_image_with_style(self.img_process, 'fu_shi_hui')
            elif code == 'anime':
                self.img_process = process_image_with_style(self.img_process, 'anime')
            elif code == 'mc':
                self.img_process = process_image_with_style(self.img_process, 'mc')

        except Exception as e:
            # 使用 QMessageBox 显示错误信息
            QMessageBox.critical(self, "错误", f"图像处理操作失败：\n{str(e)}")

        self.update_img(self.img_process)  # 更新界面显示

    # 用户输入参数操作
    def Operation_by_Input(self, code):
        """
            用户输入参数操作入口函数，用于需要用户提供参数的图像处理任务。

            参数:
                   code (str): 操作命令字符串（如 'scale', 'translate' 等）
        """
        self.clear_controls()
        self.value1 = QLineEdit(self)
        self.vbox.addWidget(self.value1)
        self.value2 = None
        try:
            if code == 'idealLowPass':
                self.value1.setPlaceholderText("滤波器控制参数 (截止频率)")
                self.update_button = QPushButton("确定应用理想低通滤波")
            elif code == 'gaussLowPass':
                self.value1.setPlaceholderText("滤波器控制参数 (截止频率)")
                self.update_button = QPushButton("确定应用高斯低通滤波")
            elif code == 'butterworthLowPass':
                self.value1.setPlaceholderText("滤波器控制参数 (截止频率)")
                self.update_button = QPushButton("确定应用巴特沃斯低通滤波")
            if code == 'idealHighPass':
                self.value1.setPlaceholderText("滤波器控制参数 (截止频率)")
                self.update_button = QPushButton("确定应用理想高通滤波")
            elif code == 'gaussHighPass':
                self.value1.setPlaceholderText("滤波器控制参数 (截止频率)")
                self.update_button = QPushButton("确定应用高斯高通滤波")
            elif code == 'butterworthHighPass':
                self.value1.setPlaceholderText("滤波器控制参数 (截止频率)")
                self.update_button = QPushButton("确定应用巴特沃斯高通滤波")
            if code == 'scale':
                self.value1.setPlaceholderText("请输入缩放比例")
                self.value2 = QLineEdit(self)
                self.vbox.addWidget(self.value2)
                self.value2.setPlaceholderText("请输入缩放比例")
                self.update_button = QPushButton("确定放缩")
            elif code == 'translate':
                self.value1.setPlaceholderText("请输入平移x距离")
                self.value2 = QLineEdit(self)
                self.vbox.addWidget(self.value2)
                self.value2.setPlaceholderText("请输入平移y距离")
                self.update_button = QPushButton("确定平移")
            elif code == 'linear':
                self.value1.setPlaceholderText("请输入原始灰度起始值 a")
                if self.value2 is None:
                    self.value2 = QLineEdit(self)
                    self.vbox.addWidget(self.value2)
                    self.value2.setPlaceholderText("请输入原始灰度结束值 b")
                self.value3 = QLineEdit(self)
                self.value4 = QLineEdit(self)
                self.vbox.addWidget(self.value3)
                self.vbox.addWidget(self.value4)
                self.value3.setPlaceholderText("请输入目标灰度起始值 c")
                self.value4.setPlaceholderText("请输入原始灰度结束值 d")
                self.update_button = QPushButton("确定线性变换")
            elif code == 'binary_layered':
                self.value1.setPlaceholderText("请输入二值阈值起始")
                self.value2 = QLineEdit(self)
                self.vbox.addWidget(self.value2)
                self.value2.setPlaceholderText("请输入二值阈值结束")
                self.update_button = QPushButton("确定二值处理")
            elif code == 'scale_layered':
                self.value1.setPlaceholderText("请输入窗口起始值")
                self.value2 = QLineEdit(self)
                self.vbox.addWidget(self.value2)
                self.value2.setPlaceholderText("请输入窗口结束值")
                self.update_button = QPushButton("确定窗口处理")
            elif code == 'bigger':
                self.value1.setPlaceholderText("请输入左眼放大强度")
                self.value2 = QLineEdit(self)
                self.vbox.addWidget(self.value2)
                self.value2.setPlaceholderText("请输入右眼放大强度")
                self.update_button = QPushButton("确定应用大眼")
            elif code=='globalThreshold':
                self.value1.setPlaceholderText("请输入阈值")
                self.update_button = QPushButton("确定应用全局阈值处理")
            elif code == 'multiThreshold':
                self.value1.setPlaceholderText("请输入分类数（2-4之间）")
                self.update_button = QPushButton("确定应用多阈值处理")
            elif code == 'regionalGrowth':
                self.value1.setPlaceholderText("请输入区域生长阈值")
                self.update_button = QPushButton("确定应用区域生长处理")
            elif code == 'wavelet_compress':
                self.value1.setPlaceholderText("请输入压缩参数（0-100）越大压缩越明显")
                self.update_button = QPushButton("确定应用小波压缩")
        except Exception as e:
            # 使用 QMessageBox 显示错误信息
            QMessageBox.critical(self, "错误", f"图像处理操作失败：\n{str(e)}")
        self.vbox.addWidget(self.update_button)
        self.update_button.clicked.connect(lambda: self.tmp_Operation_by_Input(code))

    # 输入参数操作执行
    def tmp_Operation_by_Input(self, code):
        """
               实际执行用户输入参数的图像处理操作。

               参数:
                   code (str): 操作命令字符串
               """
        self.tmp_img = None
        value1 = 0
        if not self.is_continuous:
            self.reset_image()
        try:
            if self.value1 is not None:
                value1 = int(self.value1.text())
            if self.value2 is not None:
                value2 = int(self.value2.text())
            if code == 'scale' or code == 'translate':
                self.tmp_img = Scale_Or_Translate_by_input(self.img_process, code, value1, value2)
            elif code == 'binary_layered':
                self.tmp_img = gray_layered(self.img_process, 'binary', value1, value2)
            elif code == 'scale_layered':
                self.tmp_img = gray_layered(self.img_process, 'scale', value1, value2)
            elif code == 'idealLowPass':
                self.tmp_img = Low_Pass(self.img_process, value1, 'ideal')
            elif code == 'gaussLowPass':
                self.tmp_img = Low_Pass(self.img_process, value1, 'gauss')
            elif code == 'butterworthLowPass':
                self.tmp_img = Low_Pass(self.img_process, value1, 'butterworth')
            elif code == 'idealHighPass':
                self.tmp_img = High_Pass(self.img_process, value1, 'ideal')
            elif code == 'gaussHighPass':
                self.tmp_img = High_Pass(self.img_process, value1, 'gauss')
            elif code == 'butterworthHighPass':
                self.tmp_img = High_Pass(self.img_process, value1, 'butterworth')
            elif code == 'globalThreshold':
                self.tmp_img = global_threshold(self.img_process, value1)
            elif code =='multiThreshold':
                self.tmp_img = multi_threshold(self.img_process, value1)
            elif code == 'regionalGrowth':
                self.tmp_img = auto_region_growing(self.img_process, value1)
            elif code == 'wavelet_compress':
                self.tmp_img = wavelet_compress(self.img_process, threshold=(value1/100))
            elif code == 'linear':
                value3 = int(self.value3.text())
                value4 = int(self.value4.text())
                self.tmp_img = linear_enhancement(self.img_process, value1, value2, value3, value4)
            elif code == 'bigger':
                self.tmp_img = eye_bigger_auto(self.img_process, value1, value2)
        except ValueError:
            QMessageBox.critical(self, "错误", f"请输入有效的数字作为宽度和高度。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像处理操作失败：\n{str(e)}")
        self.vbox.removeWidget(self.value1)
        self.value1 = None
        self.vbox.removeWidget(self.value2)
        self.value2 = None
        if code == 'linear':
            self.vbox.removeWidget(self.value3)
            self.value3 = None
            self.vbox.removeWidget(self.value4)
            self.value4 = None

        self.click_to_update()

    # 添加滑块控件
    def add_slider(self, type):
        """
               添加交互式滑块控件，用于连续参数调节。

               参数:
                   type (str): 滑块类型（如 'bright', 'contrast', 'gamma' 等）
               """
        self.clear_controls()
        self.slider = QSlider(Qt.Horizontal)
        try:
            if type == 'rotate':
                self.slider.setMinimum(-180)
                self.slider.setMaximum(180)
                self.slider.setValue(0)
                self.update_button = QPushButton("确定旋转")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Rotate', self.slider.value(), None))
            elif type == 'scale':
                self.slider.setMinimum(0)
                self.slider.setMaximum(200)
                self.slider.setValue(100)
                self.update_button = QPushButton("确定放缩")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Scale', self.slider.value(), None))
            elif type == 'bright':
                self.slider.setMinimum(-100)
                self.slider.setMaximum(100)
                self.slider.setValue(0)
                self.update_button = QPushButton("确定调整亮度")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Bright_or_Contrast', self.slider.value(), 'bright'))
            elif type == 'contrast':
                self.slider.setMinimum(0)
                self.slider.setMaximum(200)
                self.slider.setValue(100)
                self.update_button = QPushButton("确定调整对比度")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Bright_or_Contrast', self.slider.value(), 'contrast'))
            elif type == 'saturation':
                self.slider.setMinimum(0)
                self.slider.setMaximum(200)
                self.slider.setValue(100)
                self.update_button = QPushButton("确定调整饱和度")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Saturation', self.slider.value(), None))
            elif type == 'exposure':
                self.slider.setMinimum(0)
                self.slider.setMaximum(200)
                self.slider.setValue(100)
                self.update_button = QPushButton("确定调整曝光")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Exposure', self.slider.value(), None))
            elif type == 'value':
                self.slider.setMinimum(0)
                self.slider.setMaximum(200)
                self.slider.setValue(100)
                self.update_button = QPushButton("确定调整明度")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('Value', self.slider.value(), None))
            elif type == 'gamma':
                self.slider.setMinimum(0)
                self.slider.setMaximum(400)
                self.slider.setValue(100)
                self.update_button = QPushButton("确定应用伽马变换")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('gamma', self.slider.value(), None))
            elif type == 'hue':
                self.slider.setMinimum(0)
                self.slider.setMaximum(180)
                self.slider.setValue(0)
                self.update_button = QPushButton("确定应用色相变换")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('hue', self.slider.value(), None)
                )
            elif type == 'whiten':
                self.slider.setMinimum(30)
                self.slider.setMaximum(70)
                self.slider.setValue(50)
                self.update_button = QPushButton("确定应用美白")
                self.slider.valueChanged.connect(
                    lambda: self.slider_image_operation('whiten', self.slider.value(), None))
        except Exception as e:
            # 使用 QMessageBox 显示错误信息
            QMessageBox.critical(self, "错误", f"图像处理操作失败：\n{str(e)}")
        self.vbox.addWidget(self.slider)
        self.vbox.addWidget(self.update_button)
        self.update_button.clicked.connect(self.click_to_update)

    # 滑块操作执行
    def slider_image_operation(self, command, value, type):
        """
               执行滑块控制的图像处理操作。

               参数:
                   command (str): 操作命令（如 'Bright_or_Contrast', 'Saturation' 等）
                   value (int): 滑块数值
                   type (str): 子类型（如 'bright', 'contrast' 等）
               """
        self.tmp_img = self.img_process
        if not self.is_continuous:
            self.reset_image()
        try:
            if command == 'Bright_or_Contrast':
                self.tmp_img = BorC_Convert(self.img_process, value, type)
            elif command == 'Scale':
                self.tmp_img = scale_by_equal_position(self.img_process, value)
            elif command == 'Rotate':
                self.tmp_img = rotate_angle(self.img_process, value)
            elif command == 'Saturation':
                self.tmp_img = Saturation_by_LUT(self.img_process, value / 100)
            elif command == 'Exposure':
                self.tmp_img = Exposure_by_Gamma(self.img_process, value / 100)
            elif command == 'Value':
                self.tmp_img = Value_by_LUT(self.img_process, value / 100)
            elif command == 'gamma':
                self.tmp_img = gray_gamma(self.img_process, value / 100)
            elif command== 'hue':
                self.tmp_img = Hue_by_LUT(self.img_process, value)
            elif command == 'whiten':
                self.tmp_img = SkinWhiten(self.img_process, value / 100)
        except Exception as e:
            # 使用 QMessageBox 显示错误信息
            QMessageBox.critical(self, "错误", f"图像处理操作失败：\n{str(e)}")

        self.update_img(self.tmp_img)

    # 更新图像
    def click_to_update(self):
        """
               更新当前处理图像，并清理界面上的控件。
               """

        self.clear_controls()

        # 触发布局重算
        self.vbox.update()
        self.vbox.parent().adjustSize()

        self.img_process = self.tmp_img
        self.update_img(self.img_process)


    # 更新 QLabel 显示图像
    def update_img(self, img):
        """
               更新 QLabel 显示图像内容。

               参数:
                   img (numpy.ndarray): 待显示的图像数据
               """
        if isinstance(img, np.ndarray):
            Qimg = cv2_img_to_qimg(img)
            pixmap = QPixmap.fromImage(Qimg)
            self.label_modified.setPixmap(pixmap.scaled(self.label_modified.size(), Qt.KeepAspectRatio))

    # 重置图像到原始状态
    def reset_image(self):
        """
               将图像恢复到原始状态。
               """
        try:
                self.img_process = self.img_original.copy()
                # if self.original_width and self.original_height:
                #     self.label_modified.setPixmap(
                #         QPixmap.fromImage(cv2_img_to_qimg(self.img_original))
                #         .scaled(self.original_width, self.original_height, Qt.KeepAspectRatio)
                #     )
                self.update_img(self.img_process)
        except :
            QMessageBox.critical(self, "没有可重置的图片")



    def clear_controls(self):
        """
               清理界面上的控件。
               """
        if self.update_button:
            self.vbox.removeWidget(self.update_button)
            self.update_button.deleteLater()
            self.update_button = None

        if self.slider:
            self.vbox.removeWidget(self.slider)
            self.slider.deleteLater()
            self.slider = None

        if self.value1:
            self.vbox.removeWidget(self.value1)
            self.value1.deleteLater()
            self.value1 = None

        if self.value2:
            self.vbox.removeWidget(self.value2)
            self.value2.deleteLater()
            self.value2 = None

        if self.value3:
            self.vbox.removeWidget(self.value3)
            self.value3.deleteLater()
            self.value3 = None

        if  self.value4:
            self.vbox.removeWidget(self.value4)
            self.value4.deleteLater()
            self.value4 = None



# 程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = myUI()
    ui.show()  # 显示窗口
    sys.exit(app.exec_())
