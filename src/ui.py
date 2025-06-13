
from PyQt5.QtCore import QRect, QMetaObject, QCoreApplication
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMenuBar, QAction, QWidget,  QMenu, QLabel, QHBoxLayout, QVBoxLayout
from myQtLabel import *

"""
ui.py - PyQt5 GUI 设计主界面模块

提供完整的图像处理软件菜单栏结构与图像显示布局：
- 支持多种图像操作：旋转、缩放、裁剪、平移、翻转
- 图像增强：亮度/对比度/饱和度调整、Gamma 校正、直方图均衡
- 滤波器：低通、高通、中值滤波等
- 边缘检测、霍夫变换、区域生长、K-Means 分割
- 形态学操作（腐蚀、膨胀、开闭运算）
- 人像美化功能（美白、磨皮、大眼、瘦脸、上妆）
- 使用 HoverLabel 实现鼠标悬停信息显示
"""

class Ui_ImageProcess(object):
    """
       图像处理软件的主界面 UI 类。

       功能概述：
           - 提供完整的菜单栏系统，支持图像处理功能分类导航
           - 包含图像显示区域（两个 QLabel 显示原始与处理后图像）
           - 支持中文路径图像加载
           - 所有菜单项在初始化时绑定到主窗口

       成员变量:
           MyMenu (QMenuBar): 主菜单栏
           FileList (QMenu): 文件菜单（打开/保存）
           BasicOperation (QMenu): 基本操作（旋转、缩放、裁剪等）
           Enhancement (QMenu): 图像增强（亮度、对比度、Gamma 等）
           Smooth (QMenu): 平滑滤波器（空域、频域）
           Sharpen (QMenu): 锐化算子（Sobel、Laplacian、Canny 等）
           Restoration (QMenu): 图像恢复（噪声添加与去除）
           Split (QMenu): 图像分割（边缘检测、阈值处理、聚类分割）
           Morphology (QMenu): 形态学操作（腐蚀、膨胀、开闭运算）
           Facial (QMenu): 人像美化（美白、大眼、瘦脸等）
           label_original / label_modified (HoverLabel): 图像显示控件
           vbox / hbox (QVBoxLayout/QHBoxLayout): 布局管理
       """
    def setupUi(self, ImageProcess):
        """
               初始化主窗口界面布局与菜单栏结构。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        if ImageProcess.objectName():
            ImageProcess.setObjectName(u"ImageProcess")
        ImageProcess.resize(1024, 618)
        ImageProcess.setWindowTitle(QCoreApplication.translate("ImageProcess", u"图像处理软件 - 关闭连续操作", None))

        self.MyMenu = QMenuBar(ImageProcess)
        self.MyMenu.setObjectName(u"MyMenu")
        self.MyMenu.setGeometry(QRect(0, 0, 1024, 24))

        self.file_list_init(ImageProcess)
        self.basic_operation_list_init(ImageProcess)
        self.strength_list_init(ImageProcess)
        self.smooth_list_init(ImageProcess)
        self.sharpen_list_init(ImageProcess)
        self.restoration_list_init(ImageProcess)
        self.split_list_init(ImageProcess)
        self.morphology_list_init(ImageProcess)
        self.face_list_init(ImageProcess)
        self.style_list_init(ImageProcess)

        self.Setting = QMenu(self.MyMenu)
        self.Setting.setObjectName(u"StyleChange")
        self.Setting.setTitle(QCoreApplication.translate("ImageProcess", u"设置", None))

        self.partialScale = QMenu(self.Setting)
        self.partialScale.setObjectName(u"partialScale")
        self.partialScale.setTitle(QCoreApplication.translate("ImageProcess", u"显示光标信息", None))
        self.Setting.addAction(self.partialScale.menuAction())
        self.setFlag = QAction(self.partialScale)
        self.setFlag.setObjectName(u"setFlag")
        self.setFlag.setText(QCoreApplication.translate("ImageProcess", u"开关", None))
        self.partialScale.addAction(self.setFlag)

        self.isContinuous = QMenu(self.Setting)
        self.isContinuous.setObjectName(u"isContinuous")
        self.isContinuous.setTitle(QCoreApplication.translate("ImageProcess", u"连续操作", None))
        self.Setting.addAction(self.isContinuous.menuAction())
        self.setContinuousFlag = QAction(self.isContinuous)
        self.setContinuousFlag.setObjectName(u"setContinuousFlag")
        self.setContinuousFlag.setText(QCoreApplication.translate("ImageProcess", u"开关", None))
        self.isContinuous.addAction(self.setContinuousFlag)

        ImageProcess.setMenuBar(self.MyMenu)

        self.ResetMenu = QMenu(self.MyMenu)
        self.ResetMenu.setObjectName(u"ResetMenu")
        self.ResetMenu.setTitle(QCoreApplication.translate("ImageProcess", u"重置", None))

        self.ResetAction = QAction(self.ResetMenu)
        self.ResetAction.setObjectName(u"ResetAction")
        self.ResetAction.setText(QCoreApplication.translate("ImageProcess", u"重置图像", None))
        self.ResetMenu.addAction(self.ResetAction)



        self.MyMenu.addAction(self.FileList.menuAction())
        self.MyMenu.addAction(self.BasicOperation.menuAction())
        self.MyMenu.addAction(self.Enhancement.menuAction())
        self.MyMenu.addAction(self.Split.menuAction())
        self.MyMenu.addAction(self.Smooth.menuAction())
        self.MyMenu.addAction(self.Sharpen.menuAction())
        self.MyMenu.addAction(self.Restoration.menuAction())
        self.MyMenu.addAction(self.Morphology.menuAction())
        self.MyMenu.addAction(self.Facial.menuAction())
        self.MyMenu.addAction(self.Style.menuAction())
        # 将 ResetMenu 添加到菜单栏中
        self.MyMenu.addAction(self.ResetMenu.menuAction())
        self.MyMenu.addAction(self.Setting.menuAction())


        self.addImage()


        # 将布局设置为主窗口的中心部件
        self.container = QWidget()
        self.container.setLayout(self.vbox)
        ImageProcess.setCentralWidget(self.container)


        QMetaObject.connectSlotsByName(ImageProcess)
        # setupUi

    def file_list_init(self,ImageProcess):
        """
               初始化文件菜单（打开图像、保存图像）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        # 创建文件菜单选项
        self.FileList = QMenu(self.MyMenu)
        self.FileList.setObjectName(u"FileList")
        self.FileList.setTitle(QCoreApplication.translate("ImageProcess", u"\u6587\u4ef6", None))

        # 打开彩色图片
        self.OpenColorImg = QAction(ImageProcess)
        self.OpenColorImg.setObjectName(u"OpenColorImg")
        self.OpenColorImg.setText(
            QCoreApplication.translate("ImageProcess", u"\u6253\u5f00\u5f69\u8272\u56fe\u50cf", None))

        # 打开灰度图像
        self.OpenGrayImg = QAction(ImageProcess)
        self.OpenGrayImg.setObjectName(u"OpenGrayImg")
        self.OpenGrayImg.setText(
            QCoreApplication.translate("ImageProcess", u"\u6253\u5f00\u7070\u5ea6\u56fe\u50cf", None))

        # 创建打开图像子菜单，添加两种打开图片的方法
        self.OpenImage = QMenu(self.FileList)
        self.OpenImage.setObjectName(u"OpenImage")
        self.OpenImage.setTitle(QCoreApplication.translate("ImageProcess", u"\u6253\u5f00\u56fe\u7247", None))
        self.OpenImage.addAction(self.OpenColorImg)
        self.OpenImage.addAction(self.OpenGrayImg)
        self.FileList.addAction(self.OpenImage.menuAction())

        # 关闭图片选项
        self.CloseImage = QAction(ImageProcess)
        self.CloseImage.setObjectName(u"CloseImage")
        self.CloseImage.setText(QCoreApplication.translate("ImageProcess", u"\u4fdd\u5b58\u56fe\u7247", None))
        self.FileList.addAction(self.CloseImage)

    def Rotate_Operation_Init(self,ImageProcess,parent_menu):
        """
                初始化旋转菜单及其子项（90°、180°、270°、任意角度）。

                参数:
                    ImageProcess (QMainWindow): 主窗口对象
                    parent_menu (QMenu): 父级菜单，用于添加该菜单项
                """
        self.Rotate = QMenu(parent_menu)
        self.Rotate.setObjectName(u"Rotate")
        # 添加4种旋转
        self.Rotate90 = QAction(ImageProcess)
        self.Rotate90.setObjectName(u"Rotate90")
        self.Rotate90.setText(QCoreApplication.translate("ImageProcess", u"旋转90度", None))

        self.Rotate180 = QAction(ImageProcess)
        self.Rotate180.setObjectName(u"Rotate180")
        self.Rotate180.setText(QCoreApplication.translate("ImageProcess", u"旋转180度", None))
        self.Rotate270 = QAction(ImageProcess)
        self.Rotate270.setObjectName(u"Rotate270")
        self.Rotate270.setText(QCoreApplication.translate("ImageProcess", u"旋转270度", None))

        self.RotateFree = QAction(ImageProcess)
        self.RotateFree.setObjectName(u"RotateFree")
        self.RotateFree.setText(
            QCoreApplication.translate("ImageProcess", u"旋转任意角度", None))
        self.Rotate.addAction(self.Rotate90)
        self.Rotate.addAction(self.Rotate180)
        self.Rotate.addAction(self.Rotate270)
        self.Rotate.addAction(self.RotateFree)
        self.Rotate.setTitle(QCoreApplication.translate("ImageProcess", u"旋转", None))
        parent_menu.addAction(self.Rotate.menuAction())

    def Scale_Operation_Init(self, ImageProcess, parent_menu):
        """
                初始化缩放菜单及其子项（等比例缩放、输入比例缩放）。

                参数:
                    ImageProcess (QMainWindow): 主窗口对象
                    parent_menu (QMenu): 父级菜单，用于添加该菜单项
                """
        self.Scale = QMenu(parent_menu)
        self.Scale.setObjectName(u"Scale")
        self.Scale.setTitle(QCoreApplication.translate("ImageProcess", u"放缩", None))

        # 创建等比例缩放动作
        self.scaleByEqualPosition = QAction(ImageProcess)
        self.scaleByEqualPosition.setText(
            QCoreApplication.translate("ImageProcess", u"\u7b49\u6bd4\u4f8b\u653e\u7f29", None))
        self.scaleByEqualPosition.setObjectName(u"scaleByEqualPosition")
        self.Scale.addAction(self.scaleByEqualPosition)

        # 创建输入比例缩放动作
        self.scaleByInput = QAction(ImageProcess)
        self.scaleByInput.setObjectName(u"scaleByInput")
        self.scaleByInput.setText(
            QCoreApplication.translate("ImageProcess", u"\u4efb\u610f\u6bd4\u4f8b\u653e\u7f29", None))
        self.Scale.addAction(self.scaleByInput)

        parent_menu.addAction(self.Scale.menuAction())

    def Translate_Operation_Init(self, ImageProcess, parent_menu):
        """
                初始化平移菜单及其子项（平移）。

                参数:
                    ImageProcess (QMainWindow): 主窗口对象
                    parent_menu (QMenu): 父级菜单，用于添加该菜单项
                """
        self.Translate = QAction(parent_menu)
        self.Translate.setObjectName(u"Translate")
        self.Translate.setText(QCoreApplication.translate("ImageProcess", u"平移", None))
        parent_menu.addAction(self.Translate)

    def Flip_Operation_Init(self, ImageProcess, parent_menu):
        """
               初始化翻转菜单（水平、垂直、双向）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
                   parent_menu (QMenu): 父级菜单
               """
        self.Flip = QMenu(parent_menu)
        self.Flip.setObjectName(u"Translate")
        self.Flip.setTitle(QCoreApplication.translate("ImageProcess", u"翻转", None))
        # 创建水平翻转
        self.HorizontalFlip = QAction(ImageProcess)
        self.HorizontalFlip.setText(
            QCoreApplication.translate("ImageProcess", u"水平翻转", None))
        self.HorizontalFlip.setObjectName(u"HorizontalFlip")
        self.Flip.addAction(self.HorizontalFlip)

        # 创建垂直翻转
        self.VerticalFlip = QAction(ImageProcess)
        self.VerticalFlip.setText(
            QCoreApplication.translate("ImageProcess", u"垂直翻转", None))
        self.VerticalFlip.setObjectName(u"VerticalFlip")
        self.Flip.addAction(self.VerticalFlip)

        # 水平垂直翻转
        self.DualFlip = QAction(ImageProcess)
        self.DualFlip.setText(
            QCoreApplication.translate("ImageProcess", u"水平垂直翻转", None))
        self.DualFlip.setObjectName(u"DualFlip")
        self.Flip.addAction(self.DualFlip)

        parent_menu.addAction(self.Flip.menuAction())

    def Crop_Operation_Init(self, ImageProcess, parent_menu):
        """
               初始化图像裁剪菜单项。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
                   parent_menu (QMenu): 父级菜单
               """
        self.Crop = QAction(parent_menu)
        self.Crop.setObjectName(u"Crop")
        self.Crop.setText(QCoreApplication.translate("ImageProcess", u"裁剪", None))
        parent_menu.addAction(self.Crop)

    def basic_operation_list_init(self,ImageProcess):
        """
               初始化基本操作菜单（旋转、缩放、翻转、裁剪）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.BasicOperation = QMenu(self.MyMenu)
        self.BasicOperation.setObjectName(u"BasicOperation")
        self.BasicOperation.setTitle(QCoreApplication.translate("ImageProcess", u"基本操作", None))

        self.Rotate_Operation_Init(ImageProcess, self.BasicOperation)
        self.Scale_Operation_Init(ImageProcess, self.BasicOperation)
        self.Translate_Operation_Init(ImageProcess, self.BasicOperation)
        self.Flip_Operation_Init(ImageProcess, self.BasicOperation)
        self.Crop_Operation_Init(ImageProcess, self.BasicOperation)

    def color_enhancement_init(self,ImageProcess):
        """
               初始化彩色图像增强菜单项（亮度、对比度、饱和度、曝光等）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Exposure = QAction(ImageProcess)
        self.Exposure.setObjectName(u"Exposure")
        self.Exposure.setText(QCoreApplication.translate("ImageProcess", u"曝光", None))
        self.ColorStrength.addAction(self.Exposure)

        self.Brightness = QAction(ImageProcess)
        self.Brightness.setObjectName(u"Brightness")
        self.Brightness.setText(QCoreApplication.translate("ImageProcess", u"亮度", None))
        self.ColorStrength.addAction(self.Brightness)

        self.Contrast = QAction(ImageProcess)
        self.Contrast.setObjectName(u"Contrast")
        self.Contrast.setText(QCoreApplication.translate("ImageProcess", u"对比度", None))
        self.ColorStrength.addAction(self.Contrast)

        self.Saturation = QAction(ImageProcess)
        self.Saturation.setObjectName(u"Saturation")
        self.Saturation.setText(QCoreApplication.translate("ImageProcess", u"饱和度", None))
        self.ColorStrength.addAction(self.Saturation)

        self.Hue = QAction(ImageProcess)
        self.Hue.setObjectName(u"Hue")
        self.Hue.setText(QCoreApplication.translate("ImageProcess", u"色相", None))
        self.ColorStrength.addAction(self.Hue)

        self.Value = QAction(ImageProcess)
        self.Value.setObjectName(u"Value")
        self.Value.setText(QCoreApplication.translate("ImageProcess", u"明度", None))
        self.ColorStrength.addAction(self.Value)

        self.globalHist = QAction(ImageProcess)
        self.globalHist.setObjectName(u"globalHist")
        self.globalHist.setText(
            QCoreApplication.translate("ImageProcess", u"全局直方图均衡化", None))

        self.adjustiveHist = QAction(ImageProcess)
        self.adjustiveHist.setObjectName(u"adjustiveHist")
        self.adjustiveHist.setText(
            QCoreApplication.translate("ImageProcess", u"自适应直方图均衡化", None))

    def grey_enhancement_init(self,ImageProcess):
        """
                初始化灰度图像增强菜单项（线性变换、对数变换、伽马校正、直方图均衡）。

                参数:
                    ImageProcess (QMainWindow): 主窗口对象
                """
        self.linearStrength = QAction(ImageProcess)
        self.linearStrength.setObjectName(u"linearStrength")
        self.linearStrength.setText(QCoreApplication.translate("ImageProcess", u"线性变换", None))
        self.GrayStrength.addAction(self.linearStrength)

        self.logStrength = QAction(ImageProcess)
        self.logStrength.setObjectName(u"logStrength")
        self.logStrength.setText(QCoreApplication.translate("ImageProcess", u"对数变换", None))
        self.GrayStrength.addAction(self.logStrength)

        self.gammaStrength = QAction(ImageProcess)
        self.gammaStrength.setObjectName(u"logStrength")
        self.gammaStrength.setText(QCoreApplication.translate("ImageProcess", u"伽马变换", None))
        self.GrayStrength.addAction(self.gammaStrength)

        self.actiongrayHist = QAction(ImageProcess)
        self.actiongrayHist.setObjectName(u"actiongrawHist")
        self.actiongrayHist.setText(
            QCoreApplication.translate("ImageProcess", u"直方图均衡化", None))
        self.GrayStrength.addAction(self.actiongrayHist)

        self.garyLayered = QMenu(ImageProcess)
        self.garyLayered.setObjectName(u"garyLayered")
        self.garyLayered.setTitle(
            QCoreApplication.translate("ImageProcess", u"灰度级分层", None))
        self.GrayStrength.addAction(self.garyLayered.menuAction())

        self.binaryLayered = QAction(self.garyLayered)
        self.binaryLayered.setObjectName(u"garyLayered")
        self.binaryLayered.setText(
            QCoreApplication.translate("ImageProcess", u"二值处理", None))
        self.garyLayered.addAction(self.binaryLayered)

        self.scaleLayered = QAction(self.garyLayered)
        self.scaleLayered.setObjectName(u"garyLayered")
        self.scaleLayered.setText(
            QCoreApplication.translate("ImageProcess", u"窗口处理", None))
        self.garyLayered.addAction(self.scaleLayered)

    def strength_list_init(self,ImageProcess):
        """
               初始化图像增强菜单（包含彩色与灰度增强子菜单）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        # 创建图像增强菜单项
        self.Enhancement = QMenu(self.MyMenu)
        self.Enhancement.setObjectName(u"Enhancement")
        self.Enhancement.setTitle(QCoreApplication.translate("ImageProcess", u"图像增强", None))

        # 增加彩色图像增强项
        self.ColorStrength = QMenu(self.Enhancement)
        self.Enhancement.addAction(self.ColorStrength.menuAction())
        self.ColorStrength.setObjectName(u"ColorStrength")
        self.color_enhancement_init(ImageProcess)
        self.ColorStrength.setTitle(
            QCoreApplication.translate("ImageProcess", u"彩色图像增强", None))

        self.Clarity = QMenu(self.ColorStrength)
        self.Clarity.setObjectName(u"Clarify")
        self.Clarity.addAction(self.globalHist)
        self.Clarity.addAction(self.adjustiveHist)
        self.Clarity.setTitle(QCoreApplication.translate("ImageProcess", u"清晰度", None))
        self.ColorStrength.addAction(self.Clarity.menuAction())

        # 增加灰度图像增强项
        self.GrayStrength = QMenu(self.Enhancement)
        self.GrayStrength.setObjectName(u"GrayStrength")
        self.grey_enhancement_init(ImageProcess)
        self.GrayStrength.setTitle(
            QCoreApplication.translate("ImageProcess", u"灰度图像增强", None))
        self.Enhancement.addAction(self.GrayStrength.menuAction())

    def smooth_list_init(self,ImageProcess):
        """
               初始化图像平滑菜单（空域、频域滤波器）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Smooth = QMenu(self.MyMenu)
        self.Smooth.setObjectName(u"Smooth")
        self.Smooth.setTitle(QCoreApplication.translate("ImageProcess", u"图像平滑", None))


        self.SpaceSmooth = QMenu(self.Smooth)
        self.Smooth.addAction(self.SpaceSmooth.menuAction())
        self.SpaceSmooth.setObjectName(u"SpaceSmooth")
        self.SpaceSmooth.setTitle(
            QCoreApplication.translate("ImageProcess", u"空域平滑", None))

        self.MedianLowPass = QAction(ImageProcess)
        self.MedianLowPass.setObjectName(u"MedianlLowPass")
        self.MedianLowPass.setText(QCoreApplication.translate("ImageProcess", u"中值滤波器", None))
        self.SpaceSmooth.addAction(self.MedianLowPass)

        self.BilaLowPass = QAction(ImageProcess)
        self.BilaLowPass.setObjectName(u"BilaLowPass")
        self.BilaLowPass.setText(QCoreApplication.translate("ImageProcess", u"双边滤波器", None))
        self.SpaceSmooth.addAction(self.BilaLowPass)

        self.MeanShiftLowPass = QAction(ImageProcess)
        self.MeanShiftLowPass.setObjectName(u"MeanShiftLowPass")
        self.MeanShiftLowPass.setText(QCoreApplication.translate("ImageProcess", u"均值偏移滤波器", None))
        self.SpaceSmooth.addAction(self.MeanShiftLowPass)


        self.FreSmooth = QMenu(self.Smooth)
        self.Smooth.addAction(self.FreSmooth.menuAction())
        self.FreSmooth.setObjectName(u"FreSmooth")
        self.FreSmooth.setTitle(
            QCoreApplication.translate("ImageProcess", u"频域平滑", None))

        self.idealLowPass = QAction(ImageProcess)
        self.idealLowPass.setObjectName(u"idealLowPass")
        self.idealLowPass.setText(QCoreApplication.translate("ImageProcess", u"理想低通滤波器", None))
        self.FreSmooth.addAction(self.idealLowPass)

        self.gaussLowPass = QAction(ImageProcess)
        self.gaussLowPass.setObjectName(u"gaussLowPass")
        self.gaussLowPass.setText(QCoreApplication.translate("ImageProcess", u"高斯低通滤波器", None))
        self.FreSmooth.addAction(self.gaussLowPass)

        self.butterworthLowPass = QAction(ImageProcess)
        self.butterworthLowPass.setObjectName(u"idealLowPass")
        self.butterworthLowPass.setText(QCoreApplication.translate("ImageProcess", u"巴特沃斯低通滤波器", None))
        self.FreSmooth.addAction(self.butterworthLowPass)

    def sharpen_list_init(self, ImageProcess):
        """
               初始化图像锐化菜单（空域锐化、频域锐化）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Sharpen = QMenu(self.MyMenu)
        self.Sharpen.setObjectName(u"Sharpen")
        self.Sharpen.setTitle(QCoreApplication.translate("ImageProcess", u"图像锐化", None))

        self.SpaceSharpen = QMenu(self.Sharpen)
        self.Sharpen.addAction(self.SpaceSharpen.menuAction())
        self.SpaceSharpen.setObjectName(u"SpaceSharpen")
        self.SpaceSharpen.setTitle(
            QCoreApplication.translate("ImageProcess", u"空域锐化", None))

        self.roberts_sharpen = QAction(ImageProcess)
        self.roberts_sharpen.setObjectName(u"roberts_sharpen")
        self.roberts_sharpen.setText(QCoreApplication.translate("ImageProcess", u"应用roberts算子", None))
        self.SpaceSharpen.addAction(self.roberts_sharpen)

        self.sobel_sharpen = QAction(ImageProcess)
        self.sobel_sharpen.setObjectName(u"sobel_sharpen")
        self.sobel_sharpen.setText(QCoreApplication.translate("ImageProcess", u"应用sobel算子", None))
        self.SpaceSharpen.addAction(self.sobel_sharpen)

        self.laplacian_sharpen = QAction(ImageProcess)
        self.laplacian_sharpen.setObjectName(u"laplacian_sharpen")
        self.laplacian_sharpen.setText(QCoreApplication.translate("ImageProcess", u"应用laplacian算子", None))
        self.SpaceSharpen.addAction(self.laplacian_sharpen)

        self.scharr_sharpen = QAction(ImageProcess)
        self.scharr_sharpen.setObjectName(u"scharr_sharpen")
        self.scharr_sharpen.setText(QCoreApplication.translate("ImageProcess", u"应用scharr算子", None))
        self.SpaceSharpen.addAction(self.scharr_sharpen)

        self.FreSharpen = QMenu(self.Sharpen)
        self.Sharpen.addAction(self.FreSharpen.menuAction())
        self.FreSharpen.setObjectName(u"FreSharpen")
        self.FreSharpen.setTitle(
            QCoreApplication.translate("ImageProcess", u"频域锐化", None))

        self.idealHighPass = QAction(ImageProcess)
        self.idealHighPass.setObjectName(u"idealHighPass")
        self.idealHighPass.setText(QCoreApplication.translate("ImageProcess", u"理想高通滤波器", None))
        self.FreSharpen.addAction(self.idealHighPass)

        self.gaussHighPass = QAction(ImageProcess)
        self.gaussHighPass.setObjectName(u"gaussLowPass")
        self.gaussHighPass.setText(QCoreApplication.translate("ImageProcess", u"高斯高通滤波器", None))
        self.FreSharpen.addAction(self.gaussHighPass)

        self.butterworthHighPass = QAction(ImageProcess)
        self.butterworthHighPass.setObjectName(u"idealLowPass")
        self.butterworthHighPass.setText(QCoreApplication.translate("ImageProcess", u"巴特沃斯高通滤波器", None))
        self.FreSharpen.addAction(self.butterworthHighPass)

    def restoration_list_init(self,ImageProcess):
        """
               初始化图像恢复菜单（噪声添加与去除、统计排序滤波）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Restoration = QMenu(self.MyMenu)
        self.Restoration.setObjectName(u"Restoration")
        self.Restoration.setTitle(QCoreApplication.translate("ImageProcess", u"图像恢复", None))

        self.addNoise = QMenu(self.Restoration)
        self.Restoration.addAction(self.addNoise.menuAction())
        self.addNoise.setObjectName(u"addNoise")
        self.addNoise.setTitle(
            QCoreApplication.translate("ImageProcess", u"添加噪声", None))

        self.Gauss_Noise = QAction(ImageProcess)
        self.Gauss_Noise.setObjectName(u"Gauss_Noise")
        self.Gauss_Noise.setText(QCoreApplication.translate("ImageProcess", u"添加高斯噪声", None))
        self.addNoise.addAction(self.Gauss_Noise)

        self.Rayleigh_Noise = QAction(ImageProcess)
        self.Rayleigh_Noise.setObjectName(u"Rayleigh_Noise")
        self.Rayleigh_Noise.setText(QCoreApplication.translate("ImageProcess", u"添加瑞利噪声", None))
        self.addNoise.addAction(self.Rayleigh_Noise)

        self.Ireland_Noise = QAction(ImageProcess)
        self.Ireland_Noise.setObjectName(u"Ireland_Noise")
        self.Ireland_Noise.setText(QCoreApplication.translate("ImageProcess", u"添加爱尔兰噪声", None))
        self.addNoise.addAction(self.Ireland_Noise)

        self.Exponential_Noise = QAction(ImageProcess)
        self.Exponential_Noise.setObjectName(u"Exponential_Noise")
        self.Exponential_Noise.setText(QCoreApplication.translate("ImageProcess", u"添加指数噪声", None))
        self.addNoise.addAction(self.Exponential_Noise)

        self.Uniform_Noise = QAction(ImageProcess)
        self.Uniform_Noise.setObjectName(u"Uniform_Noise")
        self.Uniform_Noise.setText(QCoreApplication.translate("ImageProcess", u"添加均匀噪声", None))
        self.addNoise.addAction(self.Uniform_Noise)

        self.SaltPepper_Noise = QAction(ImageProcess)
        self.SaltPepper_Noise.setObjectName(u"SaltPepper_Noise")
        self.SaltPepper_Noise.setText(QCoreApplication.translate("ImageProcess", u"添加椒盐噪声", None))
        self.addNoise.addAction(self.SaltPepper_Noise)

        self.removeNoise = QMenu(self.Restoration)
        self.Restoration.addAction(self.removeNoise.menuAction())
        self.removeNoise.setObjectName(u"addNoise")
        self.removeNoise.setTitle(
            QCoreApplication.translate("ImageProcess", u"去除噪声", None))

        self.Arithmentic_Mean_Filter = QAction(ImageProcess)
        self.Arithmentic_Mean_Filter.setObjectName(u"Arithmentic_Mean_Filter")
        self.Arithmentic_Mean_Filter.setText(QCoreApplication.translate("ImageProcess", u"算数平均滤波器", None))
        self.removeNoise.addAction(self.Arithmentic_Mean_Filter)

        self.Geometric_Mean_Filter = QAction(ImageProcess)
        self.Geometric_Mean_Filter.setObjectName(u"Geometric_Mean_Filter")
        self.Geometric_Mean_Filter.setText(QCoreApplication.translate("ImageProcess", u"几何平均滤波器", None))
        self.removeNoise.addAction(self.Geometric_Mean_Filter)

        self.Harmonic_Mean_Filter = QAction(ImageProcess)
        self.Harmonic_Mean_Filter.setObjectName(u"Harmonic_Mean_Filter")
        self.Harmonic_Mean_Filter.setText(QCoreApplication.translate("ImageProcess", u"谐波平均滤波器", None))
        self.removeNoise.addAction(self.Harmonic_Mean_Filter)

        self.wavelet_denoise=QAction(ImageProcess)
        self.wavelet_denoise.setObjectName(u"wavelet_denoise")
        self.wavelet_denoise.setText(QCoreApplication.translate("ImageProcess", u"小波去噪", None))
        self.removeNoise.addAction(self.wavelet_denoise)

        self.Statistical_Sorting_Filter = QMenu(ImageProcess)
        self.Statistical_Sorting_Filter.setObjectName(u"Statistical_Sorting_Filter")
        self.Statistical_Sorting_Filter.setTitle(QCoreApplication.translate("ImageProcess", u"统计排序滤波器", None))
        self.removeNoise.addAction(self.Statistical_Sorting_Filter.menuAction())

        self.Max_Filter = QAction(ImageProcess)
        self.Max_Filter.setObjectName(u"Max_Filter")
        self.Max_Filter.setText(QCoreApplication.translate("ImageProcess", u"最大值滤波器", None))
        self.Statistical_Sorting_Filter.addAction(self.Max_Filter)

        self.Min_Filter = QAction(ImageProcess)
        self.Min_Filter.setObjectName(u"Min_Filter")
        self.Min_Filter.setText(QCoreApplication.translate("ImageProcess", u"最小值滤波器", None))
        self.Statistical_Sorting_Filter.addAction(self.Min_Filter)

        self.Middle_Filter = QAction(ImageProcess)
        self.Middle_Filter.setObjectName(u"Middle_Filter")
        self.Middle_Filter.setText(QCoreApplication.translate("ImageProcess", u"中点滤波器", None))
        self.Statistical_Sorting_Filter.addAction(self.Middle_Filter)

        self.Midian_Filter = QAction(ImageProcess)
        self.Midian_Filter.setObjectName(u"Midian_Filter")
        self.Midian_Filter.setText(QCoreApplication.translate("ImageProcess", u"中值滤波器", None))
        self.Statistical_Sorting_Filter.addAction(self.Midian_Filter)

        self.compress = QMenu(self.Restoration)
        self.Restoration.addAction(self.compress.menuAction())
        self.compress.setObjectName(u"compress")
        self.compress.setTitle(
            QCoreApplication.translate("ImageProcess", u"图像压缩", None))

        self.wavelet_compress = QAction(ImageProcess)
        self.wavelet_compress.setObjectName(u"wavelet_compress")
        self.wavelet_compress.setText(QCoreApplication.translate("ImageProcess", u"小波变换压缩", None))
        self.compress.addAction(self.wavelet_compress)

    def split_list_init(self,ImageProcess):
        """
               初始化图像分割菜单（边缘检测、直线检测、区域分割、阈值处理）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Split = QMenu(self.MyMenu)
        self.Split.setObjectName(u"Split")
        self.Split.setTitle(QCoreApplication.translate("ImageProcess", u"图像分割", None))

        self.edgeCheck = QMenu(self.Split)
        self.Split.addAction(self.edgeCheck.menuAction())
        self.edgeCheck.setObjectName(u"edgeCheck")
        self.edgeCheck.setTitle(
            QCoreApplication.translate("ImageProcess", u"边缘检测", None))

        self.roberts_split = QAction(ImageProcess)
        self.roberts_split.setObjectName(u"roberts_split")
        self.roberts_split.setText(QCoreApplication.translate("ImageProcess", u"应用roberts算子", None))
        self.edgeCheck.addAction(self.roberts_split)

        self.sobel_split = QAction(ImageProcess)
        self.sobel_split.setObjectName(u"sobel_split")
        self.sobel_split.setText(QCoreApplication.translate("ImageProcess", u"应用sobel算子", None))
        self.edgeCheck.addAction(self.sobel_split)

        self.laplacian_split = QAction(ImageProcess)
        self.laplacian_split.setObjectName(u"laplacian_split")
        self.laplacian_split.setText(QCoreApplication.translate("ImageProcess", u"应用laplacian算子", None))
        self.edgeCheck.addAction(self.laplacian_split)

        self.log_split = QAction(ImageProcess)
        self.log_split.setObjectName(u"log_split")
        self.log_split.setText(QCoreApplication.translate("ImageProcess", u"应用log算子", None))
        self.edgeCheck.addAction(self.log_split)

        self.canny_split = QAction(ImageProcess)
        self.canny_split.setObjectName(u"canny_split")
        self.canny_split.setText(QCoreApplication.translate("ImageProcess", u"应用canny算子", None))
        self.edgeCheck.addAction(self.canny_split)

        self.LineSplit = QMenu(self.Split)
        self.Split.addAction(self.LineSplit.menuAction())
        self.LineSplit.setObjectName(u"rigionSplit")
        self.LineSplit.setTitle(
            QCoreApplication.translate("ImageProcess", u"直线检测", None))

        self.HoughLines = QAction(ImageProcess)
        self.HoughLines.setObjectName(u"HoughLines")
        self.HoughLines.setText(QCoreApplication.translate("ImageProcess", u"霍夫变换", None))
        self.LineSplit.addAction(self.HoughLines)

        self.HoughLinesP = QAction(ImageProcess)
        self.HoughLinesP.setObjectName(u"HoughLinesP")
        self.HoughLinesP.setText(QCoreApplication.translate("ImageProcess", u"累计概率霍夫变换", None))
        self.LineSplit.addAction(self.HoughLinesP)

        self.rigionSplit = QMenu(self.Split)
        self.Split.addAction(self.rigionSplit.menuAction())
        self.rigionSplit.setObjectName(u"rigionSplit")
        self.rigionSplit.setTitle(
            QCoreApplication.translate("ImageProcess", u"图像分割", None))

        self.regionalGrowth = QAction(ImageProcess)
        self.regionalGrowth.setObjectName(u"regionalGrowth")
        self.regionalGrowth.setText(QCoreApplication.translate("ImageProcess", u"区域生长", None))
        self.rigionSplit.addAction(self.regionalGrowth)

        self.regionalSplit = QAction(ImageProcess)
        self.regionalSplit.setObjectName(u"regionalSplit")
        self.regionalSplit.setText(QCoreApplication.translate("ImageProcess", u"区域分离", None))
        self.rigionSplit.addAction(self.regionalSplit)

        self.kmeans = QAction(ImageProcess)
        self.kmeans.setObjectName(u"kmeans")
        self.kmeans.setText(QCoreApplication.translate("ImageProcess", u"k均值聚类", None))
        self.rigionSplit.addAction(self.kmeans)

        self.disWaterHold = QAction(ImageProcess)
        self.disWaterHold.setObjectName(u"disWaterHold")
        self.disWaterHold.setText(QCoreApplication.translate("ImageProcess", u"距离变换分水岭算法", None))
        self.rigionSplit.addAction(self.disWaterHold)

        self.sobelWaterHold = QAction(ImageProcess)
        self.sobelWaterHold.setObjectName(u"sobelWaterHold")
        self.sobelWaterHold.setText(QCoreApplication.translate("ImageProcess", u"sobel梯度分水岭算法", None))
        self.rigionSplit.addAction(self.sobelWaterHold)

        self.thresholdChange = QMenu(self.Split)
        self.Split.addAction(self.thresholdChange.menuAction())
        self.thresholdChange.setObjectName(u"rigionSplit")
        self.thresholdChange.setTitle(
            QCoreApplication.translate("ImageProcess", u"阈值处理", None))

        self.globaThreshold = QAction(ImageProcess)
        self.globaThreshold.setObjectName(u"globaThreshold")
        self.globaThreshold.setText(QCoreApplication.translate("ImageProcess", u"全局阈值处理", None))
        self.thresholdChange.addAction(self.globaThreshold)

        self.localThreshold = QAction(ImageProcess)
        self.localThreshold.setObjectName(u"localThreshold")
        self.localThreshold.setText(QCoreApplication.translate("ImageProcess", u"局部阈值处理", None))
        self.thresholdChange.addAction(self.localThreshold)


        self.multiThreshold = QAction(ImageProcess)
        self.multiThreshold.setObjectName(u"multiThreshold")
        self.multiThreshold.setText(QCoreApplication.translate("ImageProcess", u"多阈值处理", None))
        self.thresholdChange.addAction(self.multiThreshold)



    def morphology_list_init(self,ImageProcess):
        """
               初始化形态学操作菜单（腐蚀、膨胀、开闭运算等）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Morphology = QMenu(self.MyMenu)
        self.Morphology.setObjectName(u"Morphology")
        self.Morphology.setTitle(QCoreApplication.translate("ImageProcess", u"形态学操作", None))

        self.Erodision = QMenu(self.Morphology)
        self.Morphology.addAction(self.Erodision.menuAction())
        self.Erodision.setObjectName(u"Erodision")
        self.Erodision.setTitle(
            QCoreApplication.translate("ImageProcess", u"腐蚀核操作", None))

        self.Erodision_Rect = QAction(self.Erodision)
        self.Erodision.addAction(self.Erodision_Rect)
        self.Erodision_Rect.setObjectName(u"Erodision_Rect")
        self.Erodision_Rect.setText(
            QCoreApplication.translate("ImageProcess", u"矩形核腐蚀操作", None))

        self.Erodision_Cross = QAction(self.Erodision)
        self.Erodision.addAction(self.Erodision_Cross)
        self.Erodision_Cross.setObjectName(u"Erodision_Cross")
        self.Erodision_Cross.setText(
            QCoreApplication.translate("ImageProcess", u"交叉核腐蚀操作", None))

        self.Erodision_Ellipse = QAction(self.Erodision)
        self.Erodision.addAction(self.Erodision_Ellipse)
        self.Erodision_Ellipse.setObjectName(u"Erodision_Ellipse")
        self.Erodision_Ellipse.setText(
            QCoreApplication.translate("ImageProcess", u"椭圆核腐蚀操作", None))

        self.Dilation = QMenu(self.Morphology)
        self.Morphology.addAction(self.Dilation.menuAction())
        self.Dilation.setObjectName(u"Dilation")
        self.Dilation.setTitle(
            QCoreApplication.translate("ImageProcess", u"膨胀核操作", None))

        self.Dilation_Rect = QAction(self.Dilation)
        self.Dilation.addAction(self.Dilation_Rect)
        self.Dilation_Rect.setObjectName(u"Dilation_Rect")
        self.Dilation_Rect.setText(
            QCoreApplication.translate("ImageProcess", u"矩形核膨胀操作", None))

        self.Dilation_Cross = QAction(self.Erodision)
        self.Dilation.addAction(self.Dilation_Cross)
        self.Dilation_Cross.setObjectName(u"Dilation_Cross")
        self.Dilation_Cross.setText(
            QCoreApplication.translate("ImageProcess", u"交叉核膨胀操作", None))

        self.Dilation_Ellipse = QAction(self.Dilation)
        self.Dilation.addAction(self.Dilation_Ellipse)
        self.Dilation_Ellipse.setObjectName(u"Dilation_Ellipse")
        self.Dilation_Ellipse.setText(
            QCoreApplication.translate("ImageProcess", u"椭圆核膨胀操作", None))

        self.OpenOperation = QMenu(self.Morphology)
        self.Morphology.addAction(self.OpenOperation.menuAction())
        self.OpenOperation.setObjectName(u"OpenOperation")
        self.OpenOperation.setTitle(
            QCoreApplication.translate("ImageProcess", u"开运算", None))

        self.Open_Rect = QAction(self.OpenOperation)
        self.OpenOperation.addAction(self.Open_Rect)
        self.Open_Rect.setObjectName(u"Open_Rect")
        self.Open_Rect.setText(
            QCoreApplication.translate("ImageProcess", u"矩形核开运算", None))

        self.Open_Cross = QAction(self.OpenOperation)
        self.OpenOperation.addAction(self.Open_Cross)
        self.Open_Cross.setObjectName(u"Open_Cross")
        self.Open_Cross.setText(
            QCoreApplication.translate("ImageProcess", u"交叉核开运算", None))

        self.Open_Ellipse = QAction(self.OpenOperation)
        self.OpenOperation.addAction(self.Open_Ellipse)
        self.Open_Ellipse.setObjectName(u"Open_Ellipse")
        self.Open_Ellipse.setText(
            QCoreApplication.translate("ImageProcess", u"椭圆核开运算", None))

        self.CloseOperation = QMenu(self.Morphology)
        self.Morphology.addAction(self.CloseOperation .menuAction())
        self.CloseOperation .setObjectName(u"CloseOperation ")
        self.CloseOperation .setTitle(
            QCoreApplication.translate("ImageProcess", u"闭运算", None))

        self.Close_Rect = QAction(self.CloseOperation)
        self.CloseOperation.addAction(self.Close_Rect)
        self.Close_Rect.setObjectName(u"Open_Rect")
        self.Close_Rect.setText(
            QCoreApplication.translate("ImageProcess", u"矩形核闭运算", None))

        self.Close_Cross = QAction(self.CloseOperation)
        self.CloseOperation.addAction(self.Close_Cross)
        self.Close_Cross.setObjectName(u"Open_Cross")
        self.Close_Cross.setText(
            QCoreApplication.translate("ImageProcess", u"交叉核闭运算", None))

        self.Close_Ellipse = QAction(self.CloseOperation)
        self.CloseOperation.addAction(self.Close_Ellipse)
        self.Close_Ellipse.setObjectName(u"Open_Ellipse")
        self.Close_Ellipse.setText(
            QCoreApplication.translate("ImageProcess", u"椭圆核闭运算", None))

    def face_list_init(self,ImageProcess):
        """
               初始化人像处理菜单（美白、磨皮、大眼、瘦脸、美妆）。

               参数:
                   ImageProcess (QMainWindow): 主窗口对象
               """
        self.Facial = QMenu(self.MyMenu)
        self.Facial.setObjectName(u"Facial")
        self.Facial.setTitle(QCoreApplication.translate("ImageProcess", u"人像处理", None))

        self.skin_whiten = QAction(self.Facial)
        self.skin_whiten.setObjectName(u"skin_whiten")
        self.skin_whiten.setText(QCoreApplication.translate("ImageProcess", u"美白", None))
        self.Facial.addAction(self.skin_whiten)

        self.skin_smooth = QAction(self.Facial)
        self.skin_smooth.setObjectName(u"skin_smooth")
        self.skin_smooth.setText(QCoreApplication.translate("ImageProcess", u"磨皮", None))
        self.Facial.addAction(self.skin_smooth)

        self.eye_bigger = QAction(self.Facial)
        self.eye_bigger.setObjectName(u"eye_bigger")
        self.eye_bigger.setText(QCoreApplication.translate("ImageProcess", u"大眼", None))
        self.Facial.addAction(self.eye_bigger)

        self.face_thinner = QAction(self.Facial)
        self.face_thinner.setObjectName(u"face_thinner")
        self.face_thinner.setText(QCoreApplication.translate("ImageProcess", u"瘦脸", None))
        self.Facial.addAction(self.face_thinner)

        self.make_up = QAction(self.Facial)
        self.make_up.setObjectName(u"make_up")
        self.make_up.setText(QCoreApplication.translate("ImageProcess", u"美妆", None))
        self.Facial.addAction(self.make_up)



    def style_list_init(self,  ImageProcess):
        self.Style = QMenu(self.MyMenu)
        self.Style.setObjectName(u"Style")
        self.Style.setTitle(QCoreApplication.translate("ImageProcess", u"风格处理", None))

        self.People = QMenu(self.Style)
        self.Style.addAction(self.People.menuAction())
        self.People.setObjectName(u"People")
        self.People.setTitle(
            QCoreApplication.translate("ImageProcess", u"人像", None))

        self.candy  = QAction(self.People)
        self.candy.setObjectName(u"candy")
        self.candy.setText(QCoreApplication.translate("ImageProcess", u"candy", None))
        self.People.addAction(self.candy)

        self.mosaic=  QAction(self.People)
        self.mosaic.setObjectName(u"mosaic")
        self.mosaic.setText(QCoreApplication.translate("ImageProcess", u"mosaic", None))
        self.People.addAction(self.mosaic)

        self.rain_princess =  QAction(self.People)
        self.rain_princess.setObjectName(u"rain_princess")
        self.rain_princess.setText(QCoreApplication.translate("ImageProcess", u"rain_princess", None))
        self.People.addAction(self.rain_princess)

        self.udnie =  QAction(self.People)
        self.udnie.setObjectName(u"udnie")
        self.udnie.setText(QCoreApplication.translate("ImageProcess", u"udnie", None))
        self.People.addAction(self.udnie)

        self.scenery = QMenu(self.Style)
        self.Style.addAction(self.scenery.menuAction())
        self.scenery.setObjectName(u"scenery")
        self.scenery.setTitle(
            QCoreApplication.translate("ImageProcess", u"风景", None))


        self.fu_shi_hui  = QAction(self.scenery)
        self.fu_shi_hui.setObjectName(u"fu_shi_hui")
        self.fu_shi_hui.setText(QCoreApplication.translate("ImageProcess", u"浮世绘", None))
        self.scenery.addAction(self.fu_shi_hui)

        self.starry_night= QAction(self.scenery)
        self.starry_night.setObjectName(u"starry_night")
        self.starry_night.setText(QCoreApplication.translate("ImageProcess", u"starry_night", None))
        self.scenery.addAction(self.starry_night)

        self.picasso=  QAction(self.scenery)
        self.picasso.setObjectName(u"picasso")
        self.picasso.setText(QCoreApplication.translate("ImageProcess", u"picasso", None))
        self.scenery.addAction(self.picasso)

        self.cuphead=  QAction(self.scenery)
        self.cuphead.setObjectName(u"cuphead")
        self.cuphead.setText(QCoreApplication.translate("ImageProcess", u"cuphead", None))
        self.scenery.addAction(self.cuphead)

        self.JoJo=  QAction(self.scenery)
        self.JoJo.setObjectName(u"JoJo")
        self.JoJo.setText(QCoreApplication.translate("ImageProcess", u"JoJo", None))
        self.scenery.addAction(self.JoJo)

        self.anime = QAction(self.scenery)
        self.anime.setObjectName(u"anime")
        self.anime.setText(QCoreApplication.translate("ImageProcess", u"anime", None))
        self.scenery.addAction(self.anime)

        self.mc=  QAction(self.scenery)
        self.mc.setObjectName(u"mc")
        self.mc.setText(QCoreApplication.translate("ImageProcess", u"mc", None))
        self.scenery.addAction(self.mc)





    def loadImage(self, image_path, label):
        """
              加载图像并设置到指定 QLabel 控件中。

              参数:
                  image_path (str): 图像路径
                  label (QLabel): 要显示图像的 QLabel 控件
              """
        pixmap = QPixmap(image_path)

        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))
        label.image = pixmap
        label.storage = pixmap


    def addImage(self):
        """
               创建图像显示控件（左右两个 HoverLabel），并设置主布局。
               """
        # 创建左右两个标签用于展示图片
        image_original = QPixmap('')
        image_modified = QPixmap('')


        self.label_original = HoverLabel(image_original, self)
        self.label_modified = HoverLabel(image_modified, self)

        hbox = QHBoxLayout()
        hbox.addWidget(self.label_original)
        hbox.addWidget(self.label_modified)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(hbox)


