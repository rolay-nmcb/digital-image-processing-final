from PyQt5.QtCore import QPoint, pyqtSignal, QSize
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QMouseEvent, QColor, QPainter, QPen
from PyQt5.QtWidgets import QSizePolicy

from infoWidget import *

class HoverLabel(QLabel):
    cropPointsSelected = pyqtSignal(QPoint, QPoint)

    def __init__(self, pixmap, parent=None):
        super(HoverLabel, self).__init__(parent)
        self.image = pixmap
        self.storage = pixmap
        # 判断并缩放 pixmap
        scaled_pixmap = self.scale_pixmap_if_needed(pixmap)

        self.setPixmap(scaled_pixmap)
        # self.setPixmap(pixmap)
        # self.resize(512,512)
        # self.setFixedSize(512, 512)  # 固定 QLabel 的显示尺寸
        # self.setScaledContents(True)  # 让 pixmap 自动缩放到 QLabel 尺寸
        self.tooltip_widget = TooltipWidget()
        self.setMouseTracking(True)
        self.flag = False
        self.start_point = None

    def scale_pixmap_if_needed(self, pixmap, max_size=512):
        """
        如果 pixmap 超过 max_size，则等比缩放；否则保留原图。
        """
        if pixmap.isNull():
            return pixmap

        width = pixmap.width()
        height = pixmap.height()

        if width <= max_size and height <= max_size:
            return pixmap  # 不需要缩放

        # 等比缩放
        scaled_pixmap = pixmap.scaled(
            max_size, max_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation  # 可选：平滑缩放
        )
        return scaled_pixmap


    def open_show(self):
        self.flag = True

    def off_show(self):
        self.flag = False

    @pyqtSlot(QMouseEvent)
    def mouseMoveEvent(self, event):
        # 获取光标在 QLabel 上的位置
        cursor_pos = event.pos()
        global_pos = self.mapToGlobal(cursor_pos)
        offset = QPoint(5, 5)
        global_pos += offset
        tooltip_pos = global_pos + offset
        # 显示 tooltip，这通常会在鼠标悬停时自动发生
        if self.flag:
            self.tooltip_widget.show_tooltip(f"({cursor_pos.x()}, {cursor_pos.y()})",
                                             cursor_pos, tooltip_pos, self.storage)
            if self.start_point is not None:
                temp_pixmap = self.storage.copy()  # 每次都基于原始图像绘制
                self.draw_rectangle_on_image(self.start_point, cursor_pos, temp_pixmap)

    def draw_rectangle_on_image(self, start_point, cursor_pos, temp_pixmap):
        painter = QPainter(temp_pixmap)
        pen = QPen(QColor(0, 255, 0), 2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        rect = QRect(start_point, cursor_pos).normalized()
        painter.drawRect(rect)
        painter.end()
        self.setPixmap(temp_pixmap)

    def leaveEvent(self, event):
        # 当鼠标离开时隐藏提示框
        if self.flag:
            self.tooltip_widget.hide_tooltip()

    def mousePressEvent(self, event):
        if self.flag:
            if self.start_point is None:  # 如果没有选择起点，设置它
                self.start_point = event.pos()
                self.setCursor(Qt.CrossCursor)
            else:
                # 否则，设置终点并选择裁剪区域（或执行其他操作）
                end_point = event.pos()
                self.cropPointsSelected.emit(self.start_point, end_point)
                self.start_point = None  # 重置起点以便进行下一次选择
                self.setCursor(Qt.ArrowCursor)  # 重置鼠标光标
