from PyQt5.QtCore import Qt
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtWidgets import  QWidget,   QLabel, QVBoxLayout
class TooltipWidget(QWidget):
    def __init__(self, parent=None):
        super(TooltipWidget, self).__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.hide()  # 默认隐藏提示框

        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)  # 移除边距

        # 布局和标签用于显示文本
        self.text_label = QLabel(self)
        font = QFont("Arial", 10, QFont.Bold)
        self.text_label.setFont(font)
        # 设置文本居中
        self.text_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_label)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

    def set_text(self, text):
        self.text_label.setText(text)

    def set_pixmap(self, pixmap):
        self.image_label.setPixmap(pixmap)

    def move_to(self, pos):
        # 移动提示框到指定位置
        self.move(pos)

    def scale_pixmap_blocky(self,original_image, scale_factor):
        # 计算新图像的大小
        new_width = original_image.width() * scale_factor
        new_height = original_image.height() * scale_factor

        # 创建一个新的 QImage
        scaled_image = QImage(new_width, new_height, QImage.Format_ARGB32)
        scaled_image.fill(Qt.transparent)  # 用透明色填充新图像

        # 复制原始图像的每个像素到新的图像上，放大 scale_factor 倍
        for y in range(original_image.height()):
            for x in range(original_image.width()):
                pixel_color = original_image.pixel(x, y)
                for sy in range(scale_factor):
                    for sx in range(scale_factor):
                        scaled_x = x * scale_factor + sx
                        scaled_y = y * scale_factor + sy
                        scaled_image.setPixel(scaled_x, scaled_y, pixel_color)

                        # 如果需要 QPixmap，可以从 QImage 转换
        scaled_pixmap = QPixmap.fromImage(scaled_image)

        return scaled_pixmap

    def show_tooltip(self, text, cursor_pos, widget_pos, original_pixmap):
        # 将QPixmap转换为QImage
        original_image = original_pixmap.toImage()
        # cursor_pos是相对于HoverLabel的坐标
        crop_rect = QRect(cursor_pos.x(), cursor_pos.y(), 5, 5)

        # 确保裁剪区域不越界
        if crop_rect.right() > original_image.width():
            crop_rect.setRight(original_image.width())

        if crop_rect.bottom() > original_image.height():
            crop_rect.setBottom(original_image.height())

        cropped_image = original_image.copy(crop_rect)

        scaled_image = self.scale_pixmap_blocky(cropped_image, 15)

        # 设置文本和图像
        self.set_text(text)
        self.set_pixmap(scaled_image)
        # 移动到指定位置并显示
        self.move_to(widget_pos)
        self.show()

    def hide_tooltip(self):
        # 隐藏提示框
        self.hide()
