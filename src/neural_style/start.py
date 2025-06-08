import os
import cv2
from  .neural_style_func import *

def process_image_with_style(image, model_name: str):
    """
    使用指定风格模型处理图像

    参数:
        image: 输入的图像数据 (numpy.ndarray)
        model_name: 风格模型名称 (不含扩展名)

    返回:
        styled_image: 风格化后的图像
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建 saved_models 文件夹路径
    models_dir = os.path.join(current_dir, 'saved_models')
    # 动态构建完整模型路径
    model_path = os.path.join(models_dir, f'{model_name}.pth')

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 使用指定模型进行图像风格化
    styled_image = stylize_image(image, model_path)
    return styled_image


if __name__ == '__main__':
    # 示例：加载图像并应用风格
    img_path = '../1.png'
    img = cv2.imread(img_path)

    if img is None:
        print(f"无法加载图像: {img_path}")
    else:
        try:
            # 调用入口函数，传入图像和模型名称
            result = process_image_with_style(img, 'mosaic')

            # 显示结果
            cv2.imshow('Styled Image', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"处理图像时出错: {e}")