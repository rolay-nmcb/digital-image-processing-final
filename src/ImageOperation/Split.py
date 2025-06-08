import numpy as np
import cv2

"""
Split.py - 图像分割与阈值处理模块

提供多种图像分割方法：
- 区域生长（regional_growth_algorithm）
- 分裂合并（regional_split_algorithm）
- K-means聚类（kmeans_algorithm）
- 分水岭算法（dis_waterhold_algorithm）
- Sobel + 分水岭组合（sobel_waterhold_algorithm）
- 多种阈值分割方法（全局、局部、多阈值）

适用于图像语义分割、对象识别、特征提取等任务。
"""

def getGrayDiff(image, currentPoint, tmpPoint):
    """
       计算两个像素点之间的灰度差值。

       参数:
           image (numpy.ndarray): 输入图像（灰度图）
           currentPoint (tuple): 当前像素坐标 (x, y)
           tmpPoint (tuple): 比较像素坐标 (x, y)

       返回:
           int: 两个像素的灰度差绝对值
       """
    # 求两个像素的距离
    return abs(int(image[currentPoint[0], currentPoint[1]]) - int(image[tmpPoint[0], tmpPoint[1]]))

# #区域生长算法
# def seed_regional_growth(img, seeds, thresh=10):
#     """
#         区域生长算法核心实现：从给定种子点开始扩展相似区域。
#
#         参数:
#             img (numpy.ndarray): 输入图像（灰度图）
#             seeds (list of tuple): 种子点列表 [(x1,y1), (x2,y2), ...]
#             thresh (float): 灰度差阈值，用于判断是否合并邻域点
#
#         返回:
#             numpy.ndarray: 标记后的区域图像（每个区域用不同整数标记）
#         """
#     height, weight = img.shape
#     seedMark = np.zeros(img.shape)
#     seedList = []
#     for seed in seeds:
#         if (0 < seed[0] < height and 0 < seed[1] < weight): seedList.append(seed)
#     label = 1  # 种子位置标记
#     connects = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]  # 8 邻接连通
#     while (len(seedList) > 0):  # 如果列表里还存在点
#         currentPoint = seedList.pop(0)  # 将最前面的那个抛出
#         seedMark[currentPoint[0], currentPoint[1]] = label  # 将对应位置的点标记为 1
#         for i in range(8):  # 对这个点周围的8个点一次进行相似性判断
#             tmpX = currentPoint[0] + connects[i][0]
#             tmpY = currentPoint[1] + connects[i][1]
#             if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:  # 是否超出限定阈值
#                 continue
#             grayDiff = getGrayDiff(img, currentPoint, (tmpX, tmpY))  # 计算灰度差
#             if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
#                 seedMark[tmpX, tmpY] = label
#                 seedList.append((tmpX, tmpY))
#     return seedMark
#
# def regional_growth_algorithm(img,thresh=10):
#     """
#         自动执行区域生长算法，基于图像顶部阈值提取自动选择种子点。
#
#         参数:
#             img (numpy.ndarray): 输入图像（灰度或彩色）
#             thresh (float): 区域生长相似性阈值
#
#         返回:
#             numpy.ndarray: 分割后的彩色标记图像（BGR格式）
#         """
#     if len(img.shape) != 2:
#         img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     imgBlur = cv2.blur(img, (3, 3))
#     _, imgTop = cv2.threshold(imgBlur, 250, 255, cv2.THRESH_BINARY)
#     nseeds, labels, stats, centroids = cv2.connectedComponentsWithStats(imgTop)
#     seeds = centroids.astype(int)  # 获得质心像素作为种子点
#     result = seed_regional_growth(img, seeds,thresh)
#     result = result.astype(np.float32)
#
#     return cv2.cvtColor(result,cv2.COLOR_GRAY2BGR)
#
def SplitMerge(src, dst, h, w, h0, w0, maxMean=80, minVar = 10, cell=4):
    """
        图像四叉树分裂与合并算法核心实现。

        参数:
            src (numpy.ndarray): 原始图像（灰度）
            dst (numpy.ndarray): 输出图像（分割结果）
            h (int): 当前区域高度
            w (int): 当前区域宽度
            h0 (int): 起始行索引
            w0 (int): 起始列索引
            maxMean (float): 区域均值上限，用于判断是否为目标区域
            minVar (float): 区域方差下限，用于判断是否为目标区域
            cell (int): 最小分块大小

        返回:
            None（原地修改 dst）
        """
    win = src[h0: h0 + h, w0: w0 + w]
    mean = np.mean(win)  # 窗口区域的均值
    var = np.std(win, ddof=1)  # 窗口区域的标准差，无偏样本标准差

    if (mean < maxMean) and (var > minVar) and (h < 2 * cell) and (w < 2 * cell):
        # 该区域满足谓词逻辑条件，判为目标区域，设为白色
        dst[h0:h0 + h, w0:w0 + w] = 255  # 白色
    else:  # 该区域不满足谓词逻辑条件
        if (h > cell) and (w > cell):  # 区域能否继续分拆？继续拆
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0, w0, maxMean, minVar, cell)
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0, w0 + (w + 1) // 2, maxMean, minVar, cell)
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0 + (h + 1) // 2, w0, maxMean, minVar, cell)
            SplitMerge(src, dst, (h + 1) // 2, (w + 1) // 2, h0 + (h + 1) // 2, w0 + (w + 1) // 2, maxMean, minVar,
                       cell)

def regional_split_algorithm(img):
    """
       使用分裂-合并策略进行图像分割（自适应划分目标区域）

       参数:
           img (numpy.ndarray): 输入图像（灰度或彩色）

       返回:
           numpy.ndarray: 分割后的二值图像（白色为检测到的目标区域）
       """
    hImg, wImg = img.shape[:2]
    maxMean = 80  # 均值上界
    minVar = 10  # 标准差下界
    src = img.copy()
    dst = np.zeros_like(img)
    SplitMerge(src, dst, hImg, wImg, 0, 0, maxMean, minVar, 2)
    return dst


def auto_region_growing(image, threshold=80):
    """
    自动区域生长算法 - 仅需输入图像和阈值

    参数:
        image (numpy.ndarray): 输入图像，彩色(BGR)或灰度
        threshold (int): 区域生长的相似性阈值，默认80

    返回:
        numpy.ndarray: 分割结果，背景为0，前景为255
    """
    # 确保输入是有效的numpy数组
    image = np.asarray(image)

    # 自动选择种子点（使用K-means聚类）
    seeds = _auto_select_seeds(image)

    # 执行区域生长
    return _region_growing(image, seeds, threshold)


def _auto_select_seeds(image, k=3):
    """自动选择种子点（内部函数）"""
    # 确定图像类型（彩色或灰度）
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    # 重塑图像为像素数组
    if is_color:
        pixels = image.reshape((-1, 3)).astype(np.float32)
    else:
        pixels = image.reshape((-1, 1)).astype(np.float32)

    # 应用K-means聚类
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # 重塑标签回图像形状
    labels = labels.reshape(image.shape[:2])

    # 计算每个聚类的质心作为种子点
    seeds = []
    for i in range(k):
        # 找到属于当前聚类的所有像素
        mask = (labels == i).astype(np.uint8) * 255

        # 计算质心
        moments = cv2.moments(mask)
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            seeds.append((cx, cy))

    return seeds


def _region_growing(image, seeds, threshold):
    """执行区域生长（内部函数）"""
    # 确定图像类型（彩色或灰度）
    is_color = len(image.shape) == 3 and image.shape[2] == 3

    # 创建标记图像
    height, width = image.shape[:2]
    region = np.zeros((height, width), dtype=np.uint8)

    # 创建访问标记
    visited = np.zeros((height, width), dtype=bool)

    # 4-邻域
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 对每个种子点执行区域生长
    for seed in seeds:
        x, y = seed

        # 检查种子点有效性
        if x < 0 or x >= width or y < 0 or y >= height:
            continue

        # 获取种子点像素值
        if is_color:
            seed_value = np.array(image[y, x], dtype=np.float32)
        else:
            seed_value = float(image[y, x])

        # 初始化队列
        queue = [(x, y)]
        visited[y, x] = True

        # 区域生长循环
        while queue:
            cx, cy = queue.pop(0)

            # 将当前像素添加到区域
            region[cy, cx] = 255

            # 检查所有邻居
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy

                # 检查边界
                if 0 <= nx < width and 0 <= ny < height:
                    # 检查是否已访问
                    if not visited[ny, nx]:
                        # 计算相似度
                        if is_color:
                            neighbor = image[ny, nx]
                            diff = np.sqrt(np.sum((neighbor - seed_value) ** 2))
                        else:
                            neighbor = image[ny, nx]
                            diff = abs(neighbor - seed_value)

                        # 如果相似，则添加到区域
                        if diff <= threshold:
                            visited[ny, nx] = True
                            queue.append((nx, ny))

    return region

def kmeans_algorithm(img, k=3):
    """
    K-means聚类算法（兼容灰度和彩色图像）

    参数:
        img: 输入图像（灰度或BGR）
        k: 聚类数量（默认3）

    返回:
        img_kmean: 聚类结果（与输入同通道数）
    """
    # 检查输入是否为单通道（灰度）
    if len(img.shape) == 2:
        # 灰度图像处理
        data_pixel = np.float32(img.reshape((-1, 1)))  # 转为Nx1的二维数组
        is_gray = True
    else:
        # 彩色图像处理
        data_pixel = np.float32(img.reshape((-1, 3)))  # 转为Nx3的二维数组
        is_gray = False

    # K-means参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 执行K-means
    _, labels, centers = cv2.kmeans(data_pixel, k, None, criteria, 10, flags)

    # 转换为uint8并重建图像
    centers_uint = np.uint8(centers)
    clustered_data = centers_uint[labels.flatten()]

    # 根据输入恢复形状
    if is_gray:
        img_kmean = clustered_data.reshape(img.shape)  # 单通道
    else:
        img_kmean = clustered_data.reshape(img.shape)  # 三通道

    return img_kmean


def dis_waterhold_algorithm(img):
    """
    分水岭算法（兼容灰度和彩色图像）

    参数:
        img: 输入图像（灰度或BGR）

    返回:
        result: 分水岭分割结果（与输入同通道数，轮廓标记为绿色）
    """
    # 检查输入是否为单通道（灰度）
    if len(img.shape) == 2:
        # 灰度图像处理
        gray = img
        is_gray = True
        # 转换为伪彩色用于后续处理（因为分水岭需要3通道输入）
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        # 彩色图像处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_gray = False
        img_color = img.copy()

    # 阈值分割（使用OTSU自动阈值）
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 形态学操作（去噪）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换确定前景
    distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(distance, 0.1 * distance.max(), 255, cv2.THRESH_BINARY)
    sure_fg = np.uint8(sure_fg)

    # 未知区域（背景减去前景）
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通域标记
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 背景标记为1
    markers[unknown == 255] = 0  # 未知区域标记为0

    # 分水岭算法
    markers = cv2.watershed(img_color, markers)

    # 创建结果图像
    if is_gray:
        # 灰度结果：将轮廓标记为白色（255）
        result = gray.copy()
        result[markers == -1] = 255
    else:
        # 彩色结果：将轮廓标记为绿色（BGR中的[0,255,0]）
        result = img.copy()
        result[markers == -1] = [0, 255, 0]

    return result


def sobel_waterhold_algorithm(img, grad_thresh=0.2, dist_thresh=0.1):
    """
    Sobel梯度+分水岭算法（兼容灰度和彩色图像）

    参数:
        img: 输入图像（灰度或BGR）
        grad_thresh: 梯度阈值比例（默认0.2）
        dist_thresh: 距离变换阈值比例（默认0.1）

    返回:
        result: 分水岭分割结果（与输入同通道数，轮廓标记为绿色/白色）
    """
    # 检查输入是否为单通道（灰度）
    if len(img.shape) == 2:
        gray = img
        is_gray = True
        # 转换为伪彩色用于分水岭
        img_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        is_gray = False
        img_color = img.copy()

    # 高斯模糊（减少噪声影响）
    blur = cv2.GaussianBlur(gray, (5, 5), 10.0)

    # Sobel梯度计算
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0)  # x方向梯度
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1)  # y方向梯度
    grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 基于梯度的阈值分割
    _, thresh = cv2.threshold(grad_norm, grad_thresh * grad_norm.max(), 255, cv2.THRESH_BINARY)

    # 形态学操作（去噪）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定背景区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 距离变换确定前景
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, dist_thresh * dist_transform.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    # 未知区域（背景减去前景）
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 连通域标记
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # 背景标记为1
    markers[unknown == 255] = 0  # 未知区域标记为0

    # 分水岭算法
    markers = cv2.watershed(img_color, markers)

    # 创建结果图像
    if is_gray:
        # 灰度结果：轮廓标记为白色（255），其他区域保留原灰度值
        result = gray.copy()
        result[markers == -1] = 255
    else:
        # 彩色结果：轮廓标记为绿色（[0,255,0]），其他区域保留原色
        result = img.copy()
        result[markers == -1] = [0, 255, 0]

    return result


def global_threshold(src, thresh_value=127, use_otsu=False, blur_kernel=(5, 5)):
    """
    全局阈值处理

    参数:
    src (numpy.ndarray): 输入图像，灰度图或彩色图
    thresh_value (int): 固定阈值，仅在use_otsu=False时有效
    use_otsu (bool): 是否使用Otsu自动阈值
    blur_kernel (tuple): 高斯滤波核大小

    返回:
    numpy.ndarray: 阈值处理后的二值图像
    """
    # 确保输入图像为灰度图
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()

    # 应用高斯滤波减少噪声
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 全局阈值处理
    if use_otsu:
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, thresh = cv2.threshold(blurred, thresh_value, 255, cv2.THRESH_BINARY)

    return thresh


def local_threshold(src, method='mean', block_size=11, C=2, blur_kernel=(5, 5)):
    """
    局部阈值处理

    参数:
    src (numpy.ndarray): 输入图像，灰度图或彩色图
    method (str): 局部阈值方法，可选 'mean' 或 'gaussian'
    block_size (int): 邻域块大小（奇数）
    C (int): 从平均值或加权平均值中减去的常数
    blur_kernel (tuple): 高斯滤波核大小

    返回:
    numpy.ndarray: 阈值处理后的二值图像
    """
    # 确保输入图像为灰度图
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()

    # 应用高斯滤波减少噪声
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # 选择局部阈值方法
    if method.lower() == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method.lower() == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError("method参数必须为 'mean' 或 'gaussian'")

    # 局部阈值处理
    thresh = cv2.adaptiveThreshold(blurred, 255, adaptive_method, cv2.THRESH_BINARY, block_size, C)

    return thresh


def multi_threshold(src, num_classes=3, blur_kernel=(5, 5)):
    """
    多阈值处理

    参数:
    src (numpy.ndarray): 输入图像，灰度图或彩色图
    num_classes (int): 分类数（2-4之间）
    blur_kernel (tuple): 高斯滤波核大小

    返回:
    numpy.ndarray: 多阈值分割结果
    """
    if num_classes < 2 or num_classes > 4:
        raise ValueError("num_classes必须在2-4之间")

    # 确保输入图像为灰度图
    if len(src.shape) == 3:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    else:
        gray = src.copy()

    # 应用高斯滤波减少噪声
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    if num_classes == 2:
        # 双阈值（Otsu）
        ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        multi_thresh = np.zeros_like(gray)
        multi_thresh[gray < ret] = 0
        multi_thresh[gray >= ret] = 255

    elif num_classes == 3:
        # 三阈值（使用两次Otsu）
        ret1, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2, _ = cv2.threshold(255 - blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret2 = 255 - ret2

        multi_thresh = np.zeros_like(gray)
        multi_thresh[gray < ret1] = 0  # 背景
        multi_thresh[(gray >= ret1) & (gray < ret2)] = 127  # 中间区域
        multi_thresh[gray >= ret2] = 255  # 前景

    else:  # num_classes == 4
        # 四阈值（使用K-means聚类）
        pixels = blurred.reshape(-1, 1).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        multi_thresh = centers[labels.flatten()].reshape(blurred.shape)

    return multi_thresh


if __name__ == '__main__':
    img = cv2.imread('../1.png', 1)

    imgGrowth = auto_region_growing(img,80)
    cv2.imshow('OTSU', imgGrowth)
    key = cv2.waitKey(0)