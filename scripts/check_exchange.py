import numpy as np
from PIL import Image  # 用PIL读取/保存图片，兼容.jpeg格式
def process_array(arr):
    """对数组进行：指定列取负 + 前后半部分交换（全局统一触发）"""
    if should_process:  # 使用全局布尔变量，不再单独判断概率
        d = arr.shape[-1]  # 取最后一维（适配state(16,)/(14,)、actions(30,16)）
        # print(f"!!!!!!{d}")
        half_d = d // 2
        arr_processed = arr.copy()
        
        # 0-(half_d-1)、(half_d+1)-(d-1)列取负（适配任意维度，用...兼容多维数组）
        arr_processed[..., :half_d-1] = -arr_processed[..., :half_d-1]
        arr_processed[..., half_d:d-1] = -arr_processed[..., half_d:d-1]
        
        # 前后半部分交换（沿最后一维拼接）
        arr_processed = np.concatenate(
            [arr_processed[..., half_d:], arr_processed[..., :half_d]],
            axis=-1
        )
        return arr_processed
    else:
        return arr  # 不处理，返回原数组

def flip_image(image):
    """对图像进行水平翻转（全局统一触发，与数组处理同步）"""
    if should_process:  # 使用全局布尔变量，与数组处理保持一致
        return np.fliplr(image)  # 适配(360,640,3)图像格式，水平翻转
    else:
        return image  # 不翻转，返回原图像
def process_jpeg_image(img_path, save_path="flipped_rgb_image.jpeg"):
    """
    处理.jpeg图片：读取→转为HWC格式→水平翻转→确保RGB→保存
    :param img_path: 输入.jpeg图片的路径（如 "./test.jpeg"）
    :param save_path: 翻转后图片的保存路径
    :return: 无返回值，直接保存图片
    """
    # 步骤1：读取.jpeg图片，并转为NumPy数组（自动得到HWC格式）
    # Image.open() 读取图片，convert("RGB") 确保转为RGB格式（排除灰度图/透明通道干扰）
    img_pil = Image.open(img_path).convert("RGB")
    img_hwc = np.array(img_pil)  # 转为NumPy数组，形状为 (H, W, C)，C=3（RGB）
    print(f"原始图片形状（HWC）：{img_hwc.shape}")
    print(f"原始图片通道数：{img_hwc.shape[-1]}（RGB格式）")

    # 步骤2：使用np.fliplr进行水平翻转（左右翻转）
    img_hwc_flipped = np.fliplr(img_hwc)
    print(f"水平翻转后图片形状（HWC）：{img_hwc_flipped.shape}")

    # 步骤3：将翻转后的NumPy数组转回PIL图像（确保保存为RGB格式）
    img_flipped_pil = Image.fromarray(img_hwc_flipped.astype(np.uint8))  # 转为uint8类型（图片像素标准格式）

    # 步骤4：保存翻转后的.jpeg图片
    img_flipped_pil.save(save_path)
    print(f"翻转后的RGB图片已保存至：{save_path}")

# ===================== 调用示例（修改你的图片路径即可） =====================
if __name__ == "__main__":
    # 替换为你的.jpeg图片路径（绝对路径/相对路径均可）
    input_img_path = "/wx-mix01/sppro/permanent/yuanzhang10/rlds_data/for_rsluo/0.jpeg"  # 例如："C:/Users/xxx/Desktop/test.jpeg"
    # 可选：自定义保存路径
    output_img_path = "/wx-mix01/sppro/permanent/yuanzhang10/rlds_data/for_rsluo/0_exchange.jpeg"
    
    # 执行处理
    process_jpeg_image(input_img_path, output_img_path)

    d = 16
    arr_processed=np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    half_d = d//2
    arr_processed[..., :half_d-1] = -arr_processed[..., :half_d-1]
    arr_processed[..., half_d:d-1] = -arr_processed[..., half_d:d-1]

    # 前后半部分交换（沿最后一维拼接）
    arr_processed = np.concatenate(
        [arr_processed[..., half_d:], arr_processed[..., :half_d]],
        axis=-1
    )
    print(arr_processed)