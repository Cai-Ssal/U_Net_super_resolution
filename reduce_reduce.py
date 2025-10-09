import os
import numpy as np
from PIL import Image
import random


def add_white_spots_before_noise(img, num_spots=5, radius=3):
    """
    在图像上随机添加白点及其周围区域（灰度255），并返回修改后的图像和坐标。
    - img: PIL 灰度图 (64x64)
    - num_spots: 随机点的数量
    - radius: 每个点影响的半径像素
    返回修改后的图像和记录的坐标列表
    """
    arr = np.array(img, dtype=np.uint8)
    h, w = arr.shape
    coords = []

    for _ in range(num_spots):
        # 随机选择坐标
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        coords.append((x, y))

        # 将半径范围内像素全部置为255
        y_min, y_max = max(0, y - radius), min(h, y + radius + 1)
        x_min, x_max = max(0, x - radius), min(w, x + radius + 1)
        arr[y_min:y_max, x_min:x_max] = 255

    return Image.fromarray(arr, mode='L'), coords


def batch_downscale_and_upscale_with_noise(
    input_dir,
    output_dir,
    hr_suffix='_hr.png',
    lr_suffix='_lr.png',
    src_size=(512, 512),
    dst_size=(64, 64),
    noise_std=5,
    overwrite=False,
    upsample_method=Image.BICUBIC,
    num_white_spots=5,
    white_spot_radius=3
):
    """
    将 input_dir 下所有以 hr_suffix 结尾、分辨率为 src_size 的图片：
      1) 转灰度并降采样到 dst_size
      2) 在噪声前添加白点
      3) 添加高斯噪声
      4) 插值放大回 src_size
      5) 保存为灰度图，文件名以 lr_suffix 结尾
    """
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        rel_path = os.path.relpath(root, input_dir)
        save_dir = os.path.join(output_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        for fname in files:
            if not fname.endswith(hr_suffix):
                continue

            hr_path = os.path.join(root, fname)
            try:
                with Image.open(hr_path) as img:
                    if img.size != src_size:
                        print(f"跳过（分辨率不符）：{hr_path} 大小 {img.size}")
                        continue

                    # 转为灰度并降采样到 dst_size
                    lr_small = img.convert('L').resize(dst_size, resample=Image.LANCZOS)
                    arr = np.array(lr_small).astype(np.float32)

                    # ⭐ 在噪声前添加随机白点
                    lr_small, coords = add_white_spots_before_noise(lr_small, num_spots=num_white_spots, radius=white_spot_radius)
                    print(f"{fname} 随机白点坐标：{coords}")

                    # 添加高斯噪声
                    noise = np.random.normal(loc=0.0, scale=noise_std, size=arr.shape)
                    arr_noisy = arr + noise
                    arr_noisy = np.clip(arr_noisy, 0, 255).astype(np.uint8)

                    # 将 noisy 小图插值放大回 src_size
                    img_noisy_small = Image.fromarray(arr_noisy, mode='L')
                    img_up = img_noisy_small.resize(src_size, resample=upsample_method)

                    # 构造输出文件名和路径
                    base = fname[:-len(hr_suffix)]
                    lr_fname = base + lr_suffix
                    lr_path = os.path.join(save_dir, lr_fname)

                    # 保存或跳过
                    if not overwrite and os.path.exists(lr_path):
                        print(f"已存在，跳过：{lr_path}")
                    else:
                        img_up.save(lr_path)
                        print(f"已保存：{lr_path}（尺寸 {img_up.size}，seed 随机）")

            except Exception as e:
                print(f"处理失败：{hr_path}，原因：{e}")


if __name__ == '__main__':
    input_folder = 'D:/cey/CNTs_detials/ML_U_net/dev/hr'
    output_folder = 'D:/cey/CNTs_detials/ML_U_net/dev/lr'
    batch_downscale_and_upscale_with_noise(
        input_dir=input_folder,
        output_dir=output_folder,
        hr_suffix='_hr.png',
        lr_suffix='_lr.png',
        src_size=(512, 512),
        dst_size=(64, 64),
        noise_std=10,             # 噪声强度
        overwrite=False,
        upsample_method=Image.BICUBIC,
        num_white_spots = random.randint(3, 10),        # 随机白点数量
        white_spot_radius = random.randint(2, 5)       # 白点半径
    )
