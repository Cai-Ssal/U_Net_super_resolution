import tensorflow as tf
import numpy as np
import os

# 预计算 Bézier 参数 t
def get_t(n_steps):
    t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[:, None]
    one_minus_t = 1.0 - t
    return t, one_minus_t

# 计算二次 Bézier 曲线点
def bezier_curve_np(p0, p1, p2, t, one_minus_t):
    coords = (one_minus_t**2 * p0 +
              2 * one_minus_t * t * p1 +
              t**2 * p2)
    coords = np.round(coords).astype(np.int32)
    return np.clip(coords, 0, None)

# 在图像上绘制平行短线组
def draw_parallel_segments(img, occupied, ys, xs, grays,
                           max_lines=5, seg_length=16, spacing=4.0):
    n = len(ys)
    coords = np.stack([ys, xs], axis=1)

    tangents = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        if i == 0:
            d = coords[1] - coords[0]
        elif i == n - 1:
            d = coords[-1] - coords[-2]
        else:
            d = coords[i+1] - coords[i-1]
        norm = np.linalg.norm(d)
        tangents[i] = d / (norm + 1e-6)

    perps = np.stack([-tangents[:,1], tangents[:,0]], axis=1)

    for i, (y, x, gray) in enumerate(zip(ys, xs, grays)):
        num_lines = 1 + int(gray * (max_lines - 1))
        offsets = (np.arange(num_lines) - (num_lines - 1)/2) * spacing

        for off in offsets:
            center = np.array([y, x], dtype=np.float32) + off * perps[i]
            half = seg_length / 2.0
            p1 = center - tangents[i] * half
            p2 = center + tangents[i] * half
            y0, x0 = np.round(p1).astype(int)
            y1, x1 = np.round(p2).astype(int)

            steps = max(abs(y1 - y0), abs(x1 - x0)) + 1
            ys_line = np.linspace(y0, y1, steps).astype(int)
            xs_line = np.linspace(x0, x1, steps).astype(int)

            mask = (
                (ys_line >= 0) & (ys_line < img.shape[0]) &
                (xs_line >= 0) & (xs_line < img.shape[1])
            )
            img[ys_line[mask], xs_line[mask]] = gray
            occupied[ys_line[mask], xs_line[mask]] = True

# 单张图生成函数（允许起点和终点附近重叠，其他部分不可重叠或只允许小比例重叠）
def generate_single_np(img_size, n_curves, curve_length, diag_prob=0.6,
                       t=None, one_minus_t=None,
                       overlap_tol=0.06,   # 允许的中间重叠比例（0..1）
                       max_attempts_multiplier=200,  # 每条曲线尝试次数乘数
                       margin_frac=0.05,  # 起终点可允许重叠的比例（相对于曲线长度）
                       spacing=3.5):
    img = np.zeros((img_size, img_size), dtype=np.float32)
    occupied = np.zeros((img_size, img_size), dtype=bool)

    # 4 个角作为可能的起止点
    corners = np.array([[0, 0], [0, img_size-1], [img_size-1, 0], [img_size-1, img_size-1]], dtype=np.int32)
    r = np.random.rand()
    if r < diag_prob:
        idx = np.random.choice(4, size=2, replace=False)
        start, end = corners[idx]
    elif r < diag_prob + 0.2:
        edge_choices = [[0,1], [2,3], [0,2], [1,3]]
        pair = edge_choices[np.random.randint(len(edge_choices))]
        start, end = corners[pair]
    else:
        start = np.random.randint(0, img_size, size=2, dtype=np.int32)
        end = np.random.randint(0, img_size, size=2, dtype=np.int32)

    # 允许起点/终点附近 overlap 的点数
    margin = max(1, int(curve_length * float(margin_frac)))  # e.g. 5% 的曲线长度
    success = 0
    attempts = 0
    max_attempts = max(5000, int(n_curves * max_attempts_multiplier))

    while success < n_curves and attempts < max_attempts:
        attempts += 1
        base_gray = np.random.uniform(0.1, 0.9)
        n_mods = np.random.randint(1, 4)
        mods = []
        for _ in range(n_mods):
            seg_start = np.random.randint(0, curve_length)
            seg_len = np.random.randint(max(1, curve_length // 10), max(2, curve_length // 3))
            delta = np.random.uniform(-0.3, 0.3)
            mods.append((seg_start, seg_len, delta))

        ctrl = np.random.randint(img_size//6, 5*img_size//6 + 1, size=2, dtype=np.int32)
        coords = bezier_curve_np(start, ctrl, end, t, one_minus_t)
        ys, xs = coords[:,0], coords[:,1]

        # 如果中间区域太短导致 slice 为空，则允许（不检查 occupied）
        if len(ys[margin:-margin]) > 0:
            occ_slice = occupied[ys[margin:-margin], xs[margin:-margin]]
            occ_frac = float(np.mean(occ_slice))
            # 仅当中间被占比例超过阈值时跳过
            if occ_frac > overlap_tol:
                continue

        # 构造灰度变化
        grays = np.full(curve_length, base_gray, dtype=np.float32)
        for seg_start, seg_len, delta in mods:
            seg_end = min(curve_length, seg_start + seg_len)
            grays[seg_start:seg_end] = np.clip(grays[seg_start:seg_end] + delta, 0.0, 1.0)

        # 绘制平行短线曲线（传入 spacing）
        draw_parallel_segments(img, occupied, ys, xs, grays, spacing=spacing)
        success += 1

    # 如果尝试结束仍未达到期望曲线数，会返回已绘制的数量（success）
    return img

# 批量生成并保存（只保存去掉四边 edge_crop_width 后的中心图）
def generate_batch(img_size=612,
                   min_curves=30,
                   max_curves=80,
                   curve_length=200,
                   batch_size=10,
                   save_dir="./output",
                   diag_prob=0.6,
                   start_seed=10000,
                   edge_crop_width=50,
                   overlap_tol=0.1,
                   spacing=3.5):
    assert 10000 <= start_seed <= 99999, "start_seed must be a five-digit integer between 10000 and 99999"
    os.makedirs(save_dir, exist_ok=True)
    t, one_minus_t = get_t(curve_length)

    # 中心裁切坐标（去掉每边 edge_crop_width 后的中心区域）
    c = edge_crop_width
    y0, y1 = c, img_size - c  # slice y0:y1 gives height = img_size - 2*c
    x0, x1 = c, img_size - c  # same for width

    for i in range(batch_size):
        current_seed = start_seed + i
        if current_seed > 99999:
            current_seed = 10000 + (current_seed - 10000) % 90000
        np.random.seed(current_seed)
        tf.random.set_seed(current_seed)

        n_curves = np.random.randint(min_curves, max_curves+1)
        img_np = generate_single_np(img_size, n_curves, curve_length, diag_prob,
                                    t, one_minus_t,
                                    overlap_tol=overlap_tol,
                                    max_attempts_multiplier=200,
                                    margin_frac=0.05,
                                    spacing=spacing)

        # 裁切中心区域（不保存边缘部分）
        center_np = img_np[y0:y1, x0:x1].copy()  # 512x512 when img_size=612 and c=50

        # 保存裁切后的中心图（float -> uint8）
        crop_tf = tf.convert_to_tensor(center_np[..., None], dtype=tf.float32)
        crop_uint8 = tf.image.convert_image_dtype(crop_tf, tf.uint8, saturate=True)
        png_crop = tf.io.encode_png(crop_uint8)
        filename_crop = os.path.join(save_dir, f"curve_img_{i:04d}_hr.png")
        tf.io.write_file(filename_crop, png_crop)

        print(f"Saved {filename_crop} (seed {current_seed}) — requested {n_curves} curves, generated approx. {np.unique(center_np).size} greyscale values")

if __name__ == "__main__":
    if tf.config.list_physical_devices('GPU'):
        print("Using GPU for encoding.")
    else:
        print("GPU not found, using CPU for encoding.")
    generate_batch(
        img_size=712,
        min_curves=30,      # 增大最小曲线数量
        max_curves=80,      # 增大最大曲线数量
        curve_length=200,
        batch_size=3000,    # 注意：大 batch 与多曲线会很慢，先小规模测试
        save_dir="D:/cey/CNTs_detials/ML_U_net/test/hr",
        diag_prob=0.6,
        start_seed=37000,
        edge_crop_width=100,
        overlap_tol=0.15,   # 允许 6% 的中段重叠（可调）
        spacing=3.0         # 减小 spacing 可更密集
    )
