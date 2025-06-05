import os
import numpy as np
from tensorflow import keras
from PIL import Image

def predict_folder(model_path, input_dir, output_dir, tile_size=128):
    """
    对 512×512 PNG 图像做超分辨率预测（128×128→512×512），
    每个 128×128 子块输出 512×512，然后拼出一个 2048×2048 的大图保存。
    """
    model = keras.models.load_model(model_path, compile=False)
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(input_dir):
        if not fname.lower().endswith('.png'):
            continue

        img_path = os.path.join(input_dir, fname)
        img = Image.open(img_path).convert('RGB')
        arr = np.array(img)             # arr.shape == (512, 512, 3)
        h, w, _ = arr.shape

        # 计算整张图能切多少 128×128
        nh = h // tile_size             # 512//128 = 4
        nw = w // tile_size             # 4
        if nh == 0 or nw == 0:
            print(f"跳过 {fname}: 图像 < {tile_size}")
            continue

        # 由于每个 128×128 输入→输出 512×512，拼出来的完整尺寸是 (nh*512)×(nw*512)
        full_out_h = nh * 512
        full_out_w = nw * 512
        out_arr = np.zeros((full_out_h, full_out_w, 3), dtype=np.uint8)

        for i in range(nh):
            for j in range(nw):
                y0, y1 = i*tile_size, (i+1)*tile_size
                x0, x1 = j*tile_size, (j+1)*tile_size
                tile = arr[y0:y1, x0:x1, :]   # tile.shape == (128,128,3)

                # 归一化并 batch 维度
                inp = tile.astype(np.float32)/255.0
                inp = np.expand_dims(inp, axis=0)  # (1,128,128,3)

                # 模型直接输出 (1,512,512,3)，取 [0] 就是 (512,512,3)
                pred = model.predict(inp, verbose=0)[0]  # pred.shape == (512,512,3)

                # 把 pred 从 [0,1] 恢复到 [0,255]
                pred_uint8 = (pred * 255.0).clip(0,255).astype(np.uint8)  # (512,512,3)

                # 拼回大图：第 i 行第 j 列，放在 [i*512 : (i+1)*512,  j*512 : (j+1)*512]
                out_arr[i*512:(i+1)*512, j*512:(j+1)*512, :] = pred_uint8

        # 把拼好的 2048×2048 保存
        out_img = Image.fromarray(out_arr)
        save_path = os.path.join(output_dir, fname.replace('.png','_sr.png'))
        out_img.save(save_path)
        print(f"Saved super-resolved: {save_path}")

if __name__ == '__main__':
    model_file    = 'D:/cey/CNTs_detials/ML_U_net/model/unet_sr_model.keras'
    input_folder  = 'D:/cey/CNTs_detials/ML_U_net/raw'      # 里面放 512×512 的 PNG
    output_folder = 'D:/cey/CNTs_detials/ML_U_net/predict'
    predict_folder(model_file, input_folder, output_folder, tile_size=128)
