import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

# ----------------------
# 性能优化设置（混合精度）
# ----------------------
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# 可选：避免一次性占满显存
gpus = tf.config.list_physical_devices('GPU')
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ----------------------
# 路径与参数配置（按需修改）
# ----------------------
LR_DIR        = 'D:/cey/CNTs_detials/ML_U_net/train/lr'
HR_DIR        = 'D:/cey/CNTs_detials/ML_U_net/train/hr'
VAL_LR_DIR    = 'D:/cey/CNTs_detials/ML_U_net/dev/lr'
VAL_HR_DIR    = 'D:/cey/CNTs_detials/ML_U_net/dev/hr'
TEST_LR_DIR   = 'D:/cey/CNTs_detials/ML_U_net/test/lr'
TEST_HR_DIR   = 'D:/cey/CNTs_detials/ML_U_net/test/hr'

INFER_DIR     = 'D:/cey/CNTs_detials/ML_U_net/raw'
OUT_DIR       = 'D:/cey/CNTs_detials/ML_U_net/predict'
SAMPLE_OUT_DIR= 'D:/cey/CNTs_detials/ML_U_net/sample'
SAVED_DIR     = 'D:/cey/CNTs_detials/ML_U_net/saved'  # 保存模型目录

BATCH_SIZE    = 2
EPOCHS        = 50
IMG_HEIGHT    = 512
IMG_WIDTH     = 512
IMG_CHANNELS  = 1  # 单通道灰度

SUFFIX_LR = '_lr'
SUFFIX_HR = '_hr'

os.makedirs(SAVED_DIR, exist_ok=True)

# ----------------------
# 数据计数与加载
# ----------------------
def count_pairs(lr_dir, hr_dir, suffix_lr=SUFFIX_LR, suffix_hr=SUFFIX_HR):
    if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
        return 0
    lr_files = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
    cnt = 0
    for f in lr_files:
        name, ext = os.path.splitext(f)
        if name.endswith(suffix_lr):
            base = name[:-len(suffix_lr)]
            hr_name = base + suffix_hr + ext
            if os.path.isfile(os.path.join(hr_dir, hr_name)):
                cnt += 1
    return cnt

TRAIN_COUNT = count_pairs(LR_DIR, HR_DIR)
VAL_COUNT   = count_pairs(VAL_LR_DIR, VAL_HR_DIR)
TEST_COUNT  = count_pairs(TEST_LR_DIR, TEST_HR_DIR)

def load_dataset(lr_dir, hr_dir, batch_size=BATCH_SIZE, img_h=IMG_HEIGHT, img_w=IMG_WIDTH,
                 suffix_lr=SUFFIX_LR, suffix_hr=SUFFIX_HR):
    if not os.path.isdir(lr_dir) or not os.path.isdir(hr_dir):
        return tf.data.Dataset.from_tensors((tf.zeros((0, img_h, img_w, IMG_CHANNELS), tf.float32),
                                             tf.zeros((0, img_h, img_w, IMG_CHANNELS), tf.float32))).take(0)

    lr_files = sorted([f for f in os.listdir(lr_dir)
                       if os.path.splitext(f)[0].endswith(suffix_lr)
                       and os.path.isfile(os.path.join(lr_dir, f))])

    # 只保留有配对 HR 的样本
    paired = []
    for lf in lr_files:
        name, ext = os.path.splitext(lf)
        base = name[:-len(suffix_lr)]
        hr_fname = base + suffix_hr + ext
        if os.path.isfile(os.path.join(hr_dir, hr_fname)):
            paired.append((lf, hr_fname))

    def gen():
        for lr_fname, hr_fname in paired:
            lr_path = os.path.join(lr_dir, lr_fname)
            hr_path = os.path.join(hr_dir, hr_fname)
            lr = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
            hr = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
            if lr is None or hr is None:
                continue
            if (lr.shape[0], lr.shape[1]) != (img_h, img_w):
                lr = cv2.resize(lr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            if (hr.shape[0], hr.shape[1]) != (img_h, img_w):
                hr = cv2.resize(hr, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            lr = lr[..., None].astype(np.float32) / 255.0
            hr = hr[..., None].astype(np.float32) / 255.0
            yield lr, hr

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((img_h, img_w, IMG_CHANNELS), tf.float32),
            tf.TensorSpec((img_h, img_w, IMG_CHANNELS), tf.float32),
        )
    )
    buffer = max(128, len(paired))  # 使用当前数据量作为 buffer，更稳定
    ds = ds.shuffle(buffer, reshuffle_each_iteration=True).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------
# 构建 U-Net（与训练时一致）
# ----------------------
def conv_block(x, f):
    x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Activation('relu', dtype='float32')(x)

def encoder_block(x, f):
    c = conv_block(x, f)
    c = conv_block(c, f)
    return c, layers.MaxPooling2D()(c)

def decoder_block(x, skip, f):
    u = layers.UpSampling2D()(x)
    u = layers.Concatenate()([u, skip])
    u = conv_block(u, f)
    return conv_block(u, f)

def build_unet(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), base=32):
    inp = layers.Input(shape)
    s1, p1 = encoder_block(inp, base)
    s2, p2 = encoder_block(p1, base * 2)
    s3, p3 = encoder_block(p2, base * 4)
    s4, p4 = encoder_block(p3, base * 8)
    b = conv_block(p4, base * 16)
    b = conv_block(b, base * 16)
    d4 = decoder_block(b, s4, base * 8)
    d3 = decoder_block(d4, s3, base * 4)
    d2 = decoder_block(d3, s2, base * 2)
    d1 = decoder_block(d2, s1, base)
    out = layers.Conv2D(1, 1, activation='sigmoid', dtype='float32')(d1)
    return models.Model(inp, out)

# ----------------------
# 评估 / 绘图 / 可视化等
# ----------------------
def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=1.0)

def visualize_samples(dataset, out_dir, model, num=4):
    os.makedirs(out_dir, exist_ok=True)
    for lr_batch, hr_batch in dataset.take(1):
        preds = model.predict(lr_batch, verbose=0)
        for i in range(min(num, lr_batch.shape[0])):
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].imshow(lr_batch[i, ..., 0], cmap='gray'); ax[0].set_title('LR'); ax[0].axis('off')
            ax[1].imshow(hr_batch[i, ..., 0], cmap='gray'); ax[1].set_title('HR'); ax[1].axis('off')
            ax[2].imshow(preds[i, ..., 0], cmap='gray'); ax[2].set_title('Pred'); ax[2].axis('off')
            path = os.path.join(out_dir, f'sample_{i}.png')
            plt.savefig(path); plt.close(fig)
            print(f'Saved {path}')

def plot_training_curves(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    metrics = history.history or {}
    # loss
    plt.figure()
    plt.plot(metrics.get('loss', []), label='train_loss')
    if 'val_loss' in metrics: plt.plot(metrics['val_loss'], label='val_loss')
    plt.title('Training and Validation Loss'); plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend()
    loss_path = os.path.join(out_dir, 'loss_curve.png'); plt.savefig(loss_path); plt.close()
    print(f'Saved loss curve to {loss_path}')
    # psnr
    train_psnr_key = next((k for k in metrics if k.startswith('psnr') and not k.startswith('val_')), None)
    val_psnr_key   = next((k for k in metrics if k.startswith('val_psnr')), None)
    if train_psnr_key and val_psnr_key:
        plt.figure()
        plt.plot(metrics[train_psnr_key], label='train_psnr')
        plt.plot(metrics[val_psnr_key],   label='val_psnr')
        plt.title('Training and Validation PSNR'); plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend()
        psnr_path = os.path.join(out_dir, 'psnr_curve.png'); plt.savefig(psnr_path); plt.close()
        print(f'Saved PSNR curve to {psnr_path}')

# ----------------------
# 按 512x512 切块预测并重建（用于推理）
# ----------------------
def predict_image_by_tiles(model, img_gray, tile_size=512, tile_batch=8):
    if img_gray.ndim == 3 and img_gray.shape[2] == 1:
        img_gray = img_gray[..., 0]
    h, w = img_gray.shape
    pad_h = (tile_size - (h % tile_size)) % tile_size
    pad_w = (tile_size - (w % tile_size)) % tile_size
    pad_top = pad_h // 2; pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2; pad_right = pad_w - pad_left
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img_padded = np.pad(img_gray, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='reflect')
    else:
        img_padded = img_gray
    H, W = img_padded.shape
    n_h = H // tile_size; n_w = W // tile_size
    tiles = []
    for iy in range(n_h):
        for ix in range(n_w):
            y0 = iy * tile_size; x0 = ix * tile_size
            tile = img_padded[y0:y0+tile_size, x0:x0+tile_size]
            tiles.append(tile)
    if not tiles:
        # 小于一个 tile 的图，直接 resize 到 tile_size，再反裁剪回原尺寸
        resized = cv2.resize(img_padded, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        batch = (resized.astype(np.float32) / 255.0)[None, ..., None]
        pred = model.predict(batch, verbose=0)[0, ..., 0]
        pred = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
        recon_cropped = pred[pad_top:pad_top+h, pad_left:pad_left+w]
        return (np.clip(recon_cropped, 0.0, 1.0)*255).astype(np.uint8)

    tiles = np.stack(tiles, axis=0).astype(np.float32) / 255.0
    tiles = tiles[..., None]
    preds = []
    for i in range(0, tiles.shape[0], tile_batch):
        batch = tiles[i:i+tile_batch]
        pred_batch = model.predict(batch, verbose=0)
        preds.append(pred_batch)
    preds = np.concatenate(preds, axis=0)
    recon = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for iy in range(n_h):
        for ix in range(n_w):
            y0 = iy * tile_size; x0 = ix * tile_size
            recon[y0:y0+tile_size, x0:x0+tile_size] = preds[idx, ..., 0]; idx += 1
    recon_cropped = recon[pad_top:pad_top+h, pad_left:pad_left+w]
    recon_uint8 = (np.clip(recon_cropped, 0.0, 1.0) * 255).astype(np.uint8)
    return recon_uint8

# ----------------------
# 主流程：训练 -> 保存最佳模型 -> 保存 SavedModel -> 用保存的模型做推理
# ----------------------
if __name__ == '__main__':
    print(f'Train: {TRAIN_COUNT}, Val: {VAL_COUNT}, Test: {TEST_COUNT}')

    train_ds = load_dataset(LR_DIR, HR_DIR).repeat()
    val_ds   = load_dataset(VAL_LR_DIR, VAL_HR_DIR).repeat()
    test_ds  = load_dataset(TEST_LR_DIR, TEST_HR_DIR)

    print('Building model...')
    model = build_unet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 32)
    model.compile(optimizer='adam', loss='mse', metrics=[psnr_metric])

    # 回调
    best_h5 = os.path.join(SAVED_DIR, 'best_model.h5')
    savedmodel_dir = os.path.join(SAVED_DIR, 'saved_model_best')

    checkpoint_cb = ModelCheckpoint(
        best_h5, monitor='val_loss', mode='min',
        save_best_only=True, save_weights_only=False, verbose=1
    )
    earlystop_cb = EarlyStopping(monitor='val_loss', mode='min', patience=12,
                                 restore_best_weights=True, verbose=1)

    steps_per_epoch  = max(1, TRAIN_COUNT // BATCH_SIZE)
    validation_steps = max(1, VAL_COUNT   // BATCH_SIZE)

    print('Starting training...')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[TqdmCallback(verbose=1), checkpoint_cb, earlystop_cb],
        verbose=0
    )
    print('Training done.')

    # 图与样本
    plot_training_curves(history, SAMPLE_OUT_DIR)
    visualize_samples(load_dataset(VAL_LR_DIR, VAL_HR_DIR, batch_size=BATCH_SIZE), SAMPLE_OUT_DIR, model, num=4)

    # 将最佳 .h5 转为 SavedModel（或直接保存当前模型）
    try:
        if os.path.exists(best_h5):
            print(f"Converting best .h5 -> SavedModel at: {savedmodel_dir}")
            try:
                loaded = tf.keras.models.load_model(best_h5, compile=False)
                loaded.save(savedmodel_dir, include_optimizer=False)
                print("SavedModel written from .h5 via load_model().")
            except Exception as e1:
                print("Direct load_model(.h5) failed:", e1, "\nFallback to build+load_weights...")
                tmp = build_unet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), 32)
                tmp.load_weights(best_h5)
                tmp.save(savedmodel_dir, include_optimizer=False)
                print("SavedModel written from weights.")
        else:
            print("No .h5 found; saving current model as SavedModel.")
            model.save(savedmodel_dir, include_optimizer=False)
    except Exception as e:
        print("Failed to write SavedModel:", e)

    # 用保存的模型进行推理
    try:
        print("Loading SavedModel for inference:", savedmodel_dir)
        infer_model = tf.keras.models.load_model(savedmodel_dir, compile=False)
    except Exception as e:
        print("Loading SavedModel failed:", e)
        if os.path.exists(best_h5):
            print("Fallback: loading .h5 for inference:", best_h5)
            infer_model = tf.keras.models.load_model(best_h5, compile=False)
        else:
            raise RuntimeError("No saved model available for inference.")

    if os.path.isdir(INFER_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
        files = sorted([f for f in os.listdir(INFER_DIR)
                        if os.path.isfile(os.path.join(INFER_DIR, f))])
        for f in files:
            in_path = os.path.join(INFER_DIR, f)
            img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print("Skipping unreadable:", in_path); continue
            try:
                pred_uint8 = predict_image_by_tiles(infer_model, img, tile_size=512, tile_batch=8)
                out_path = os.path.join(OUT_DIR, os.path.splitext(f)[0] + '.png')
                cv2.imwrite(out_path, pred_uint8)
                print('Saved prediction to', out_path)
            except Exception as e:
                print(f"Failed to predict {in_path}: {e}")
        print('Inference saved to', OUT_DIR)
    else:
        print("INFER_DIR not found; skipping inference step.")
