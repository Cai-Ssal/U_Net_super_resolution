import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import mixed_precision
from tqdm.keras import TqdmCallback
import matplotlib.pyplot as plt

# ----------------------
# 性能优化设置
# ----------------------
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# ----------------------
# 路径与参数配置
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

BATCH_SIZE    = 4
EPOCHS        = 50
IMG_SIZE      = (512, 512)

# ----------------------
# 数据计数与加载
# ----------------------
def count_pairs(lr_dir, hr_dir):
    lr_files = [f for f in os.listdir(lr_dir) if os.path.isfile(os.path.join(lr_dir, f))]
    count = 0
    for f in lr_files:
        name, ext = os.path.splitext(f)
        if name.endswith('_lr'):
            base = name[:-3]
            hr_name = base + '_hr' + ext
            if os.path.isfile(os.path.join(hr_dir, hr_name)):
                count += 1
    return count

TRAIN_COUNT = count_pairs(LR_DIR, HR_DIR)
VAL_COUNT   = count_pairs(VAL_LR_DIR, VAL_HR_DIR)
TEST_COUNT  = count_pairs(TEST_LR_DIR, TEST_HR_DIR)

def load_dataset(lr_dir, hr_dir, batch_size=BATCH_SIZE, img_size=IMG_SIZE):
    lr_files = sorted([f for f in os.listdir(lr_dir) if os.path.splitext(f)[0].endswith('_lr')])
    def gen():
        for lr_fname in lr_files:
            name, ext = os.path.splitext(lr_fname)
            base = name[:-3]
            hr_fname = base + '_hr' + ext
            lr_path = os.path.join(lr_dir, lr_fname)
            hr_path = os.path.join(hr_dir, hr_fname)
            if not os.path.isfile(hr_path):
                continue
            lr = cv2.imread(lr_path, cv2.IMREAD_GRAYSCALE)
            hr = cv2.imread(hr_path, cv2.IMREAD_GRAYSCALE)
            if lr is None or hr is None:
                continue
            lr = cv2.resize(lr, img_size)[..., None] / 255.0  # 添加通道维度
            hr = cv2.resize(hr, img_size)[..., None] / 255.0
            yield lr.astype(np.float32), hr.astype(np.float32)
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((img_size[1], img_size[0], 1), tf.float32),
            tf.TensorSpec((img_size[1], img_size[0], 1), tf.float32),
        )
    )
    ds = ds.shuffle(200).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# ----------------------
# 可视化样本
# ----------------------
def visualize_samples(dataset, out_dir, num=4):
    os.makedirs(out_dir, exist_ok=True)
    for lr_batch, hr_batch in dataset.take(1):
        preds = model.predict(lr_batch)
        for i in range(min(num, lr_batch.shape[0])):
            fig, ax = plt.subplots(1, 3, figsize=(9, 3))
            ax[0].imshow(lr_batch[i, ..., 0], cmap='gray'); ax[0].set_title('LR'); ax[0].axis('off')
            ax[1].imshow(hr_batch[i, ..., 0], cmap='gray'); ax[1].set_title('HR'); ax[1].axis('off')
            ax[2].imshow(preds[i, ..., 0], cmap='gray'); ax[2].set_title('Pred'); ax[2].axis('off')
            path = os.path.join(out_dir, f'sample_{i}.png')
            plt.savefig(path); plt.close(fig)
            print(f'Saved {path}')

# ----------------------
# 构建 U-Net 模型
# ----------------------
def conv_block(x, f): x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x); x = layers.BatchNormalization()(x); return layers.Activation('relu', dtype='float32')(x)
def encoder_block(x, f): c = conv_block(x, f); c = conv_block(c, f); return c, layers.MaxPooling2D()(c)
def decoder_block(x, skip, f): u = layers.UpSampling2D()(x); u = layers.Concatenate()([u, skip]); u = conv_block(u, f); return conv_block(u, f)
def build_unet(shape=(512, 512, 1), base=32):
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
# PSNR 评估函数
# ----------------------
def psnr_metric(y_true, y_pred): return tf.image.psnr(y_true, y_pred, max_val=1.0)

# ----------------------
# 主函数流程
# ----------------------
if __name__ == '__main__':
    print(f'Training samples: {TRAIN_COUNT}, Validation samples: {VAL_COUNT}')
    
    # 添加 .repeat() 确保数据源足够多 epoch 使用
    train_ds = load_dataset(LR_DIR, HR_DIR).repeat()
    val_ds = load_dataset(VAL_LR_DIR, VAL_HR_DIR).repeat()
    test_ds = load_dataset(TEST_LR_DIR, TEST_HR_DIR)

    print('Building model...')
    model = build_unet((IMG_SIZE[1], IMG_SIZE[0], 1), 32)
    model.compile('adam', 'mse', metrics=[psnr_metric])

    print('Starting training...')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=TRAIN_COUNT // BATCH_SIZE,
        validation_steps=VAL_COUNT // BATCH_SIZE,
        callbacks=[TqdmCallback(verbose=1)],
        verbose=0
    )

    print('Training done.')
    visualize_samples(val_ds, SAMPLE_OUT_DIR)

    print('Testing...')
    test = model.evaluate(test_ds, steps=TEST_COUNT // BATCH_SIZE, verbose=1)
    print('Test results:', test)

    if os.path.isdir(INFER_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)
        for f in sorted(os.listdir(INFER_DIR)):
            img = cv2.imread(os.path.join(INFER_DIR, f), cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img = cv2.resize(img, IMG_SIZE)[..., None] / 255.0
            res = model.predict(img[None, ...], verbose=0)[0, ..., 0]
            res = (res * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(OUT_DIR, f), res)
        print('Inference saved to', OUT_DIR)
