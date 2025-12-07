import argparse
import os
import random
import shutil
from pathlib import Path
import yaml
from tqdm import tqdm
import cv2

from ultralytics import YOLO



def find_images(data_root, exts=None):
    exts = exts or ['.jpg', '.jpeg', '.png']
    data_root = Path(data_root)
    images = []
    for p in data_root.rglob('*'):
        if p.suffix.lower() in exts:
            images.append(p)
    return images




def split_by_image(images, val_ratio=0.2, seed=42):
    random.Random(seed).shuffle(images)
    n_val = max(1, int(len(images) * val_ratio))
    val_imgs = images[:n_val]
    train_imgs = images[n_val:]
    return {'train': sorted(train_imgs), 'val': sorted(val_imgs)}


def create_dataset_yaml(out_path, train_images, val_images, labels_root, nc=1, names=None):
    dataset = {
        'train': str(out_path / 'images' / 'train'),
        'val': str(out_path / 'images' / 'val'),
        'nc': nc,
        'names': names or {0: 'valve'}
    }
    (out_path / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (out_path / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (out_path / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (out_path / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    def copy_data(imgs, split):
        for p in imgs:
            dst = out_path / 'images' / split / (p.parent.name + '__' + p.name)
            if not dst.exists():
                shutil.copy2(p, dst)
            lab = labels_root / p.parent.name / (p.stem + '.txt')
            dstlab = out_path / 'labels' / split / (p.parent.name + '__' + p.stem + '.txt')
            if lab.exists():
                shutil.copy2(lab, dstlab)
            else:
                dstlab.write_text("")

    copy_data(train_images, 'train')
    copy_data(val_images, 'val')

    yaml_path = out_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(dataset, f)
    return str(yaml_path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    img_root = Path(cfg['data_root']).resolve()
    labels_root = Path(cfg['labels_root']).resolve()
    out_runs_root = (Path(cfg['out']) / cfg.get('name')).resolve()
    out_dataset_root = out_runs_root / 'dataset_for_training'
    out_dataset_root.mkdir(parents=True, exist_ok=True)


    # find images
    images = find_images(img_root)
    if len(images) == 0:
        raise SystemExit(f"No images found under {img_root}")

    # Generate a blank TXT file from images without labels.
    print("Ensuring label files exist...")
    for p in tqdm(images):
        lab = labels_root / p.parent.name / (p.stem + '.txt')
        lab.parent.mkdir(parents=True, exist_ok=True)
        if not lab.exists():
            lab.write_text("")

    # split dataset
    splits = split_by_image(images, val_ratio=cfg['val_ratio'], seed=cfg['seed'])
    train_images, val_images = splits['train'], splits['val']

    # create dataset yaml
    dataset_yaml = create_dataset_yaml(out_dataset_root, train_images, val_images, labels_root,
                                       nc=cfg['nc'], names=cfg['names'])

    # model + callback

    #If path dosen't work, copy path by your own and paste here
    model = YOLO(model = './ultralytics/cfg/models/11/yolo11x.yaml')


    def stop_at_180(trainer):
        if trainer.epoch + 1 >= 180:
            print("\nEarly stop at epoch 180")
            raise SystemExit("Early stop at epoch 180")

    model.add_callback("on_train_epoch_end", stop_at_180)
    total_epochs = cfg['epochs']
    best_map50 = 0.0
    now_epoch = 0


    train_args = dict(
        data=str(out_dataset_root / 'dataset.yaml'),
        epochs=cfg['epochs'], imgsz=cfg['imgsz'], batch=cfg['batch'],
        project=cfg['project'], name=cfg['name'], workers=cfg['workers'],
        exist_ok=True,
        val=True,
        patience=cfg["patience"],
        hsv_h = cfg["hsv_h"],
        hsv_s = cfg["hsv_s"],
        hsv_v = cfg["hsv_v"],
        degrees = cfg["degrees"],
        translate = cfg["translate"],
        scale = cfg["scale"],
        shear = cfg["shear"],
        perspective = cfg["perspective"],
        flipud = cfg["flipud"],
        fliplr = cfg["fliplr"],
        bgr = cfg["bgr"],
        mosaic = cfg["mosaic"],
        mixup = cfg["mixup"],
        cutmix = cfg["cutmix"],
        copy_paste = cfg["copy_paste"],
    )

    results = model.train(**train_args)

    print("Finish !")
