# make_animals_mini_lists.py
import os
import random

BASE_ROOT = "./datasets/animals-mini-uncropped"
TRAIN_ROOT = os.path.join(BASE_ROOT, "train")
TEST_ROOT  = os.path.join(BASE_ROOT, "target")

TRAIN_LIST = os.path.join(BASE_ROOT, "animals-mini_list_train.txt")
TEST_LIST  = os.path.join(BASE_ROOT, "animals-mini_list_target.txt")

TRAIN_FRAC = 0.8
SEED = 0
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_class_paths(split_root, frac=None):
    """Return list of 'class_name/filename.jpg' relative paths."""
    paths = []
    random.seed(SEED)

    for cls in sorted(os.listdir(split_root)):
        cls_dir = os.path.join(split_root, cls)
        if not os.path.isdir(cls_dir):
            continue

        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue

        random.shuffle(imgs)

        if frac is not None:
            n = int(len(imgs) * frac)
            imgs = imgs[:max(1, n)]

        for fname in imgs:
            rel_path = f"{cls}/{fname}"
            paths.append(rel_path)

    return paths


def main():
    random.seed(SEED)

    # Train: take TRAIN_FRAC of images in each class
    train_paths = []
    test_paths  = []

    for cls in sorted(os.listdir(TRAIN_ROOT)):
        cls_dir = os.path.join(TRAIN_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue

        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(IMG_EXTS)]
        if not imgs:
            continue

        random.shuffle(imgs)
        n_train = int(len(imgs) * TRAIN_FRAC)
        train_imgs = imgs[:n_train]
        test_imgs  = imgs[n_train:]  # *within the same classes* but you can ignore if you want

        for fname in train_imgs:
            train_paths.append(f"{cls}/{fname}")
        # if you prefer test only from TEST_ROOT, comment out the next loop
        for fname in test_imgs:
            test_paths.append(f"{cls}/{fname}")

    # Also include TEST_ROOT images as test (unseen images, maybe unseen classes)
    for cls in sorted(os.listdir(TEST_ROOT)):
        cls_dir = os.path.join(TEST_ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue

        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(IMG_EXTS)]
        for fname in imgs:
            test_paths.append(f"{cls}/{fname}")

    with open(TRAIN_LIST, "w") as f:
        for p in train_paths:
            f.write(p + "\n")

    with open(TEST_LIST, "w") as f:
        for p in test_paths:
            f.write(p + "\n")

    print(f"Wrote {len(train_paths)} train and {len(test_paths)} target paths.")
    print("Train list:", TRAIN_LIST)
    print("Target list:", TEST_LIST)


if __name__ == "__main__":
    main()
