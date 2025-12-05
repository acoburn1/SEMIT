import os

ROOT_DIR = "./datasets/animals-mini-uncropped/unseen-target"
OUT_FILE = "./datasets/animals-mini-uncropped/animals-mini_list_unseen-target.txt"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

paths = []

for root, _, files in os.walk(ROOT_DIR):
    for fname in files:
        if fname.lower().endswith(IMG_EXTS):
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, ROOT_DIR)
            paths.append(rel_path)

paths.sort()

with open(OUT_FILE, "w") as f:
    for p in paths:
        f.write(p + "\n")

print(f"Wrote {len(paths)} paths to:")
print(OUT_FILE)
