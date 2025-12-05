import os
import argparse
import torch
import torchvision.transforms as T
from torchvision.utils import make_grid, save_image
from PIL import Image
from utils import get_config

# ---- Adjust this import if needed ----
from networks import FewShotGen


def load_image(path, size=128):
    tfm = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ])
    img = Image.open(path).convert("RGB")
    return tfm(img)


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(exts)
    ])


def translate_pair_original(gen, src, tgt, octave_alpha=1.0):
    """
    Original-code faithful translation:
    content = enc_content(src)
    style   = enc_class_model(tgt)
    out     = decode(content, style)
    """

    # Match training/eval behavior exactly
    content, class_codes = gen.encode(src, tgt, octave_alpha)

    # In original code, they always average style codes across k-shot
    model_code = torch.mean(class_codes, dim=0).unsqueeze(0)

    # Decode with the same alpha routing used everywhere else
    out = gen.decode(content, model_code, octave_alpha)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_ckpt", required=True)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--target_dir", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--num_pairs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    # ---------------- Load generator ----------------

    print("Loading generator...")
    g_cfg = get_config("configs/gen.yaml")
    gen = FewShotGen(g_cfg)

    ckpt = torch.load(args.gen_ckpt, map_location=device)

    if isinstance(ckpt, dict) and "gen" in ckpt:
        state_dict = ckpt["gen"]
    else:
        state_dict = ckpt

    gen.load_state_dict(state_dict, strict=True)
    gen.to(device).eval()

    # ---------------- Load images ----------------

    source_imgs = list_images(args.source_dir)
    target_imgs = list_images(args.target_dir)

    assert len(source_imgs) >= args.num_pairs, "Not enough source images"
    assert len(target_imgs) >= args.num_pairs, "Not enough target images"

    source_imgs = source_imgs[:args.num_pairs]
    target_imgs = target_imgs[:args.num_pairs]

    sources = []
    targets = []
    outputs = []

    # ---------------- Forward pass ----------------

    with torch.no_grad():
        for src_path, tgt_path in zip(source_imgs, target_imgs):
            src = load_image(src_path, args.image_size).unsqueeze(0).to(device)
            tgt = load_image(tgt_path, args.image_size).unsqueeze(0).to(device)

            out = translate_pair_original(gen, src, tgt, octave_alpha=1.0)

            sources.append(src.squeeze(0).cpu())
            targets.append(tgt.squeeze(0).cpu())
            outputs.append(out.squeeze(0).cpu())

    # ---------------- Build 3x10 grid ----------------

    # Stack rows: [10 src] + [10 tgt] + [10 out] = 30 images total
    all_imgs = torch.stack(sources + targets + outputs, dim=0)

    grid = make_grid(all_imgs, nrow=args.num_pairs, normalize=True, value_range=(-1, 1))

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    save_image(grid, args.out_path)

    print("Saved grid to:", args.out_path)


if __name__ == "__main__":
    main()
