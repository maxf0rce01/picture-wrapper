#!/usr/bin/env python3
"""
make_test_dataset.py
Генерация тестовых изображений с парами:
- overlay (верхняя, оригинал)
- background (максимально испорченный, но валидный)
- modified (визуально похожий на overlay, но байтово сильно отличается)

Пример:
    python3 make_test_dataset.py --input ./input_images --output ./dataset_out --copies 3
"""

import os
import io
import csv
import argparse
import random
from pathlib import Path
from tqdm import tqdm

from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import imagehash

random.seed(42)
np.random.seed(42)

# ---------------- helper funcs ----------------


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_bytes_as_file(b: bytes, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)


def safe_recompress_jpeg_bytes(
    img: Image.Image, quality=85, subsampling=None, optimize=False
):
    buf = io.BytesIO()
    params = {"format": "JPEG", "quality": quality, "optimize": optimize}
    if subsampling is not None:
        params["subsampling"] = subsampling
    img.save(buf, **params)
    return buf.getvalue()


# ---------------- image effects ----------------


def overlay_texture_pil(img: Image.Image, alpha=0.04):
    w, h = img.size
    noise = np.random.normal(128, 26, (h, w)).astype(np.uint8)
    stripe = (np.linspace(-1, 1, w)[None, :] ** 2) * 18
    noise3 = np.clip(np.stack([noise] * 3, axis=2) + stripe[:, :, None], 0, 255).astype(
        np.uint8
    )
    tex = Image.fromarray(noise3, mode="RGB")
    return Image.blend(img.convert("RGB"), tex, alpha=alpha)


def mesh_warp_pil(img: Image.Image, strength=0.02):
    w, h = img.size
    grid_x, grid_y = 4, 4
    dx_max = int(w * strength)
    dy_max = int(h * strength)
    mesh = []
    xs = np.linspace(0, w, grid_x + 1, dtype=int)
    ys = np.linspace(0, h, grid_y + 1, dtype=int)
    for i in range(grid_x):
        for j in range(grid_y):
            x0, x1 = xs[i], xs[i + 1]
            y0, y1 = ys[j], ys[j + 1]
            src_box = (x0, y0, x1, y1)
            off = lambda m: random.randint(-m, m)
            dst_quad = (
                x0 + off(dx_max),
                y0 + off(dy_max),
                x1 + off(dx_max),
                y0 + off(dy_max),
                x1 + off(dx_max),
                y1 + off(dy_max),
                x0 + off(dx_max),
                y1 + off(dy_max),
            )
            mesh.append((src_box, dst_quad))
    return img.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC)


def local_light_pil(img: Image.Image, strength=0.18):
    w, h = img.size
    overlay = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for _ in range(random.randint(1, 3)):
        rx, ry = int(random.uniform(0.2, 0.6) * w), int(random.uniform(0.2, 0.6) * h)
        cx, cy = int(random.uniform(0.2, 0.8) * w), int(random.uniform(0.2, 0.8) * h)
        bbox = [cx - rx, cy - ry, cx + rx, cy + ry]
        shade = int(random.uniform(-80, 80))
        color = (128 + shade, 128 + shade, 128 + shade)
        draw.ellipse(bbox, fill=color)
    overlay = overlay.filter(
        ImageFilter.GaussianBlur(radius=max(1, int(min(w, h) * 0.06)))
    )
    alpha = random.uniform(0.02, strength)
    return Image.blend(img.convert("RGB"), overlay, alpha=alpha)


def small_crop_rotate(img: Image.Image, max_crop_frac=0.03, max_angle=3.0):
    w, h = img.size
    cx = int(w * random.uniform(0, max_crop_frac))
    cy = int(h * random.uniform(0, max_crop_frac))
    cropped = img.crop((cx, cy, w - cx, h - cy))
    angle = random.uniform(-max_angle, max_angle)
    return cropped.rotate(angle, resample=Image.BICUBIC, expand=True).resize((w, h))


# ---------------- aggressive background ----------------


def make_background_bytes_safe(img: Image.Image) -> bytes:
    """Максимально испорченный background, но валидный JPEG"""
    modified = img.convert("RGB")
    # Цвет, контраст, яркость, резкость
    modified = ImageEnhance.Color(modified).enhance(random.uniform(0.1, 2.5))
    modified = ImageEnhance.Contrast(modified).enhance(random.uniform(0.3, 3.0))
    modified = ImageEnhance.Brightness(modified).enhance(random.uniform(0.3, 2.0))
    modified = ImageEnhance.Sharpness(modified).enhance(random.uniform(0.2, 3.0))
    # Визуальные эффекты
    if random.random() < 0.9:
        modified = overlay_texture_pil(modified, alpha=random.uniform(0.05, 0.35))
    if random.random() < 0.9:
        modified = mesh_warp_pil(modified, strength=random.uniform(0.05, 0.2))
    if random.random() < 0.9:
        modified = local_light_pil(modified, strength=random.uniform(0.1, 0.5))
    if random.random() < 0.6:
        modified = small_crop_rotate(modified, max_crop_frac=0.1, max_angle=8.0)
    # Двойная компрессия
    q1 = random.randint(30, 85)
    q2 = random.randint(30, 85)
    buf1 = io.BytesIO()
    modified.save(buf1, format="JPEG", quality=q1, optimize=True)
    buf1.seek(0)
    tmp_img = Image.open(buf1).convert("RGB")
    buf2 = io.BytesIO()
    tmp_img.save(buf2, format="JPEG", quality=q2, optimize=True)
    return buf2.getvalue()


# ---------------- modified generator ----------------


def make_modified_from_overlay_bytes(
    overlay_img: Image.Image, background_bytes: bytes, feature_noise_strength=0.6
) -> bytes:
    """Создаёт modified визуально как overlay, но байтово сильно отличается"""
    base = overlay_img.convert("RGB")
    q = random.randint(75, 95)
    subs = random.choice([0, 1, 2])
    buf = io.BytesIO()
    base.save(buf, format="JPEG", quality=q, subsampling=subs, optimize=True)
    buf.seek(0)
    img2 = Image.open(buf).convert("RGB")
    # Добавляем микро-шум
    if feature_noise_strength > 0 and random.random() < 0.9:
        arr = np.array(img2).astype(np.int16)
        sigma = max(0.2, feature_noise_strength * 1.5)
        noise = np.random.normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img2 = Image.fromarray(arr)
    # Финальная рекомпрессия
    final_bytes = safe_recompress_jpeg_bytes(img2, quality=q, subsampling=subs)
    return final_bytes


# ---------------- pipeline ----------------


def process_image_pair(
    img_path: Path,
    overlays_dir: Path,
    backgrounds_dir: Path,
    modified_dir: Path,
    copies=2,
):
    img = Image.open(img_path)
    base_name = img_path.stem
    rows = []
    for i in range(copies):
        overlay_out = overlays_dir / f"{base_name}_{i}_overlay.png"
        background_out = backgrounds_dir / f"{base_name}_{i}_background.jpg"
        modified_out = modified_dir / f"{base_name}_{i}_modified.jpg"

        # 1) overlay
        img.convert("RGB").save(overlay_out, format="PNG")

        # 2) background
        bg_bytes = make_background_bytes_safe(img)
        save_bytes_as_file(bg_bytes, background_out)

        # 3) modified
        mod_bytes = make_modified_from_overlay_bytes(
            img, bg_bytes, feature_noise_strength=0.8
        )
        save_bytes_as_file(mod_bytes, modified_out)

        # 4) pHash и размеры
        try:
            ph_overlay = str(imagehash.phash(Image.open(overlay_out).convert("RGB")))
        except Exception:
            ph_overlay = ""
        try:
            ph_mod = str(imagehash.phash(Image.open(modified_out).convert("RGB")))
        except Exception:
            ph_mod = ""
        rows.append(
            {
                "overlay_file": str(overlay_out.relative_to(overlays_dir.parent)),
                "background_file": str(background_out.relative_to(overlays_dir.parent)),
                "modified_file": str(modified_out.relative_to(overlays_dir.parent)),
                "phash_overlay": ph_overlay,
                "phash_modified": ph_mod,
                "size_overlay": overlay_out.stat().st_size,
                "size_background": background_out.stat().st_size,
                "size_modified": modified_out.stat().st_size,
            }
        )
    return rows


def build_pairs_dataset(input_dir, output_dir, copies=2):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    overlays_dir = output_dir / "overlay"
    backgrounds_dir = output_dir / "background"
    modified_dir = output_dir / "modified"

    ensure_dir(overlays_dir)
    ensure_dir(backgrounds_dir)
    ensure_dir(modified_dir)

    files = [
        p
        for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
    ]
    csv_rows = []
    for p in tqdm(files, desc="Processing images"):
        rows = process_image_pair(
            p, overlays_dir, backgrounds_dir, modified_dir, copies=copies
        )
        csv_rows.extend(rows)

    # Сохраняем CSV
    csv_path = output_dir / "pairs_labels.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "overlay_file",
            "background_file",
            "modified_file",
            "phash_overlay",
            "phash_modified",
            "size_overlay",
            "size_background",
            "size_modified",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)

    print(
        f"✅ Dataset generated:\nOverlay: {overlays_dir}\nBackground: {backgrounds_dir}\nModified: {modified_dir}\nLabels: {csv_path}"
    )


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Input folder with images")
    parser.add_argument("--output", "-o", required=True, help="Output dataset folder")
    parser.add_argument("--copies", type=int, default=2, help="Copies per image")
    args = parser.parse_args()

    build_pairs_dataset(args.input, args.output, copies=args.copies)
