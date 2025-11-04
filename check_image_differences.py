#!/usr/bin/env python3
"""
check_image_differences.py
Проверка различий изображений между оригиналом и модифицированными.
"""

import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import argparse

# ---------------- helper funcs ----------------


def load_image_cv(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot open image {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def compute_color_histogram(img, bins=(8, 8, 8)):
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256] * 3)
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def compute_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score


def extract_orb_features(img):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors


def compare_orb(des1, des2):
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return len(matches)


def kmeans_difference(img1, img2, n_clusters=5):
    flat1 = img1.reshape(-1, 3)
    flat2 = img2.reshape(-1, 3)
    km1 = KMeans(n_clusters=n_clusters, random_state=42).fit(flat1)
    km2 = KMeans(n_clusters=n_clusters, random_state=42).fit(flat2)
    # просто сравниваем центры кластеров
    diff = np.mean(
        np.abs(
            np.sort(km1.cluster_centers_, axis=0)
            - np.sort(km2.cluster_centers_, axis=0)
        )
    )
    return diff


def dbscan_difference(img1, img2, eps=20, min_samples=10):
    flat1 = img1.reshape(-1, 3)
    flat2 = img2.reshape(-1, 3)
    db1 = DBSCAN(eps=eps, min_samples=min_samples).fit(flat1)
    db2 = DBSCAN(eps=eps, min_samples=min_samples).fit(flat2)
    # сравниваем количество кластеров
    labels1 = len(set(db1.labels_))
    labels2 = len(set(db2.labels_))
    return abs(labels1 - labels2)


# ---------------- main ----------------


def check_images(original_dir, modified_dir):
    original_dir = Path(original_dir)
    modified_dir = Path(modified_dir)

    originals = sorted(
        [p for p in original_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")]
    )
    modifieds = sorted(
        [p for p in modified_dir.iterdir() if p.suffix.lower() in (".jpg", ".png")]
    )

    if len(originals) != len(modifieds):
        print("Warning: number of images differs")

    for orig_path, mod_path in zip(originals, modifieds):
        img_orig = load_image_cv(orig_path)
        img_mod = load_image_cv(mod_path)

        hist_orig = compute_color_histogram(img_orig)
        hist_mod = compute_color_histogram(img_mod)
        hist_diff = np.linalg.norm(hist_orig - hist_mod)

        ssim_score = compute_ssim(img_orig, img_mod)

        _, des_orig = extract_orb_features(img_orig)
        _, des_mod = extract_orb_features(img_mod)
        orb_matches = compare_orb(des_orig, des_mod)

        kmeans_diff = kmeans_difference(img_orig, img_mod)
        dbscan_diff = dbscan_difference(img_orig, img_mod)

        print(f"Image: {orig_path.name}")
        print(f"  Color histogram diff: {hist_diff:.4f}")
        print(f"  SSIM: {ssim_score:.4f}")
        print(f"  ORB matches: {orb_matches}")
        print(f"  KMeans center diff: {kmeans_diff:.4f}")
        print(f"  DBSCAN cluster diff: {dbscan_diff}")
        print("-" * 50)


# ---------------- CLI ----------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original", "-o", required=True, help="Folder with original images"
    )
    parser.add_argument(
        "--modified", "-m", required=True, help="Folder with modified images"
    )
    args = parser.parse_args()

    check_images(args.original, args.modified)
    input("Press Enter to exit...")
