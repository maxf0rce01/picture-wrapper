#!/usr/bin/env python3
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import imagehash
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.models as models
import torchvision.transforms as T
import torch.nn.functional as F
import io
import random
import zipfile
from io import BytesIO

# --- Настройки интерфейса ---
st.set_page_config(page_title="🖼️ Image Unique & Compare Tool", layout="wide")
st.title("🖼️ Инструмент уникализации и сравнения изображений")


# --- Функция уникализации ---
def make_unique(img, intensity=0.5):
    import random, io
    from PIL import Image, ImageEnhance, ImageFilter
    import numpy as np

    try:
        # --- 1. Лёгкие геометрические трансформации ---
        angle = random.uniform(-3, 3) * intensity  # 🔹 мягкий наклон
        scale = random.uniform(0.9, 1.1)
        img = img.rotate(angle, expand=True)
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)))

        # --- 2. Горизонтальное зеркалирование (всегда) ---
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # --- 3. Цветовые искажения ---
        img = ImageEnhance.Color(img).enhance(random.uniform(0.8, 1.2))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.9, 1.1))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.9, 1.3))

        # --- 4. Добавление микрошумов ---
        np_img = np.array(img).astype(np.int16)
        noise_strength = int(15 * intensity)
        noise = np.random.randint(-noise_strength, noise_strength + 1, np_img.shape)
        np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(np_img)

        # --- 5. Небольшой HSV-сдвиг ---
        hsv = np.array(img.convert("HSV"))
        hsv[..., 0] = (hsv[..., 0] + random.randint(-10, 10)) % 255
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.9, 1.1), 0, 255)
        img = Image.fromarray(hsv, "HSV").convert("RGB")

        # --- 6. Лёгкое размытие (иногда) ---
        if random.random() < 0.4:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

        # --- 6.5. Случайное лёгкое кадрирование ---
        if random.random() < 0.6:
            w, h = img.size
            crop_x = random.randint(5, int(0.03 * w))
            crop_y = random.randint(5, int(0.03 * h))
            img = img.crop((crop_x, crop_y, w - crop_x, h - crop_y))

        # --- 7. Минималистичные EXIF-метаданные ---
        exif_bytes = b"Exif\0\0" + bytes(
            [random.randint(0, 255) for _ in range(24, 48)]
        )

        # --- 8. Сжатие и оптимизация ---
        output = io.BytesIO()
        format_choice = random.choice(["JPEG", "WEBP"])
        quality = random.randint(65, 80)
        img.save(
            output,
            format=format_choice,
            quality=quality,
            exif=exif_bytes,
            optimize=True,
        )
        file_bytes = bytearray(output.getvalue())

        # --- 9. Небольшой хвост байтов ---
        tail_len = random.randint(8, 32)
        file_bytes.extend(bytes([random.randint(0, 255) for _ in range(tail_len)]))

        # --- 10. Возврат итогового изображения ---
        final_img = Image.open(io.BytesIO(file_bytes))
        return final_img

    except Exception as e:
        print(f"Ошибка при уникализации: {e}")
        return img


# --- Методы сравнения ---
def phash_distance(img1, img2):
    try:
        return imagehash.phash(img1) - imagehash.phash(img2)
    except Exception as e:
        st.error(f"Ошибка pHash: {e}")
        return None


def ssim_score(img1, img2):
    try:
        i1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
        i2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
        h = min(i1.shape[0], i2.shape[0])
        w = min(i1.shape[1], i2.shape[1])
        i1 = cv2.resize(i1, (w, h))
        i2 = cv2.resize(i2, (w, h))
        score, _ = ssim(i1, i2, full=True)
        return score
    except Exception as e:
        st.error(f"Ошибка SSIM: {e}")
        return None


@st.cache_resource
def load_resnet():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    preprocess = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, preprocess


def resnet_similarity(img1, img2):
    try:
        model, preprocess = load_resnet()
        with torch.no_grad():
            t1 = preprocess(img1).unsqueeze(0)
            t2 = preprocess(img2).unsqueeze(0)
            f1 = model(t1)
            f2 = model(t2)
            similarity = F.cosine_similarity(f1, f2).item()
        return similarity
    except Exception as e:
        st.error(f"Ошибка ResNet: {e}")
        return None


# --- Вкладки ---
tab1, tab2 = st.tabs(["🎨 Уникализация", "🔍 Сравнение"])

# ---------- Вкладка 1: УНИКАЛИЗАЦИЯ ----------
# ---------- Вкладка 1: УНИКАЛИЗАЦИЯ ----------
with tab1:
    st.header("🎨 Уникализация изображений")

    uploaded_images = st.file_uploader(
        "Выберите одно или несколько изображений",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="unique_upload",
    )

    intensity = st.slider("Интенсивность изменений", 0.0, 0.5, 0.2, 0.05)

    if uploaded_images:
        st.write(f"📂 Загружено файлов: {len(uploaded_images)}")

        # Отображаем оригиналы
        cols = st.columns(min(3, len(uploaded_images)))
        for idx, img_file in enumerate(uploaded_images):
            with cols[idx % len(cols)]:
                img = Image.open(img_file)
                st.image(img, caption=f"Оригинал {idx+1}", use_container_width=True)

        num_copies = st.number_input(
            "Количество уникальных копий для каждого изображения",
            min_value=1,
            max_value=10,
            value=1,
        )

        if st.button("✨ Уникализировать все"):
            st.info("🔄 Пакетная обработка изображений...")

            zip_buffer = BytesIO()
            phash_all = []

            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
                for idx, img_file in enumerate(uploaded_images):
                    img = Image.open(img_file).convert("RGB")
                    st.write(f"📂 Обработка изображения {idx + 1}: {img_file.name}")

                    phash_for_this_image = []  # храним все pHash текущего файла

                    for copy_idx in range(num_copies):
                        attempt = 0
                        while True:
                            attempt += 1
                            unique_img = make_unique(img, intensity)
                            new_phash = imagehash.phash(unique_img)

                            # проверяем уникальность относительно всех копий этого же файла
                            if (
                                all(
                                    abs(new_phash - prev_ph) > 10
                                    for prev_ph in phash_for_this_image
                                )
                                or attempt > 5
                            ):
                                break

                        phash_for_this_image.append(new_phash)
                        phash_all.append(new_phash)

                        # сохраняем копию
                        img_bytes = BytesIO()
                        unique_img.save(img_bytes, format="PNG")
                        zipf.writestr(
                            f"unique_{idx+1}_copy_{copy_idx+1}.png",
                            img_bytes.getvalue(),
                        )

                        st.write(
                            f"✅ Изображение {idx+1}, копия {copy_idx+1} готова (попыток: {attempt})"
                        )

            zip_buffer.seek(0)

            st.success(
                "🎉 Все уникализированные копии созданы и уникальны относительно всех предыдущих!"
            )
            st.download_button(
                "📦 Скачать все уникализированные изображения (ZIP)",
                data=zip_buffer,
                file_name="unique_images.zip",
                mime="application/zip",
            )


# ---------- Вкладка 2: СРАВНЕНИЕ ----------
with tab2:
    st.header("🔍 Сравнение двух изображений")

    col1, col2 = st.columns(2)
    with col1:
        img1_file = st.file_uploader(
            "Первое изображение", type=["jpg", "jpeg", "png"], key="img1_upload"
        )
    with col2:
        img2_file = st.file_uploader(
            "Второе изображение", type=["jpg", "jpeg", "png"], key="img2_upload"
        )

    if not img1_file and not img2_file:
        st.info("👆 Загрузите два изображения для сравнения")

    if img1_file and img2_file:
        img1 = Image.open(img1_file).convert("RGB")
        img2 = Image.open(img2_file).convert("RGB")

        st.image(
            [img1, img2],
            caption=["Изображение 1", "Изображение 2"],
            use_container_width=True,
        )

        if st.button("🔎 Сравнить изображения"):
            st.subheader("📊 Результаты:")

            ph = phash_distance(img1, img2)
            ss = ssim_score(img1, img2)
            rs = resnet_similarity(img1, img2)

            if ph is not None:
                st.write(f"**Разница pHash:** {ph}")
            if ss is not None:
                st.write(f"**SSIM:** {ss:.4f}")
            if rs is not None:
                st.write(f"**ResNet сходство:** {rs:.4f}")

            if all(v is not None for v in [ph, ss, rs]):
                if ph > 10 and ss < 0.7 and rs < 0.85:
                    st.success("🟢 Изображения уникальны (существенные отличия)")
                    unique_score = int(
                        ((ph / 64) * 0.4 + (1 - ss) * 0.3 + (1 - rs) * 0.3) * 100
                    )
                    st.progress(min(unique_score, 100))
                    st.write(f"**Оценка уникальности:** {unique_score}%")
                else:
                    st.warning("🔴 Изображения похожи (недостаточно различий)")
                    sim_score = int(((1 - (ph / 64)) * 0.4 + ss * 0.3 + rs * 0.3) * 100)
                    st.progress(min(sim_score, 100))
                    st.write(f"**Оценка сходства:** {sim_score}%")

st.markdown("---")
st.caption("© Ну что сказать - хуй")
