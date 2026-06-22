import io
import zipfile
from typing import List, Tuple

from PIL import Image, ImageFilter, ImageEnhance
import streamlit as st
from rembg import remove


st.set_page_config(page_title="Bottle Background AI", layout="wide")


def load_image(uploaded_file) -> Image.Image:
    image = Image.open(uploaded_file)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image


def remove_background(img: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = remove(buf.getvalue())
    result = Image.open(io.BytesIO(out)).convert("RGBA")
    return trim_transparency(result)


def trim_transparency(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def make_shadow(
    bottle: Image.Image,
    blur_radius: int = 18,
    opacity: float = 0.35,
    offset_x: int = 18,
    offset_y: int = 22,
    scale_x: float = 1.02,
    scale_y: float = 0.98,
) -> Tuple[Image.Image, Tuple[int, int]]:
    alpha = bottle.getchannel("A")

    shadow = Image.new("RGBA", bottle.size, (0, 0, 0, 0))
    shadow.putalpha(alpha)

    new_w = max(1, int(shadow.width * scale_x))
    new_h = max(1, int(shadow.height * scale_y))
    shadow = shadow.resize((new_w, new_h))

    shadow = ImageEnhance.Brightness(shadow).enhance(0)

    r, g, b, a = shadow.split()
    a = a.point(lambda p: int(p * opacity))
    shadow = Image.merge("RGBA", (r, g, b, a))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))

    return shadow, (offset_x, offset_y)


def compose_bottle_on_background(
    background: Image.Image,
    bottle: Image.Image,
    width_ratio: float,
    bottom_margin: int,
    horizontal_position: str,
    custom_x_percent: int,
    add_shadow_flag: bool,
    shadow_blur: int,
    shadow_opacity: float,
    shadow_offset_x: int,
    shadow_offset_y: int,
) -> Image.Image:
    bg = background.copy().convert("RGBA")

    target_width = int(bg.width * width_ratio)
    scale = target_width / bottle.width
    target_height = int(bottle.height * scale)
    bottle_resized = bottle.resize((target_width, target_height), Image.LANCZOS)

    if horizontal_position == "Lewo":
        x = int(bg.width * 0.08)
    elif horizontal_position == "Środek":
        x = (bg.width - bottle_resized.width) // 2
    elif horizontal_position == "Prawo":
        x = bg.width - bottle_resized.width - int(bg.width * 0.08)
    else:
        x = int((bg.width - bottle_resized.width) * (custom_x_percent / 100))

    y = bg.height - bottle_resized.height - bottom_margin
    y = max(0, y)

    if add_shadow_flag:
        shadow, (sx, sy) = make_shadow(
            bottle_resized,
            blur_radius=shadow_blur,
            opacity=shadow_opacity,
            offset_x=shadow_offset_x,
            offset_y=shadow_offset_y,
        )
        shadow_x = x - (shadow.width - bottle_resized.width) // 2 + sx
        shadow_y = y - (shadow.height - bottle_resized.height) // 2 + sy
        bg.alpha_composite(shadow, (shadow_x, shadow_y))

    bg.alpha_composite(bottle_resized, (x, y))
    return bg


def image_to_bytes(img: Image.Image, output_format: str, jpeg_quality: int = 95) -> bytes:
    buf = io.BytesIO()
    if output_format == "JPG":
        rgb = img.convert("RGB")
        rgb.save(buf, format="JPEG", quality=jpeg_quality)
    else:
        img.save(buf, format="PNG")
    return buf.getvalue()


st.title("🍾 Agent AI do wycinania tła i wklejania butelek")
st.write(
    "Wgraj tło oraz zdjęcia butelek. Aplikacja usunie tło z butelek, "
    "wklei je na wybrane tło i przygotuje paczkę gotowych grafik."
)

with st.sidebar:
    st.header("Ustawienia")
    width_ratio = st.slider("Szerokość butelki względem tła", 0.08, 0.6, 0.22, 0.01)
    bottom_margin = st.slider("Margines od dołu (px)", 0, 300, 40, 5)
    horizontal_position = st.selectbox(
        "Pozycja pozioma",
        ["Środek", "Lewo", "Prawo", "Własna"],
        index=0,
    )
    custom_x_percent = 50
    if horizontal_position == "Własna":
        custom_x_percent = st.slider("Pozycja własna X (%)", 0, 100, 50, 1)

    st.subheader("Cień")
    add_shadow_flag = st.checkbox("Dodaj cień", value=True)
    shadow_blur = st.slider("Rozmycie cienia", 0, 60, 18, 1)
    shadow_opacity = st.slider("Przezroczystość cienia", 0.0, 1.0, 0.35, 0.05)
    shadow_offset_x = st.slider("Przesunięcie cienia X", -100, 100, 18, 1)
    shadow_offset_y = st.slider("Przesunięcie cienia Y", -100, 100, 22, 1)

    st.subheader("Eksport")
    output_format = st.selectbox("Format wyjściowy", ["PNG", "JPG"], index=0)

col1, col2 = st.columns(2)

with col1:
    background_files = st.file_uploader(
        "Wgraj tło / tła",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

with col2:
    bottle_files = st.file_uploader(
        "Wgraj zdjęcia butelek",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

if background_files:
    bg_names = [f.name for f in background_files]
    selected_bg_name = st.selectbox("Wybierz tło robocze", bg_names)
    selected_bg_file = next(f for f in background_files if f.name == selected_bg_name)
    background_img = load_image(selected_bg_file)
    st.image(background_img, caption=f"Wybrane tło: {selected_bg_name}", use_container_width=True)
else:
    background_img = None

if background_img and bottle_files:
    if st.button("🚀 Generuj grafiki", use_container_width=True):
        previews: List[Tuple[str, Image.Image]] = []
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            progress = st.progress(0)
            status = st.empty()

            for idx, bottle_file in enumerate(bottle_files, start=1):
                status.write(f"Przetwarzanie: {bottle_file.name}")

                bottle_original = load_image(bottle_file)
                bottle_cut = remove_background(bottle_original)

                result = compose_bottle_on_background(
                    background=background_img,
                    bottle=bottle_cut,
                    width_ratio=width_ratio,
                    bottom_margin=bottom_margin,
                    horizontal_position=horizontal_position,
                    custom_x_percent=custom_x_percent,
                    add_shadow_flag=add_shadow_flag,
                    shadow_blur=shadow_blur,
                    shadow_opacity=shadow_opacity,
                    shadow_offset_x=shadow_offset_x,
                    shadow_offset_y=shadow_offset_y,
                )

                stem = bottle_file.name.rsplit(".", 1)[0]
                ext = "jpg" if output_format == "JPG" else "png"
                file_name = f"{stem}_gotowe.{ext}"
                result_bytes = image_to_bytes(result, output_format=output_format)
                zip_file.writestr(file_name, result_bytes)
                previews.append((file_name, result))

                progress.progress(idx / len(bottle_files))

        status.success("Gotowe! Możesz pobrać paczkę ZIP albo pojedyncze pliki.")

        st.subheader("Podgląd wyników")
        for file_name, img in previews:
            st.image(img, caption=file_name, use_container_width=True)
            st.download_button(
                label=f"Pobierz {file_name}",
                data=image_to_bytes(img, output_format=output_format),
                file_name=file_name,
                mime="image/jpeg" if output_format == "JPG" else "image/png",
                key=file_name,
            )

        st.download_button(
            label="📦 Pobierz wszystko jako ZIP",
            data=zip_buffer.getvalue(),
            file_name="gotowe_grafiki.zip",
            mime="application/zip",
            use_container_width=True,
        )
else:
    st.info("Najpierw wgraj przynajmniej 1 tło i 1 zdjęcie butelki.")