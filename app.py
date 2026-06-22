import base64
import io
import json
import tempfile
from typing import Optional, Tuple, Dict, Any, List

import streamlit as st
from PIL import Image
from openai import OpenAI


# =========================
# KONFIGURACJA
# =========================
IMAGE_MODEL = "gpt-image-1"
VISION_MODEL = "gpt-4o-mini"

st.set_page_config(page_title="Bottle Scene AI Pro", layout="wide")


# =========================
# POMOCNICZE
# =========================
@st.cache_resource
def get_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Brak OPENAI_API_KEY w Streamlit Secrets.")
    return OpenAI(api_key=api_key)


def uploaded_file_to_pil(uploaded_file) -> Image.Image:
    uploaded_file.seek(0)
    img = Image.open(uploaded_file).convert("RGBA")
    return img


def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def uploaded_file_to_png_bytes(uploaded_file) -> bytes:
    img = uploaded_file_to_pil(uploaded_file)
    return pil_to_png_bytes(img)


def bytes_to_temp_file(data: bytes, suffix: str = ".png") -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.flush()
    tmp.close()
    return tmp.name


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGBA")


def decode_b64_image(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return bytes_to_pil(image_bytes)


def build_generation_prompt(
    has_reference: bool,
    user_notes: str,
    retry_feedback: str = ""
) -> str:
    reference_part = ""
    if has_reference:
        reference_part = (
            "Image C is a visual reference for framing, scale, bottle position, "
            "camera angle, and realism. Match that overall look as closely as possible, "
            "while preserving the actual product from Image B."
        )

    retry_part = ""
    if retry_feedback:
        retry_part = (
            "\nAdditional correction instructions based on the previous quality check:\n"
            f"{retry_feedback}\n"
        )

    prompt = f"""
Create a realistic product photograph.

Image A is the background scene and should remain the real background.
Image B is the source product photo and must be preserved as the real product identity.
{reference_part}

Goal:
Place the product from Image B into the background from Image A so the final image looks
like a real photo taken with a camera, not like a pasted object.

Important rules:
- Preserve the actual product identity from Image B.
- Preserve bottle shape, packaging shape, label type, general product proportions,
  and whether the product includes a box/carton/tube.
- If the source image contains both bottle and carton, keep both in the final result.
- Make the product look naturally placed on the surface.
- Add realistic contact shadow and natural lighting.
- Match reflections, perspective, depth, and color temperature to the background.
- The final composition should be clean and premium, like an e-commerce or catalog photo.
- Keep the product at a realistic scale.
- If a reference image is provided, keep a similar composition, size, framing,
  central positioning, and camera feel.
- Make the result look believable and photographic.
- Do not stylize it as illustration or CGI.
- Avoid obvious distortions in the label and packaging.
- Make the product sharp and clear.

User notes:
{user_notes if user_notes else "No extra notes."}
{retry_part}
""".strip()

    return prompt


def generate_scene(
    client: OpenAI,
    background_file,
    product_file,
    reference_file=None,
    user_notes: str = "",
    retry_feedback: str = "",
    size: str = "1024x1024"
) -> Tuple[Image.Image, str]:
    bg_bytes = uploaded_file_to_png_bytes(background_file)
    product_bytes = uploaded_file_to_png_bytes(product_file)

    bg_path = bytes_to_temp_file(bg_bytes, ".png")
    product_path = bytes_to_temp_file(product_bytes, ".png")

    image_inputs = [
        open(bg_path, "rb"),
        open(product_path, "rb"),
    ]

    has_reference = reference_file is not None
    ref_path = None
    if has_reference:
        ref_bytes = uploaded_file_to_png_bytes(reference_file)
        ref_path = bytes_to_temp_file(ref_bytes, ".png")
        image_inputs.append(open(ref_path, "rb"))

    prompt = build_generation_prompt(
        has_reference=has_reference,
        user_notes=user_notes,
        retry_feedback=retry_feedback
    )

    try:
        result = client.images.edit(
            model=IMAGE_MODEL,
            image=image_inputs,
            prompt=prompt,
            size=size,
            quality="high",
        )
    finally:
        for f in image_inputs:
            try:
                f.close()
            except Exception:
                pass

    generated_b64 = result.data[0].b64_json
    generated_img = decode_b64_image(generated_b64)
    return generated_img, prompt


def verify_scene(
    client: OpenAI,
    background_file,
    product_file,
    generated_img: Image.Image,
    reference_file=None,
) -> Dict[str, Any]:
    background_img = uploaded_file_to_pil(background_file)
    product_img = uploaded_file_to_pil(product_file)

    bg_data_url = pil_to_data_url(background_img)
    product_data_url = pil_to_data_url(product_img)
    generated_data_url = pil_to_data_url(generated_img)

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "You are a strict quality-control reviewer for e-commerce alcohol product photography. "
                "Evaluate the final generated image.\n\n"
                "Image A = target background.\n"
                "Image B = source product photo.\n"
                "Image C = optional reference composition.\n"
                "Image D = generated final image.\n\n"
                "Check:\n"
                "1. Product correctness\n"
                "2. Whether bottle and packaging/carton/tube are preserved if present\n"
                "3. Realism / whether it looks like a real camera photo\n"
                "4. Lighting, shadow, reflections, and integration with background\n"
                "5. Similarity of scale/position/framing to the reference if reference exists\n"
                "6. Label and packaging distortion risk\n\n"
                "Return ONLY valid JSON with this schema:\n"
                "{\n"
                '  "overall_score": 0-10,\n'
                '  "pass": true_or_false,\n'
                '  "product_correctness_score": 0-10,\n'
                '  "realism_score": 0-10,\n'
                '  "reference_match_score": 0-10,\n'
                '  "label_risk": "low|medium|high",\n'
                '  "packaging_present": true_or_false,\n'
                '  "summary": "short summary",\n'
                '  "issues": ["issue1", "issue2"],\n'
                '  "suggested_fix_prompt": "short correction prompt for a retry"\n'
                "}\n"
            ),
        },
        {"type": "text", "text": "Image A: background"},
        {"type": "image_url", "image_url": {"url": bg_data_url}},
        {"type": "text", "text": "Image B: source product"},
        {"type": "image_url", "image_url": {"url": product_data_url}},
    ]

    if reference_file is not None:
        reference_img = uploaded_file_to_pil(reference_file)
        ref_data_url = pil_to_data_url(reference_img)
        content.extend([
            {"type": "text", "text": "Image C: composition reference"},
            {"type": "image_url", "image_url": {"url": ref_data_url}},
        ])

    content.extend([
        {"type": "text", "text": "Image D: generated final image"},
        {"type": "image_url", "image_url": {"url": generated_data_url}},
    ])

    response = client.chat.completions.create(
        model=VISION_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You return only strict JSON. No markdown. No extra commentary."
            },
            {
                "role": "user",
                "content": content
            }
        ],
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    try:
        report = json.loads(raw)
    except Exception:
        report = {
            "overall_score": 0,
            "pass": False,
            "product_correctness_score": 0,
            "realism_score": 0,
            "reference_match_score": 0,
            "label_risk": "high",
            "packaging_present": False,
            "summary": "Nie udało się poprawnie sparsować odpowiedzi QA.",
            "issues": ["Błąd parsowania raportu QA"],
            "suggested_fix_prompt": "Preserve the product more accurately and improve realism."
        }

    return report


def generate_with_auto_qc(
    client: OpenAI,
    background_file,
    product_file,
    reference_file=None,
    user_notes: str = "",
    size: str = "1024x1024",
    auto_retry: bool = True,
    max_attempts: int = 2
):
    attempts = []
    best_result = None
    best_report = None
    retry_feedback = ""

    total_attempts = max_attempts if auto_retry else 1

    for attempt_no in range(1, total_attempts + 1):
        generated_img, used_prompt = generate_scene(
            client=client,
            background_file=background_file,
            product_file=product_file,
            reference_file=reference_file,
            user_notes=user_notes,
            retry_feedback=retry_feedback,
            size=size,
        )

        report = verify_scene(
            client=client,
            background_file=background_file,
            product_file=product_file,
            reference_file=reference_file,
            generated_img=generated_img,
        )

        attempts.append({
            "attempt_no": attempt_no,
            "image": generated_img,
            "report": report,
            "prompt": used_prompt,
        })

        score = int(report.get("overall_score", 0))
        best_score = -1 if best_report is None else int(best_report.get("overall_score", 0))

        if best_result is None or score > best_score:
            best_result = generated_img
            best_report = report

        if report.get("pass", False):
            break

        retry_feedback = report.get(
            "suggested_fix_prompt",
            "Preserve the product more accurately, improve realism, and keep packaging."
        )

    return best_result, best_report, attempts


def image_download_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# =========================
# UI
# =========================
st.title("🍾 Bottle Scene AI Pro")
st.write(
    "Aplikacja generuje realistyczne zdjęcie produktowe na bazie tła, produktu i opcjonalnego zdjęcia wzorcowego. "
    "Następnie sama ocenia wynik i może zrobić automatyczną poprawkę."
)

with st.sidebar:
    st.header("Ustawienia")

    size = st.selectbox(
        "Rozmiar obrazu",
        ["1024x1024", "1536x1024", "1024x1536"],
        index=0,
    )

    auto_retry = st.checkbox("Auto-poprawka po kontroli jakości", value=True)

    max_attempts = st.slider(
        "Maksymalna liczba prób",
        min_value=1,
        max_value=3,
        value=2,
        step=1,
    )

    st.caption("Więcej prób = lepsza szansa na dobry efekt, ale też większy koszt API.")


col1, col2 = st.columns(2)

with col1:
    background_file = st.file_uploader(
        "Wgraj tło",
        type=["png", "jpg", "jpeg", "webp"]
    )

    reference_file = st.file_uploader(
        "Wgraj zdjęcie wzorcowe (opcjonalnie)",
        type=["png", "jpg", "jpeg", "webp"]
    )

with col2:
    product_file = st.file_uploader(
        "Wgraj zdjęcie produktu",
        type=["png", "jpg", "jpeg", "webp"]
    )

    user_notes = st.text_area(
        "Dodatkowe instrukcje",
        value=(
            "Produkt ma wyglądać jak prawdziwe zdjęcie zrobione aparatem. "
            "Ustawienie podobne do zdjęcia wzorcowego: naturalna skala, realistyczne światło, "
            "brak sztucznego efektu wklejenia."
        ),
        height=130
    )


if background_file is not None:
    st.subheader("Podgląd tła")
    st.image(background_file, use_container_width=True)

if product_file is not None:
    st.subheader("Podgląd produktu")
    st.image(product_file, use_container_width=True)

if reference_file is not None:
    st.subheader("Podgląd zdjęcia wzorcowego")
    st.image(reference_file, use_container_width=True)


if st.button("🚀 Generuj realistyczne zdjęcie", use_container_width=True):
    if background_file is None or product_file is None:
        st.error("Wgraj tło i zdjęcie produktu.")
    else:
        try:
            client = get_client()

            with st.spinner("Generuję scenę i sprawdzam jakość..."):
                final_img, final_report, attempts = generate_with_auto_qc(
                    client=client,
                    background_file=background_file,
                    product_file=product_file,
                    reference_file=reference_file,
                    user_notes=user_notes,
                    size=size,
                    auto_retry=auto_retry,
                    max_attempts=max_attempts,
                )

            st.success("Gotowe.")

            st.subheader("Finalny wynik")
            st.image(final_img, use_container_width=True)

            st.download_button(
                label="Pobierz obraz PNG",
                data=image_download_bytes(final_img),
                file_name="final_product_scene.png",
                mime="image/png",
                use_container_width=True,
            )

            st.subheader("Raport jakości")
            score = final_report.get("overall_score", 0)
            passed = final_report.get("pass", False)

            c1, c2, c3 = st.columns(3)
            c1.metric("Ocena ogólna", score)
            c2.metric("Realizm", final_report.get("realism_score", 0))
            c3.metric("Zgodność produktu", final_report.get("product_correctness_score", 0))

            st.write(f"**Status:** {'✅ Akceptowalne' if passed else '⚠️ Wymaga uwagi'}")
            st.write(f"**Podsumowanie:** {final_report.get('summary', '-')}")
            st.write(f"**Ryzyko etykiety:** {final_report.get('label_risk', '-')}")
            st.write(f"**Czy packaging obecny:** {final_report.get('packaging_present', False)}")

            issues = final_report.get("issues", [])
            if issues:
                st.write("**Wykryte uwagi:**")
                for issue in issues:
                    st.write(f"- {issue}")

            with st.expander("Pokaż wszystkie próby"):
                for item in attempts:
                    st.markdown(f"### Próba {item['attempt_no']}")
                    st.image(item["image"], use_container_width=True)
                    st.json(item["report"])

            with st.expander("Pokaż prompt użyty do generacji"):
                if attempts:
                    st.code(attempts[-1]["prompt"])

        except Exception as e:
            st.error(f"Wystąpił błąd: {e}")
            st.info(
                "Jeśli błąd dotyczy modelu obrazu, upewnij się, że Twoje konto API ma dostęp do modelu "
                f"'{IMAGE_MODEL}' i że klucz API został poprawnie dodany do Streamlit Secrets."
            )
