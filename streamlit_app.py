# streamlit_app.py
import os
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 0) 페이지/스타일 설정
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기 (스냅샷)", page_icon="🤖")

st.markdown("""
<style>
h1 { color: #1E88E5; text-align: center; font-weight: bold; }
.stFileUploader, .stCameraInput {
  border: 2px dashed #1E88E5; border-radius: 10px; padding: 15px; background-color: #f5fafe;
}
.prediction-box {
  background-color: #E3F2FD; border: 2px solid #1E88E5; border-radius: 10px;
  padding: 25px; text-align: center; margin: 20px 0; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.prediction-box h2 { color: #0D47A1; margin: 0; font-size: 2.0rem; }
.prob-card {
  background-color: #FFFFFF; border-radius: 8px; padding: 15px; margin: 10px 0;
  box-shadow: 0 2px 5px rgba(0,0,0,0.08); transition: transform 0.2s ease;
}
.prob-card:hover { transform: translateY(-3px); }
.prob-label { font-weight: bold; font-size: 1.05rem; color: #333; }
.prob-bar-bg { background-color: #E0E0E0; border-radius: 5px; width: 100%; height: 22px; overflow: hidden; }
.prob-bar-fg {
  background-color: #4CAF50; height: 100%; border-radius: 5px 0 0 5px; text-align: right;
  padding-right: 8px; color: white; font-weight: bold; line-height: 22px; transition: width 0.5s ease-in-out;
}
.prob-bar-fg.highlight { background-color: #FF6F00; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 카메라 스냅샷/파일 업로드")

# ======================
# 1) 세션 상태 초기화
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None

# ======================
# 2) 모델 로드 (Google Drive)
# ======================
# secrets.toml에 넣어두면 편리합니다. 없으면 기본값 사용
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1uj2lD8goJDLo9uSg_8HcT4bxnl2trPc8")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    # CPU 강제 로드
    learner = load_learner(output_path, cpu=True)
    return learner

with st.spinner("🤖 AI 모델을 불러오는 중입니다..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 3) 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])

new_bytes = None

with tab_cam:
    st.write("카메라 권한을 허용한 뒤, 스냅샷을 촬영하세요.")
    camera_photo = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if camera_photo is not None:
        new_bytes = camera_photo.getvalue()

with tab_file:
    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
        type=["jpg", "png", "jpeg", "webp", "tiff"]
    )
    if uploaded_file is not None:
        new_bytes = uploaded_file.getvalue()

# 리런에도 유지되도록 세션에 저장
if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 4) 전처리/유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    """EXIF 회전 보정 + RGB 강제."""
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    return pil

# ======================
# 5) 사이드바 옵션(선택)
# ======================
with st.sidebar:
    st.header("설정")
    resize_on = st.toggle("입력 리사이즈 사용", value=False, help="느리다면 켜서 속도를 개선하세요.")
    target_size = st.slider("리사이즈 한 변 길이", min_value=128, max_value=1024, value=384, step=32)
    st.caption("모델 파일은 최초 1회 다운로드 후 캐시됩니다.")
    st.write(f"**모델 파일**: `{MODEL_PATH}`")
    st.write(f"**Drive File ID**: `{FILE_ID}`")

# ======================
# 6) 예측 파이프라인
# ======================
if st.session_state.img_bytes:
    col1, col2 = st.columns([1, 1], vertical_alignment="top")

    # PIL 로드 + 선택적 리사이즈
    try:
        pil_img = load_pil_from_bytes(st.session_state.img_bytes)
        if resize_on:
            # 작은 변 기준 비율 유지 리사이즈(간단 구현: 정사각 리사이즈)
            pil_img = pil_img.resize((target_size, target_size))
    except Exception as e:
        st.exception(e)
        st.stop()

    with col1:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    # fastai 입력: numpy 배열 → PILImage.create
    try:
        fa_img = PILImage.create(np.array(pil_img))
    except Exception as e:
        st.exception(e)
        st.stop()

    with st.spinner("🧠 이미지를 분석 중입니다..."):
        try:
            prediction, pred_idx, probs = learner.predict(fa_img)
        except Exception as e:
            st.exception(e)
            st.stop()

    with col1:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size: 1.0rem; color: #555;">예측 결과:</span>
                <h2>{prediction}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("<h3>상세 예측 확률:</h3>", unsafe_allow_html=True)

        # 파이썬 float로 변환 후 내림차순 정렬
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1],
            reverse=True
        )

        for label, prob in prob_list:
            highlight_class = "highlight" if label == str(prediction) else ""
            prob_percent = prob * 100.0
            st.markdown(
                f"""
                <div class="prob-card">
                    <span class="prob-label">{label}</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fg {highlight_class}" style="width: {prob_percent:.4f}%;">
                            {prob_percent:.2f}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("카메라에서 스냅샷을 촬영하거나, 파일을 업로드하면 AI가 분석을 시작합니다.")
