import streamlit as st
import torch
import clip
from PIL import Image
from transformers import GPT2Tokenizer
from clipcap import ClipCaptionModel

# Thiết bị (CPU hoặc GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model và preprocess
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Load tokenizer GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load ClipCaptionModel
model_path = "pytorch_model.pt"  # Đường dẫn tới mô hình ClipCap
prefix_length = 10  # Độ dài prefix của mô hình
model = ClipCaptionModel(prefix_length, tokenizer=tokenizer)

# Chỉ load trọng số (weights)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model = model.eval().to(device)

# Thiết lập tiêu đề ứng dụng
st.set_page_config(page_title="Sinh Caption Ảnh - CLIP + ClipCap", page_icon=":camera:")

# Thanh bên: tải ảnh, thông tin
st.sidebar.title("Tải Ảnh Lên")
uploaded_file = st.sidebar.file_uploader("Chọn một hình ảnh", type=["png", "jpg", "jpeg"])

st.sidebar.markdown("---")
st.sidebar.markdown("*Hướng dẫn:*")
st.sidebar.markdown("""
1. Tải lên một hình ảnh.
2. Đợi hệ thống xử lý.
3. Xem kết quả caption sinh ra.
""")

# Nội dung chính
st.markdown("<h1 style='text-align: center;'>Hệ thống sinh caption từ ảnh</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Ứng dụng này sử dụng mô hình CLIP và ClipCap để sinh caption mô tả nội dung ảnh bạn tải lên.</p>", unsafe_allow_html=True)
st.markdown("---")

if uploaded_file is not None:
    # Hiển thị ảnh
    st.image(uploaded_file, caption="Ảnh đã chọn", use_container_width=True)

    # Xử lý sinh caption
    with torch.no_grad():
        with st.spinner("Đang sinh caption, vui lòng đợi..."):
            image = Image.open(uploaded_file).convert("RGB")
            image_preprocessed = preprocess(image).unsqueeze(0).to(device)

            # Trích xuất đặc trưng bằng CLIP
            prefix = clip_model.encode_image(image_preprocessed).to(device, dtype=torch.float32)
            prefix_embed = model.clip_project(prefix).reshape(1, prefix_length, -1)

            # Sinh caption
            caption = model.generate_beam(embed=prefix_embed, tokenizer=tokenizer)[0]

    st.markdown("*Caption sinh ra:*")
    st.info(caption)
else:
    st.markdown("<p style='text-align:center; font-style: italic;'>Vui lòng tải lên một hình ảnh để bắt đầu.</p>", unsafe_allow_html=True)