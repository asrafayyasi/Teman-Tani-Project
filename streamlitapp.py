import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Fungsi untuk memproses gambar
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Fungsi untuk membuat prediksi
def predict_image(model, img):
    img_array = preprocess_image(img)
    prediction = model.predict(img_array)
    return prediction

# Muat model yang telah dilatih
model = tf.keras.models.load_model('my_model_mobilenetV2.h5')

# Nama kelas
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Header aplikasi dengan kustomisasi warna
st.title("Teman Tani Project")
st.markdown("Prediksi Penyakit Tanaman Padi", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Tanaman Padi", type=["jpg", "png"])

# Sidebar untuk unggah gambar dengan kustomisasi warna
st.sidebar.header("Teman Tani Team:")
st.sidebar.markdown("1. (ML) M180D4KY2897 – Asraf Ayyasi Putra – Universitas Airlangga", unsafe_allow_html=True)
st.sidebar.markdown("2. (ML) M281D4KY1762– Fikri Fahreza– Universitas Negeri Medan", unsafe_allow_html=True)
st.sidebar.markdown("3. (ML) M004D4KY1642 – Imam Nur Rizky Gusman – Institut Teknologi Sepuluh Nopember", unsafe_allow_html=True)
st.sidebar.markdown("4. (CC) C180D4KY0808 - Moch Ilyas Saktiono Putra – Universitas Airlangga", unsafe_allow_html=True)
st.sidebar.markdown("5. (CC) C004D4KX0753– Fathika Afrine Azaruddin – Institut Teknologi Sepuluh Nopember", unsafe_allow_html=True)
st.sidebar.markdown("6. (MD) A180D4KX3849 – Anis Nur Fitria – Universitas Airlangga", unsafe_allow_html=True)
st.sidebar.markdown("7. (MD) A553D4KY4253 – Yonatan Hidson Simbolon  – Universitas Advent Indonesia", unsafe_allow_html=True)

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah di sidebar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Melakukan prediksi
    prediction = predict_image(model, img)
    predicted_class = np.argmax(prediction[0])

    # Tampilkan hasil prediksi di sidebar
    st.markdown("Prediksi:")
    st.write(f"**{class_names[predicted_class]}**")


# Tambahkan kustomisasi CSS dengan warna yang Anda inginkan
# st.markdown(
#     """
#     <style>
#     .main {
#         background-color: #7AB2B2; /* Warna background main */
#         color: black;
#     }
#     h1 {
#         color: #2e7d32; /* Warna tulisan Teman Tani Project */
#     }
#     h2, h3 {
#         color: #EEF7FF; /* Warna tulisan Teman Tani Team */
#     }
#     .stButton>button {
#         background-color: #66bb6a;
#         color: white;
#     }
    
#     .stMarkdown p {
#         color: #CDE8E5; /* Warna tulisan Prediksi Penyakit Tanaman Padi, Upload Tanaman Padi */
#     }

#     .sidebar .sidebar-content {
#         background-color: #4D869C; /* Warna background sidebar */
#     }

#     .sidebar .sidebar-content .element-container {
#         color: #CDE8E5; /* Warna tulisan Teman Tani Team */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
