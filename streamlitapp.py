import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

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
model = tf.keras.models.load_model('my_model.h5')

# Nama kelas
class_names = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

# Header aplikasi
st.title("Aplikasi Prediksi Penyakit Daun Padi")

# Unggah gambar
uploaded_file = st.file_uploader("Unggah gambar daun padi", type=["jpg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang diunggah", use_column_width=True)

    # Melakukan prediksi
    prediction = predict_image(model, img)
    predicted_class = np.argmax(prediction[0])
    
    # Tampilkan hasil prediksi
    st.write(f"Prediksi: {class_names[predicted_class]}")

# Menjalankan aplikasi Streamlit:
# Simpan kode ini dalam file `app.py`, lalu jalankan perintah berikut di terminal:
# streamlit run app.py
