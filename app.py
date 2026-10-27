import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("yolov8_best.pt")

model = load_model()
confidence_threshold = 0.15

st.title("🚧 Deteksi Jalan Berlubang & Retak (YOLOv8)")
st.write("Upload gambar untuk melakukan deteksi")

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # =========================
    # SIMPAN FILE SEMENTARA
    # =========================
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    # =========================
    # BACA GAMBAR ASLI
    # =========================
    img_original = cv2.imread(temp_path)

    if img_original is None:
        st.error("Gagal membaca gambar")
    else:
        # ==============================================================
        # KONVERSI WARNA KE RGB SEBELUM DIGAMBAR KOTAK/TEKS
        # ==============================================================
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_annotated_rgb = img_original_rgb.copy()

        # =========================
        # DETEKSI VIA YOLO
        # =========================
        results = model.predict(source=temp_path, conf=confidence_threshold)
        result = results[0] # Bounded index ke-0 untuk membaca output list YOLO

        detected_classes = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(cls)
                class_name = model.names[class_id]
                confidence = float(conf)

                detected_classes.append(class_name.lower())

                # ==============================================================
                # PENENTUAN WARNA (FORMAT RGB LANGSUNG)
                # ==============================================================
                if class_id == 0:
                    color = (255, 0, 0)   # Merah murni di RGB
                elif class_id == 1:
                    color = (0, 255, 0)   # Hijau murni di RGB
                else:
                    color = (0, 255, 255) # Cyan / Kuning di RGB

                label = f"{class_name.lower()} {confidence:.2f}"

                # ==============================================================
                # PROSES GAMBAR BOX & TEKS (DIATAS GAMBAR RGB)
                # ==============================================================
                cv2.rectangle(img_annotated_rgb, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_annotated_rgb, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2)

            # =========================
            # HASIL KLASIFIKASI TEXT
            # =========================
            detected_classes = list(set(detected_classes))

            # Mapping ke bahasa Indonesia
            mapping_label = {
                "pothole": "Jalan Berlubang",
                "crack": "Jalan Retak"
            }

            detected_classes = [mapping_label.get(cls, cls) for cls in detected_classes]

            if len(detected_classes) == 1:
                st.success(f"Terdeteksi: {detected_classes[0]}")
            else:
                st.success(f"Terdeteksi beberapa objek: {', '.join(detected_classes)}")

        else:
            st.warning("Tidak ada objek terdeteksi")

        # =========================
        # TAMPILKAN KE STREAMLIT
        # =========================
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_original_rgb, caption="Gambar Asli", use_column_width=True)

        with col2:
            st.image(img_annotated_rgb, caption="Hasil Deteksi dan Klasifikasi", use_column_width=True)

    # Hapus file sementara dari memori server
    os.remove(temp_path)
