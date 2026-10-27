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
    return YOLO("weight/yolov8_best.pt")

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
        img_annotated = img_original.copy()

        # =========================
        # DETEKSI
        # =========================
        results = model.predict(source=temp_path, conf=confidence_threshold)
        result = results[0]

        detected_classes = []

        if result.boxes is not None and len(result.boxes) > 0:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):

                x1, y1, x2, y2 = map(int, box)

                class_id = int(cls)
                class_name = model.names[class_id]
                confidence = float(conf)

                detected_classes.append(class_name.lower())

                # =========================
                # WARNA
                # =========================
                if class_id == 0:
                    color = (0, 0, 255)   # merah
                elif class_id == 1:
                    color = (0, 255, 0)   # hijau
                else:
                    color = (255, 255, 0)

                label = f"{class_name.lower()} {confidence:.2f}"

                # BOX + TEXT
                cv2.rectangle(img_annotated, (x1, y1), (x2, y2), color, 4)
                cv2.putText(img_annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            color,
                            4)

            # =========================
            # HASIL KLASIFIKASI
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
        # KONVERSI WARNA
        # =========================
        img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

        # =========================
        # TAMPILKAN
        # =========================
        col1, col2 = st.columns(2)

        with col1:
            st.image(img_original_rgb, caption="Gambar Asli", use_column_width=True)

        with col2:
            st.image(img_annotated_rgb, caption="Hasil Deteksi dan Klasifikasi", use_column_width=True)

    # Hapus file sementara
    os.remove(temp_path)
