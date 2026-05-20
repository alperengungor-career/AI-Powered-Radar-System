import cv2
import numpy as np
import re
import tempfile
import streamlit as st
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

# WEB ARAYÜZÜ BAŞLIĞI VE AYARLARI
st.set_page_config(page_title="AI-Powered Radar System", layout="wide")
st.title("📹 AI-Powered Radar (Plaka Tanıma) Sistemi")
st.write("YOLOv8 ve Oylama Algoritması ile Canlı Video Üzerinden Plaka Tespiti")

# MODELLERİ YÜKLE
# Hugging Face üzerinde modeller klasör içinde yan yana duracağı için doğrudan yüklenir
@st.cache_resource
def load_models():
    detector = YOLO('best_detection.pt')
    recognizer = YOLO('best_reading.pt')
    return detector, recognizer

try:
    plate_detector, char_recognizer = load_models()
    st.success("YOLO Modelleri başarıyla yüklendi!")
    models_loaded = True
except Exception as e:
    st.error(f"Modeller yüklenirken hata oluştu! Lütfen .pt dosyalarının yüklendiğinden emin olun. Hata: {e}")
    models_loaded = False

# ALGORİTMA AYARLARI
plate_pattern = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[0-1])"
    r"([A-Z]{1}[0-9]{4}|[A-Z]{2}[0-9]{3,4}|[A-Z]{3}[0-9]{2,3})$"
)
correction_map = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}

VOTE_THRESHOLD = 1
REEVAL_INTERVAL = 5

# KULLANICIDAN VİDEO ALMA ALANI
uploaded_file = st.file_uploader("Sisteme işlemek için bir MP4 videosu yükleyin...", type=["mp4", "avi", "mov"])

if uploaded_file is not None and models_loaded:
    # Yüklenen videoyu geçici bir dosyaya kaydet (OpenCV'nin okuyabilmesi için)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    
    # Hafıza sözlükleri
    vehicle_votes = {}       
    locked_vehicles = {}     
    eval_checkpoints = {}    
    logged_plates = []       

    # Ekranda yan yana iki sütun oluştur (Sol: Video, Sağ: Tespit Edilen Plakalar)
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Canlı İşleme Akışı")
        frame_placeholder = st.empty() # Videonun kare kare güncelleneceği yer
        
    with col2:
        st.subheader("📋 Okunan Plaka Günlüğü")
        log_placeholder = st.empty()

    # ANA VİDEO İŞLEME DÖNGÜSÜ 
    frame_count = 0
    frame_skip = 5  # CPU YÜKÜNÜ HAFİFLETEN AYAR: Her 5 kareden sadece 1'ini işler.

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue  # Bu kareyi pas geç, sunucuyu yorma!

        # DİKKAT: Burada çözünürlüğü düşürmüyoruz ki yapay zeka plakayı net okusun!
        
        # Plaka Tespiti ve Takibi
        results = plate_detector.track(frame, persist=True, imgsz=640, conf=0.5, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Orijinal çözünürlükten kırpma yapıyoruz
                plate_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                current_color = (0, 165, 255) # Turuncu
                display_text = "Taraniyor..."
                status_text = "Format Bekleniyor"
                clean_plate = ""
                
                if plate_crop.size > 0:
                    char_results = char_recognizer(plate_crop, imgsz=320, conf=0.4, verbose=False)
                    
                    detected_chars = []
                    for c_box in char_results[0].boxes:
                        c_x1 = c_box.xyxy[0][0].item()
                        c_label = char_recognizer.names[int(c_box.cls.item())].upper()
                        detected_chars.append((c_x1, c_label))
                    
                    detected_chars.sort(key=lambda x: x[0])
                    raw_text = "".join([c[1] for c in detected_chars]).replace(" ", "")

                    for i, char in enumerate(raw_text):
                        if i < 2 and char in correction_map:
                            clean_plate += correction_map[char]
                        else:
                            clean_plate += char

                    total_votes = len(vehicle_votes.get(track_id, []))
                    
                    if clean_plate and plate_pattern.match(clean_plate):
                        if track_id not in vehicle_votes:
                            vehicle_votes[track_id] = []
                        
                        vehicle_votes[track_id].append(clean_plate)
                        total_votes = len(vehicle_votes[track_id])
                        
                        vote_counts = Counter(vehicle_votes[track_id])
                        most_common_plate, highest_count = vote_counts.most_common(1)[0]

                        # SENARYO A: HENÜZ KİLİTLENMEDİ
                        if track_id not in locked_vehicles:
                            display_text = clean_plate 
                            status_text = f"Voting: {highest_count}/{VOTE_THRESHOLD}"
                            
                            if highest_count >= VOTE_THRESHOLD and most_common_plate:
                                locked_vehicles[track_id] = most_common_plate
                                eval_checkpoints[track_id] = total_votes
                                
                                current_color = (0, 255, 0) # Yeşil
                                display_text = most_common_plate
                                status_text = f"LOCKED"
                                
                                time_str = datetime.now().strftime('%H:%M:%S')
                                logged_plates.append(f"[{time_str}] 🚗 Araç #{track_id} -> {most_common_plate}")

                        # SENARYO B: ZATEN KİLİTLİ
                        else:
                            current_color = (0, 255, 0)
                            display_text = locked_vehicles[track_id]
                            votes_since_last_check = total_votes - eval_checkpoints[track_id]
                            status_text = "LOCKED"
                            
                            if votes_since_last_check >= REEVAL_INTERVAL:
                                eval_checkpoints[track_id] = total_votes
                                
                                if most_common_plate != locked_vehicles[track_id] and most_common_plate:
                                    old_plate = locked_vehicles[track_id]
                                    locked_vehicles[track_id] = most_common_plate
                                    
                                    time_str = datetime.now().strftime('%H:%M:%S')
                                    logged_plates.append(f"[{time_str}] 🔄 Güncellendi #{track_id} -> {most_common_plate} (Eski: {old_plate})")

                    else:
                        if track_id in locked_vehicles:
                            current_color = (0, 255, 0)
                            display_text = locked_vehicles[track_id]
                            status_text = "LOCKED"
                        else:
                            current_color = (0, 165, 255)
                            display_text = clean_plate if clean_plate else "Taraniyor..."
                            highest_count = Counter(vehicle_votes[track_id]).most_common(1)[0][1] if total_votes > 0 else 0
                            status_text = f"Voting: {highest_count}/{VOTE_THRESHOLD}"

                # UI Çizimleri (Orijinal karenin üzerine çiziyoruz)
                cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
                cv2.putText(frame, f"{display_text} ({status_text})", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)

        # Web arayüzüne gönderirken donma yapmasın diye BURADA küçültüyoruz
        display_frame = cv2.resize(frame, (720, 405))
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Sağ paneldeki plaka listesini güncelle
        log_placeholder.markdown("\n".join([f" `{p}`" for p in reversed(logged_plates)]))

    cap.release()
    st.success("🎬 Video işleme tamamlandı!")