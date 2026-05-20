import cv2
import numpy as np
import re
from datetime import datetime
from collections import Counter
from ultralytics import YOLO

# 1. MODELLERİ YÜKLE
# Plaka tespiti (Detector) ve Okuma (Recognizer) modelleri
plate_detector = YOLO('best_detection.pt') 
char_recognizer = YOLO('best_reading.pt') 

# 2. AYARLAR VE GLOBAL DEĞİŞKENLER
video_path = 'test3.mp4'
log_file = 'tespit_edilen_plakalar.txt'
cap = cv2.VideoCapture(video_path)

# Türkiye Plaka Standartları Regex (1 Harf 4 Rakam | 2 Harf 3-4 Rakam | 3 Harf 2-3 Rakam)
plate_pattern = re.compile(
    r"^(0[1-9]|[1-7][0-9]|8[0-1])"
    r"([A-Z]{1}[0-9]{4}|[A-Z]{2}[0-9]{3,4}|[A-Z]{3}[0-9]{2,3})$"
)

# Karakter düzeltme haritası (İlk 2 hanedeki O/0, I/1 karışıklığını önler)
correction_map = {'O': '0', 'I': '1', 'S': '5', 'B': '8'}

# Hafıza sözlüklerinin içi tamamen boş başlatılıyor (Eski kalıntıları kesinlikle önler)
vehicle_votes = {}       # {track_id: [sandıktaki_tüm_geçerli_plakalar]}
locked_vehicles = {}     # {track_id: "GÜNCEL_KAZANAN_PLAKA"}
eval_checkpoints = {}    # {track_id: son_sayim_yapilan_toplam_oy}

# TXT Loglama Yönetimi (Sadece son güncellenmiş halleri tutar)
logged_plates = {}       # Başlangıçta kesinlikle boş olmalı

# Algoritma Parametreleri
VOTE_THRESHOLD = 1       # İlk kilitleme ve yeşile dönme için gereken oy sayısı
REEVAL_INTERVAL = 5      # Kilitlendikten sonra her 5 geçerli oyda bir sandığı tekrar say

# 3. DOSYA GÜNCELLEME VE OTURUM FONKSİYONLARI
def update_log_file():
    """Tüm güncel doğrulanmış plakaları dosyaya baştan yazar (Sadece son halleri kalır)."""
    with open(log_file, "w", encoding="utf-8") as f:
        separator = "=" * 60
        f.write(f"{separator}\n")
        f.write(f">>> OTURUM AKTİF (Güncel Plaka Listesi)\n")
        f.write(f"{separator}\n\n")
        
        for t_id, data in logged_plates.items():
            f.write(f"[{data['time']}] {data['plate']}\n")

def log_session_marker(status):
    """Kapanışta log dosyasına son durumu ekler."""
    with open(log_file, "a", encoding="utf-8") as f:
        separator = "=" * 60
        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n{separator}\n")
        f.write(f">>> OTURUM {status}: {time_str}\n")
        f.write(f"{separator}\n")

# Program başlarken log dosyasını temiz bir başlangıçla oluştur
update_log_file()
print("--- AI Radar: Oturum Başlatıldı. Algoritma (1-5) ve Dinamik Loglama Aktif... ---")

# 4. ANA DÖNGÜ
try:
    while cap.isOpened():
        success, frame = cap.read()
        # Eğer video karesi okunamadıysa veya akış bittiyse döngüyü kır
        if not success or frame is None: 
            break

        # Plaka Tespiti ve Takibi
        results = plate_detector.track(frame, persist=True, imgsz=640, conf=0.5, verbose=False)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()
            
            for box, track_id in zip(boxes, ids):
                track_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Plakayı ana kadrajdan güvenli koordinatlarla kırp
                plate_crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                
                # Varsayılan UI Durumları (Turuncu ve Taranıyor)
                current_color = (0, 165, 255) 
                display_text = "Taraniyor..."
                status_text = "Format Bekleniyor"
                clean_plate = ""
                
                if plate_crop.size > 0:
                    # OCR Modelini Çalıştır
                    char_results = char_recognizer(plate_crop, imgsz=320, conf=0.4, verbose=False)
                    
                    detected_chars = []
                    for c_box in char_results[0].boxes:
                        c_x1 = c_box.xyxy[0][0].item()
                        c_label = char_recognizer.names[int(c_box.cls.item())].upper()
                        detected_chars.append((c_x1, c_label))
                    
                    # Karakterleri x eksenine göre soldan sağa sırala
                    detected_chars.sort(key=lambda x: x[0])
                    raw_text = "".join([c[1] for c in detected_chars]).replace(" ", "")

                    # İl Kodu Düzeltmesi (İlk 2 hanedeki harfleri rakama zorla)
                    for i, char in enumerate(raw_text):
                        if i < 2 and char in correction_map:
                            clean_plate += correction_map[char]
                        else:
                            clean_plate += char

                    # 5. OYLAMA VE DİNAMİK KİLİT ALGORİTMASI
                    total_votes = len(vehicle_votes.get(track_id, []))
                    
                    # KRİTİK KONTROL: Okunan metin boş değilse VE Regex'e tam uyuyorsa oylamaya al
                    if clean_plate and plate_pattern.match(clean_plate):
                        if track_id not in vehicle_votes:
                            vehicle_votes[track_id] = []
                        
                        vehicle_votes[track_id].append(clean_plate)
                        total_votes = len(vehicle_votes[track_id])
                        
                        # Sandıktaki en popüler plakayı bul
                        vote_counts = Counter(vehicle_votes[track_id])
                        most_common_plate, highest_count = vote_counts.most_common(1)[0]

                        # SENARYO A: ARAÇ HENÜZ KİLİTLENMEDİ (1. Oyu arıyoruz)
                        if track_id not in locked_vehicles:
                            display_text = clean_plate 
                            status_text = f"Voting: {highest_count}/{VOTE_THRESHOLD}"
                            current_color = (0, 165, 255) # TURUNCU
                            
                            # Eşik (1) geçildiğinde ve geçerli plaka varken anında kilitle
                            if highest_count >= VOTE_THRESHOLD and most_common_plate:
                                locked_vehicles[track_id] = most_common_plate
                                eval_checkpoints[track_id] = total_votes # Checkpoint al
                                
                                current_color = (0, 255, 0) # YEŞİL
                                display_text = most_common_plate
                                status_text = f"LOCKED [Re: 0/{REEVAL_INTERVAL}]"
                                
                                # TXT Dosyasına Kaydet (İlk Kayıt)
                                time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                logged_plates[track_id] = {"plate": most_common_plate, "time": time_str}
                                update_log_file() # Dosyayı güncel durumla ez
                                
                                print(f"🔒 İLK KİLİT KAYDEDİLDİ: {most_common_plate} ({highest_count} oy)")

                        # SENARYO B: ARAÇ ZATEN KİLİTLİ (Her 5 yeni oyda bir denetle)
                        else:
                            current_color = (0, 255, 0) # Kilitli araç hep YEŞİL kalır
                            display_text = locked_vehicles[track_id]
                            
                            votes_since_last_check = total_votes - eval_checkpoints[track_id]
                            status_text = f"LOCKED [Re: {votes_since_last_check}/{REEVAL_INTERVAL}]"
                            
                            # 5 yeni geçerli oy biriktiyse otomatik düzeltme kontrolü yap
                            if votes_since_last_check >= REEVAL_INTERVAL:
                                eval_checkpoints[track_id] = total_votes # Checkpoint sıfırla
                                
                                # Eğer sandıktan artık daha net ve farklı bir plaka çıkıyorsa güncelle
                                if most_common_plate != locked_vehicles[track_id] and most_common_plate:
                                    old_plate = locked_vehicles[track_id]
                                    locked_vehicles[track_id] = most_common_plate
                                    display_text = most_common_plate
                                    
                                    # TXT Dosyasına Kaydet (Güncelleme - Zaman damgasını koru)
                                    if track_id in logged_plates:
                                        logged_plates[track_id]["plate"] = most_common_plate
                                    else:
                                        time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                                        logged_plates[track_id] = {"plate": most_common_plate, "time": time_str}
                                        
                                    update_log_file() # Dosyayı güncel durumla ez
                                    print(f"🔄 KARAR GÜNCELLENDİ: {old_plate} -> {most_common_plate}")

                    # Eğer okuma Regex'e uymuyorsa ama araç önceden kilitliyse yeşili koru
                    else:
                        if track_id in locked_vehicles:
                            current_color = (0, 255, 0) # YEŞİL
                            display_text = locked_vehicles[track_id]
                            votes_since_last_check = total_votes - eval_checkpoints.get(track_id, 0)
                            status_text = f"LOCKED [Re: {votes_since_last_check}/{REEVAL_INTERVAL}]"
                        else:
                            current_color = (0, 165, 255) # TURUNCU
                            display_text = clean_plate if clean_plate else "Taraniyor..."
                            highest_count = Counter(vehicle_votes[track_id]).most_common(1)[0][1] if total_votes > 0 else 0
                            status_text = f"Voting: {highest_count}/{VOTE_THRESHOLD}"

                # 6. GÖRSELLEŞTİRME VE UI ENTEGRASYONU
                if plate_crop.size > 0:
                    zoomed_view = cv2.resize(plate_crop, (250, 80))
                    frame[10:90, 10:260] = zoomed_view
                    cv2.rectangle(frame, (10, 10), (260, 90), current_color, 2)
                
                cv2.putText(frame, f"READ: {display_text}", (15, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
                cv2.putText(frame, status_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)

        # İşlenmiş video karesini ekrana yansıt
        cv2.imshow("AI Radar - Full ALPR System", frame)
        
        # 'q' tuşuna basılarak güvenli çıkış
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            print("Kullanıcı talebiyle video akışı durduruldu.")
            break

# 7. GÜVENLİ KAPANIŞ 
finally:
    log_session_marker("KAPATILDI")
    cap.release()
    cv2.destroyAllWindows()
    print("--- AI Radar: Oturum başarıyla kapatıldı ve son liste loglandı. ---")