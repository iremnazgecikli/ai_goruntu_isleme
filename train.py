from ultralytics import YOLO

# 1. Hazır YOLOv8 hafif modelini yüklüyoruz
model = YOLO('yolov8n.pt')

# 2. Modeli eğitmeye başlıyoruz
# (Roboflow'dan indirdiğin veya hazırladığın klasörün içindeki 'data.yaml' dosyasının tam yolunu tırnak içine yazacaksın)
results = model.train(
    data='data.yaml', 
    epochs=10,  
    imgsz=640
)

print("Eğitim başarıyla tamamlandı!")