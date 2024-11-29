import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array
from PIL import ImageFont, ImageDraw, Image

# Tải mô hình phân loại khuôn mặt và phát hiện cảm xúc
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
classifier = load_model('emotion.keras')
class_labels = ['Giận dữ', 'Ghê sợ', 'Sợ hãi', 'Hạnh phúc', 'Bình thường', 'Buồn', 'Bất ngờ', 'Buồn ngủ']
#class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Tải font chữ để vẽ văn bản
font = ImageFont.truetype("arial.ttf", 32)
b, g, r, a = 0, 255, 0, 0

# Bắt đầu quay video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.32, 5)

    for (x, y, w, h) in faces:
        # Vẽ hình chữ nhật quanh khuôn mặt
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Xử lý trước vùng khuôn mặt để phát hiện cảm xúc
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (224, 224), interpolation=cv2.INTER_AREA)  # Resize về 224x224
        roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2RGB)  # Chuyển đổi thang độ xám sang RGB

        # Chuẩn hóa và chuẩn bị cho đầu vào mô hình
        roi = roi_color.astype('float32') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)  # Add batch dimension

        # Dự đoán khuôn mặt
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]

        # Chuyển đổi khung sang định dạng PIL để vẽ văn bản
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y - 10), label, font=font, fill=(b, g, r, a))
        frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # Hiển thị khung có nhãn cảm xúc
    cv2.imshow('Emotion Detection', frame)

    # Sử dụng q trên bàn phím để tắt chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
