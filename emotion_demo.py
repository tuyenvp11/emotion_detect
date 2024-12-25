import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt

# load model
model = load_model("emt_det.keras")

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load font tiếng Việt
# Tải font chữ để vẽ văn bản
font = ImageFont.truetype("arial.ttf", 32)
b, g, r, a = 0, 255, 0, 0

cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]  # cropping region of interest i.e. face area from image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions)

        # Emotions in Vietnamese
        emotions = ('Giận dữ', 'Ghê sợ', 'Sợ hãi', 'Hạnh phúc', 'Buồn', 'Trung lập', 'Bất ngờ', 'Buồn ngủ')
        predicted_emotion = emotions[max_index]

        # Convert OpenCV image to PIL format
        pil_img = Image.fromarray(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y - 30), predicted_emotion, font=font, fill=(0, 0, 255, 0))

        # Convert PIL image back to OpenCV format
        test_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Phân tích cảm xúc khuôn mặt', resized_img)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
