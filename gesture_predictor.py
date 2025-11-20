import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMAGE_SIZE = 100
CLASSES = ["Fist", "Palm", "Swing"]

model = load_model("gesture_model.h5")

bg = None

def run_avg(image, aWeight):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
        return
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) == 0:
        return None
    return thresholded, max(cnts, key=cv2.contourArea)

def preprocess(img):
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype("float32") / 255.0
    img = np.reshape(img, (1, IMAGE_SIZE, IMAGE_SIZE, 1))
    return img

camera = cv2.VideoCapture(0)
top, right, bottom, left = 10, 350, 225, 590

num_frames = 0
start_prediction = False

while True:
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)
    clone = frame.copy()

    roi = frame[top:bottom, right:left]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    if num_frames < 30:
        run_avg(gray, 0.5)
    else:
        hand = segment(gray)
        if hand is not None:
            thresholded, segmented = hand

            if start_prediction:
                processed = preprocess(thresholded)
                prediction = model.predict(processed)[0]
                class_id = np.argmax(prediction)
                confidence = prediction[class_id]

                cv2.putText(clone, f"{CLASSES[class_id]} ({confidence:.2f})",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0,255,0), 2)

            cv2.imshow("Thresholded", thresholded)

    cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    cv2.imshow("Video Feed", clone)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    if key == ord("s"):
        start_prediction = True

    num_frames += 1

camera.release()
cv2.destroyAllWindows()
