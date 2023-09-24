import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import keras.preprocessing.image as kImage

# Constants
MODEL_INDEX = 0
MODEL_TYPE_INDEX = 1
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']
CAMERA_INDEX = 0
ESC_KEY = 27
SPACE_KEY = 32

MODEL_NAME = ["ResNet50", "MobileNet", "InceptionV3"][MODEL_INDEX]
MODEL_TYPE = ["libras", "personal"][MODEL_TYPE_INDEX]

# Functions


def getModelPath():
    if (MODEL_TYPE == "libras"):
        if (MODEL_NAME == "ResNet50"):
            return "../training/ResNet50/libras/resnet50_model.h5"
        elif (MODEL_NAME == "MobileNet"):
            return "../training/MobileNet/libras/mobilenet_model.h5"
        else:
            return "../training/InceptionV3/libras/inception_v3_model.h5"
    else:
        if (MODEL_NAME == "ResNet50"):
            return "../training/ResNet50/personal/resnet50_model.h5"
        elif (MODEL_NAME == "MobileNet"):
            return "../training/MobileNet/personal/mobilenet_model.h5"
        else:
            return "../training/InceptionV3/personal/inception_v3_model.h5"


def getFrame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    return frame


def getKeyPressed():
    k = cv2.waitKey(1)
    return k % 256


def getHandLandmarks(frame):
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(framergb)
    return result.multi_hand_landmarks


def drawHandLandmarks(frame):
    hand_landmarks = getHandLandmarks(frame)
    h, w, _ = frame.shape
    if hand_landmarks:
        for handLMs in hand_landmarks:
            x_max = 0
            y_max = 0
            x_min = w
            y_min = h
            for lm in handLMs.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            y_min -= 50
            y_max += 50
            x_min -= 50
            x_max += 50
            cv2.rectangle(frame, (x_min, y_min),
                          (x_max, y_max), (0, 255, 0), 2)
            return x_min, x_max, y_min, y_max


def saveImage(frame, input_size):
    imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_name = "./temp/img.png"
    save_img = cv2.resize(imgrgb, input_size)
    cv2.imwrite(img_name, save_img)
    return img_name


def modelPredict(frame):
    if (MODEL_NAME == "InceptionV3"):
        input_size = (299, 299)
        crop_size = (75, 75)
    elif (MODEL_NAME == "ResNet50"):
        input_size = (224, 224)
        crop_size = (64, 64)
    else:
        input_size = (224, 224)
        crop_size = (64, 64)
    img_name = saveImage(frame, input_size)
    img = kImage.load_img(img_name, target_size=input_size)
    img_array = kImage.img_to_array(img)
    preprocessed_img = preprocessFrame(img_array)
    preprocessed_img = cv2.resize(preprocessed_img, crop_size)
    predictions = model.predict(np.expand_dims(
        preprocessed_img, axis=0), verbose=0)
    maior, class_index = -1, -1
    for x in range(len(CLASSES)):
        if predictions[0][x] > maior:
            maior = predictions[0][x]
            class_index = x
    return [predictions, CLASSES[class_index]]


def preprocessFrame(image):
    if (MODEL_NAME == "InceptionV3"):
        frame = keras.applications.inception_v3.preprocess_input(image)
        return frame
    elif (MODEL_NAME == "ResNet50"):
        frame = keras.applications.resnet50.preprocess_input(image)
        return frame
    else:
        frame = keras.applications.mobilenet.preprocess_input(image)
        return frame


# Code
model = keras.models.load_model(getModelPath())
cap = cv2.VideoCapture(CAMERA_INDEX)

mphands = mp.solutions.hands
hands = mphands.Hands()

if __name__ == "__main__":
    while True:
        frame = getFrame()
        key = getKeyPressed()
        if (key == ESC_KEY):
            print("Fechando programa...")
            break
        hand_rec_sizes = drawHandLandmarks(frame)
        cv2.imshow("Camera", frame)
        has_hand = getHandLandmarks(frame)
        if (has_hand):
            x_min, x_max, y_min, y_max = hand_rec_sizes
            if (sum(n < 0 for n in [x_min, x_max, y_min, y_max])):
                continue
            hand_frame = frame[y_min:y_max, x_min:x_max]
            cv2.imshow("ROI", hand_frame)
            result = modelPredict(hand_frame)
            print(result[1])
        else:
            print("Não foi encontrada uma mão no frame")
    cap.release()
    cv2.destroyAllWindows()
