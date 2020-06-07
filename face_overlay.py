"""
face overlay
"""

import cv2
import sys

casc_path = cv2.data.haarcascades+"haarcascade_frontalface_alt.xml"
face_cascade = cv2.CascadeClassifier(casc_path)

device = 1 # Front camera
try:
    device = int(sys.argv[1]) # 0 for back camera
except IndexError:
    pass

cap = cv2.VideoCapture(device)

def face_overlay(image):
    # image padding
    padding_size = int(image.shape[1] / 2)
    padding_img = cv2.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = cv2.copyMakeBorder(image, padding_size, padding_size , padding_size, padding_size, cv2.BORDER_CONSTANT, value=(0,0,0))
    image_tmp = image_tmp.astype('float64')

    # face detect
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    facerect = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

    # face overlay
    if len(facerect) > 0:
        for rect in facerect:
            face_size = rect[2] * 2
            face_pos_adjust = int(rect[2] * 0.5)
            face_img = cv2.imread('./karaage_icon.png', cv2.IMREAD_UNCHANGED)
            face_img = cv2.resize(face_img, (face_size, face_size))
            mask = face_img[:,:,3]
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = mask / 255.0
            face_img = face_img[:,:,:3]

            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] *= 1 - mask
            image_tmp[rect[1]+padding_size-face_pos_adjust:rect[1]+face_size+padding_size-face_pos_adjust,
                      rect[0]+padding_size-face_pos_adjust:rect[0]+face_size+padding_size-face_pos_adjust] += face_img * mask

    image_tmp = image_tmp[padding_size:padding_size+image.shape[0], padding_size:padding_size+image.shape[1]]
    image_tmp = image_tmp.astype('uint8')

    return image_tmp


while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is not empty
    if not ret:
        continue

    # Auto rotate camera
    try:
        frame = cv2.autorotate(frame, device)
    except:
        pass

    # Face overlay
    frame = face_overlay(frame)

    # Convert from BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    try:
        if cv2.waitKey(1) == 27:
            break
    except:
        pass