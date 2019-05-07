import numpy as np
import dlib
import cv2
import pyautogui


RIGHT_EYE = list(i for i in range(36, 42))
LEFT_EYE = list(i for i in range(42, 48))
NOSE_BRIDGE = list(i for i in range(27, 31))
LIPS = list(i for i in range(60, 68))

def landmark_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def is_mouth_opened(landmarks_np):
    mr = mouth_ratio(landmarks_np)
    if mr < 0.20:
        return False
    return True


def mouth_ratio(landmarks_np):
    mouth_width = landmarks_np[64, 0] - landmarks_np[60, 0]
    mouth_height = landmarks_np[66, 1] - landmarks_np[62, 1]
    mr = mouth_height / mouth_width
    if mr < 0.05:
        return 0
    return round(mr, 2)


def nose_tip(landmarks_np):
    return landmarks_np[30, :]


def nose_bridge(landmarks_np):
    return landmarks_np[27, :]


def left_eye_outer(landmarks_np):
    return landmarks_np[36, :]


def left_eye_inner(landmarks_np):
    return landmarks_np[39, :]


def right_eye_outer(landmarks_np):
    return landmarks_np[45, :]


def right_eye_inner(landmarks_np):
    return landmarks_np[42, :]


def face_start(landmarks_np):
    return landmarks_np[0, :]


def face_end(landmarks_np):
    return landmarks_np[16, :]


def up_down_direction(landmarks_np):
    '''Value 0 means defaul position of face.
    Value > 0 means pointing down.
    Value < 0 means pointing up.
    '''
    upper_vartical_level = int((face_start(landmarks_np)[1] + face_end(landmarks_np)[1]) / 2)
    lower_vartical_level = int((landmarks_np[4,1] + landmarks_np[12,1]) / 2)
    nose_tip_pos = nose_tip(landmarks_np)[1]
    upper_vl_to_nose_distance = upper_vartical_level - nose_tip_pos
    lower_vl_to_nose_distance = nose_tip_pos - lower_vartical_level
    res = round((upper_vl_to_nose_distance / lower_vl_to_nose_distance * 10) / 6 - 1.3, 1)
    if -0.5 < res < 0.5:
        return 0
    else:
        return res


def left_right_direction(landmarks_np):
    '''Value x == 0 means face is centered
    value < 0 means face is turned right
    value > 0 means face is turned left
    '''
    nose_tip_pos = nose_tip(landmarks_np)[0]
    nose_to_fase_start_distance = nose_tip_pos - face_start(landmarks_np)[0]
    nose_to_fase_end_distance = face_end(landmarks_np)[0] - nose_tip_pos
    res = round((nose_to_fase_start_distance / nose_to_fase_end_distance * 10) / 20 - 0.5, 1)
    if -0.2 < res < 0.2:
        return 0
    else:
        return res


def display_faces(faces, frame, gray, face_landmark_predictor):
    if not len(faces):
        return
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
        left = x
        top = y
        right = x + w 
        bottom = y + h
        dlib_rectangle = dlib.rectangle(left, top, right, bottom)
        landmarks = face_landmark_predictor(gray, dlib_rectangle)
        landmarks_np = landmark_to_np(landmarks)

        mr = mouth_ratio(landmarks_np)
        mo = is_mouth_opened(landmarks_np)
        up_down = up_down_direction(landmarks_np)
        left_right = left_right_direction(landmarks_np)

        pyautogui.moveRel(-50 * left_right, 10 * up_down, duration=0.01)
        if mo:
            pyautogui.click()

        cv2.putText(frame, 'mouth ratio = {}'.format(mr), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, 'mouth opened = {}'.format(mo), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, 'up/down direction = {}'.format(up_down), (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
        cv2.putText(frame, 'left/right direction = {}'.format(left_right), (10,160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

        for idx, (x, y) in enumerate(landmarks_np):
            if idx in [36,39,27,30,42,45]:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            else:
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

