import numpy as np


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
    '''Value 3 means defaul position of face.
    Value > 3 means pointing down.
    Value < 3 means pointing up.
    Delimeter 10 means noise suppression
    '''
    vartical_level = int((face_start(landmarks_np)[1] + face_end(landmarks_np)[1]) / 2)
    nose_tip_pos = nose_tip(landmarks_np)[1]
    return int((nose_tip_pos - vartical_level) / 10)


def left_right_direction(landmarks_np):
    '''Value -2 <= x <= 2 means face is centered
    value < -2 means face is turned right
    value > 2 means face is turned left
    '''
    nose_tip_pos = nose_tip(landmarks_np)[0]
    nose_to_fase_start_distance = nose_tip_pos - face_start(landmarks_np)[0]
    nose_to_fase_end_distance = face_end(landmarks_np)[0] - nose_tip_pos
    return int((nose_to_fase_start_distance - nose_to_fase_end_distance) / 10)