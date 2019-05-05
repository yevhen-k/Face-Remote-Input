RIGHT_EYE = list(i for i in range(36, 42))
LEFT_EYE = list(i for i in range(42, 48))
NOSE_BRIDGE = list(i for i in range(27, 31))
LIPS = list(i for i in range(60, 68))


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
