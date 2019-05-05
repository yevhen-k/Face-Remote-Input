import os
import dlib
import numpy as np
import cv2

from face_processing import *



def landmark_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)

    # loop over all facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def main():
        
    cap = cv2.VideoCapture(0)

    path_to_cascade = os.path.join(os.getcwd(), 'models/haar-cascade/haarcascade_frontalface_default.xml')
    face_cascade_detector = cv2.CascadeClassifier(path_to_cascade)

    path_to_landmark = os.path.join(os.getcwd(), 'models/face-landmark-68/shape_predictor_68_face_landmarks.dat')
    face_landmark_predictor = dlib.shape_predictor(path_to_landmark)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Get all faces on gray frame
        faces = face_cascade_detector.detectMultiScale(gray, 1.3, 5)
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
            cv2.putText(frame, 'mouth ratio = {}'.format(mr), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(frame, 'mouth opened = {}'.format(mo), (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            for idx, (x, y) in enumerate(landmarks_np):
                if idx in LEFT_EYE or idx in RIGHT_EYE or idx in NOSE_BRIDGE or idx in LIPS:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
        