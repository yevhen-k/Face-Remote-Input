import os
import dlib
import cv2

from face_processing import *


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
        display_faces(faces, frame, gray, face_landmark_predictor)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
        