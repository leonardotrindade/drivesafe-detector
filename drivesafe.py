from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import argparse
import dlib
import cv2
import requests

def curr_millis(): return int(round(time.time() * 1000))

def euclidean_dist(ptA, ptB): return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])
    C = euclidean_dist(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 28

COUNTER = 0
ALARM_ON = False
ALARM_INTERVAL = 5000
LAST_SENT_TIME = 0

ap = argparse.ArgumentParser()
ap.add_argument("-u", "--url", required=True,
                help = "api endpoint")
ap.add_argument("-t", "--token", required=True,
                help="user token")
args = vars(ap.parse_args())

print("[DriveSafe] loading facial landmark predictor...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[DriveSafe] starting video stream thread...")
# vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=650)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),
                              int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                cv2.putText(frame, "CUIDADO! VOCE ESTA SONOLENTO!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if ALARM_ON and (curr_millis() - LAST_SENT_TIME) > ALARM_INTERVAL:
                    url = "https://ipinfo.io/loc"
                    r = requests.get(url)
                    lat, long = r.text.rstrip().split(",")
                    payload = {'lat': lat, 'long': long}
                    headers = { 'Authorization': 'Bearer ' + args["token"]}
                    r = requests.post(args["url"], data=payload, headers=headers)
                    print("[DriveSafe] notification sent!")
                    LAST_SENT_TIME = curr_millis()
        else:
            COUNTER = 0
            ALARM_ON = False

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (530, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("DriveSafe", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
