import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


def findEncoding(images):
    encodeList = []  # will contain all the encoded lists
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # converting the BGR Image into RBG
        encode = face_recognition.face_encodings(img)[0]  # finding the encoding of the face and trues the 128 values as a list
        encodeList.append(encode)

    return encodeList


def markAttendance(name):
    with open("Attendance.csv", 'r+') as f:
        data = f.readlines()
        nameList = []

        for line in data:
            entry = line.split(',')
            nameList.append(entry[0])  # to get the names which are already in the file

        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')  # to format the current time
            f.writelines(f"\n{name},{timeString}")


path = 'Images'
images = []
imageNames = []
directory = os.listdir(path)  # making a list of all the files present in the path

for img in directory:
    currImage = cv2.imread(f'{path}/{img}')  # opening the image
    images.append(currImage)  # appending the cv2 format to a list
    imageNames.append(os.path.splitext(img)[0])  # appending the name of the file without .jpg to the list

print("Available Database:-")
for i in range(len(imageNames)):
    print(f"{i+1}.) {imageNames[i]}")

encodeListKnown = findEncoding(images)  # list with the encoded data of all the images
print("\nEncoding Done!")

# the test image will be taken from the Web Camera of the system
cap = cv2.VideoCapture(0)  # initializing the webcam
while True:
    success, img = cap.read()  # read the current image
    imgR = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # resizing the image to make the process faster
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)  # converting the BGR Image into RBG

    # in our webcam we might find multiple faces, for that we will find the location of these faces and send them for encoding
    currFrameFaces = face_recognition.face_locations(imgR)
    currFrameEncodes = face_recognition.face_encodings(imgR, currFrameFaces)  # finding the encoding of the face and return the 128 values as a list

    for faceLoc, encodeFace in zip(currFrameFaces, currFrameEncodes):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # it will return the list of bools from the current frame and given images
        faceDistance = face_recognition.face_distance(encodeListKnown, encodeFace)  # it will return the list of distances of the current frame from all the given images
        matchIndex = np.argmin(faceDistance)  # index of the smallest face distance i.e. the closest match to the face in current frame

        # drawing the box and the name
        if matches[matchIndex]:  # checking is the face is found or not
            name = imageNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc  # taking the coordinates from the Face Location
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # multiplying it by 4 because we scaled down the image by 1/4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255))
            markAttendance(name)

    cv2.imshow("Webcam", img)
    key = cv2.waitKey(1)
    if key == ord("q"):  # if q key is pressed the program will exit
        break
