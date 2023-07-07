import cv2
import face_recognition

# Step 1 - Loading the BGR image and converting it into RGB

imgElon = face_recognition.load_image_file("Basic Images/Elon Musk.jpg")  # loading the image, done in BGR
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)  # converting the BGR Image into RBG

imgElonTest = face_recognition.load_image_file("Basic Images/Elon Musk Test.jpg")
imgElonTest = cv2.cvtColor(imgElonTest, cv2.COLOR_BGR2RGB)

# Step 2 - Finding Faces in our images and their encodings

# face_location function returns 4 values which are x1, y1, x2, y2
faceLoc = face_recognition.face_locations(imgElon)[0]  # first element because we are only sending one image
encodeElon = face_recognition.face_encodings(imgElon)[0]  # this does the encoding of the face and returns the 128 values as a list
cv2.rectangle(imgElon, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)  # making a rectangle on the face found using the parameters of faceLoc

faceLocTest = face_recognition.face_locations(imgElonTest)[0]
encodeElonTest = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (0, 255, 0), 2)

# Step 3 - Comparing both the faces and finding the distance between them using Linear SVM

results = face_recognition.compare_faces([encodeElon], encodeElonTest)  # returns true if the encodings are similar
# for higher accuracy we use the Face Distance as a parameter too
faceDistance = face_recognition.face_distance([encodeElon], encodeElonTest)  # the lower the value to more similar # they are

cv2.putText(imgElonTest, f"{results[0]} {round(faceDistance[0], 2)}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)  # used to put text over the image
cv2.imshow("Elon Musk", imgElon)  # used to display the image
cv2.imshow("Elon Musk Test", imgElonTest)
cv2.waitKey(0)
