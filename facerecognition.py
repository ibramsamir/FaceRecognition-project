# Import necessary libraries
import cv2
import numpy as np
import face_recognition
import os

# Set the path where the images of persons are stored
path = 'persons'
images = []
classNames = []
personsList = os.listdir(path)

# Loop through the images in the specified path
for cl in personsList:
    # Read each image and append it to the 'images' list
    curPersonn = cv2.imread(f'{path}/{cl}')
    images.append(curPersonn)
    
    # Extract the class name (person's name) from the file name and append to 'classNames' list
    classNames.append(os.path.splitext(cl)[0])

# Display the list of class names (persons)
print(classNames)

# Function to find face encodings for the images in the 'images' list
def findEncodeings(image):
    encodeList = []
    for img in images:
        # Convert the image to RGB format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find and append face encodings to the 'encodeList'
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Call the function to get the face encodings for known images
encodeListKnown = findEncodeings(images)
print(encodeListKnown)
print('Encoding Complete.')

# Open the default camera (camera index 0)
cap = cv2.VideoCapture(0)

# Main loop for video capture and face recognition
while True:
    # Read a frame from the camera
    _, img = cap.read()

    # Resize the frame to speed up face recognition
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    faceCurentFrame = face_recognition.face_locations(imgS)
    encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)

    # Loop through each face in the current frame
    for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
        # Compare face encodings with known face encodings
        matches = face_recognition.compare_faces(encodeListKnown, encodeface)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
        matchIndex = np.argmin(faceDis)

        # If a match is found, display the person's name on the frame
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,0,255), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

    # Display the frame with face recognition overlay
    cv2.imshow('Face Recognition', img)
    cv2.waitKey(1)
