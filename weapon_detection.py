import numpy as np
import cv2
import imutils

# Load the cascade file for weapon detection
gun_cascade = cv2.CascadeClassifier('cascade.xml')

# Choose between camera or image file
use_camera = False  # Set to True if you want to use the camera instead of an image file

# If using an image file, specify the path
test_image_path = "test_image.jpg"  # Replace with your image path if needed

if use_camera:
    camera = cv2.VideoCapture(0)
else:
    # Read the test image
    frame = cv2.imread(test_image_path)
    if frame is None:
        print(f"Error: Could not load image from {test_image_path}")
        exit(1)
    frame = imutils.resize(frame, width=500)

firstFrame = None
gun_exist = False

while True:
    # Capture frame from camera or use test image
    if use_camera:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read from camera")
            break
        frame = imutils.resize(frame, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect guns in the frame
    guns = gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100, 100))

    if len(guns) > 0:
        gun_exist = True

    # Draw rectangles around detected guns
    for (x, y, w, h) in guns:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Show the frame (works with both camera and image)
    cv2.imshow("Security Feed", frame)
    
    # Exit if 'q' is pressed (only relevant if using camera)
    if use_camera:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        # If testing with an image, break after displaying it once
        cv2.waitKey(0)
        break

# Print result after exiting the loop
if gun_exist:
    print("Weapon Detected")
else:
    print("No Weapon Detected")

# Release resources
if use_camera:
    camera.release()
cv2.destroyAllWindows()




'''
#Use this code if you want to use webcam to detect weapon.

import numpy as np
import cv2
import imutils
import datetime

gun_cascade = cv2.CascadeClassifier('cascade.xml')
#Camera On
camera = cv2.VideoCapture(0)


firstFrame = None
gun_exist = None

while True:
    #Read Image on Camera
    ret, frame = camera.read()
    
    frame = imutils.resize(frame, width=500)

    #convert RGB to gray image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    gun=gun_cascade.detectMultiScale(gray, 1.3, 5, minSize=(100,100))


    if len(gun)>0:
        gun_exist = True

    
    for (x,y,w,h) in gun:
        frame = cv2.rectangle( frame, (x,y), (x+w, y+h), (255,0,0),2 )

        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
    
    if firstFrame is None:
        firstFrame = gray
        continue

    cv2.imshow("Security Feed", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

if gun_exist:
    print("Weapon Detected")
else:
    print("No Weapon Detected")

camera.release()
cv2.destroyAllWindows()

'''



