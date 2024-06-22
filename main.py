import cv2
from util import objectDetect, drawing


# Variables 

webcam = cv2.VideoCapture(1)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
bodyCascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
toggle_face = False
toggle_body = False

# person = cv2.CascadeClassifier('haarcascade_upperbody.xml')

# Functions

while True:

  _, frame = webcam.read()

  if frame is None:
    print('No video')
    break
  

  if toggle_face:
    faces = objectDetect(frame, faceCascade)
    drawing(frame, faces)

  if toggle_body:
    bodies = objectDetect(frame, bodyCascade)
    drawing(frame, bodies)

  cv2.imshow('Classifier', frame)


  key = cv2.waitKey(1) & 0xFF
  
  if key == ord('q'):
    print('End')
    break

  elif key == ord('f'):
    toggle_face = not toggle_face


  elif key == ord('b'):
    toggle_body = not toggle_body



  print('Face detection: {} Body detection: {}'.format(toggle_face, toggle_body))
webcam.release()
cv2.destroyAllWindows()