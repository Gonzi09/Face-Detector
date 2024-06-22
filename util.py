import cv2

def objectDetect(frame, classifier):
  
  grayScale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  objects = classifier.detectMultiScale(grayScale, scaleFactor= 1.1, minNeighbors=5, minSize=(100, 100))
  return objects

def drawing(frame, objects) -> None:
  for x,y,w,h in objects:
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 25)


