import cv2
import numpy as np
import pyautogui


cap = cv2.VideoCapture(0)
cap.set(3, 320)  
cap.set(4, 240)  


top, right, bottom, left = 50, 250, 150, 400

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    roi = frame[top:bottom, right:left]

   
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    
    _, threshold = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if contours:
        hand_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(hand_contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            #this will adjust volume based on the vertical position of the hand
            volume = np.interp(cy, [top, bottom], [100, 0])
            volume = int(np.clip(volume, 0, 100))
            pyautogui.press('volumedown', presses=volume, interval=0.1)

    
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow("Volume Control", frame)

   #to exit press q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
