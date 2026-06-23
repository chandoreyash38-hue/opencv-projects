import cv2
import numpy as np

# Webcam start karein
cap = cv2.VideoCapture(0)

# Drawing ke liye points save karne ki list
points = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame = cv2.flip(frame, 1)
    
    # Image ko BGR se HSV color space me badlein (Color detect karne ke liye best hai)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # --- BLUE COLOR DETECT KARNE KE LIYE RANGE ---
    # Agar aapke paas blue color ka object hai toh ye range perfectly kaam karegi
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Blue color ka mask banayein
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Noise (kachra) saaf karne ke liye morphology use karein
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Object ke kinare (Contours) dhoondhein
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Agar koi blue object dikhta hai
    if len(contours) > 0:
        # Sabse bada contour dhoondhein (taaki background ki chhoti chizen ignore ho sakein)
        c = max(contours, key=cv2.contourArea)
        
        # Us object ka center point (X, Y) nikalein
        M = cv2.moments(c)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            
            cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
            
            
            points.append((center_x, center_y))
            
    for i in range(1, len(points)):
        if points[i - 1] is None or points[i] is None:
            continue
        
        cv2.line(frame, points[i - 1], points[i], (0, 0, 255), 5)
        
    # UI Text
    cv2.putText(frame, "Hold Blue Object to Paint", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'c' to Clear | 'q' to Quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

   
    cv2.imshow("Pure OpenCV Air Painter", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        break
    elif key == ord('c'): 
        points = []

cap.release()
cv2.destroyAllWindows()# opencv-projects
