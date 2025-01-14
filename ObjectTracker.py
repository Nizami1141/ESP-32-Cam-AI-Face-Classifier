import cv2
import os
from datetime import datetime

# Create output directory
output_dir = "annotated_frames"
os.makedirs(output_dir, exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Create tracker
tracker = cv2.TrackerCSRT_create()

# Read first frame
ret, frame = cap.read()

# Select ROI
bbox = cv2.selectROI("Tracking", frame, False)
tracker.init(frame, bbox)

def drawBox(img, bbox):
    x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 3, 1)
    cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save annotated frame with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{output_dir}/frame_{timestamp}.jpg"
    cv2.imwrite(filename, img)
    
    # Save bounding box coordinates to text file
    with open(f"{output_dir}/annotations.txt", "a") as f:
        f.write(f"{filename} {x} {y} {w} {h}\n")

while True:
    timer = cv2.getTickCount()
    ret, frame = cap.read()
    
    # Update tracker
    success, bbox = tracker.update(frame)
    
    if success:
        drawBox(frame, bbox)
    else:
        cv2.putText(frame, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display FPS
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display frame
    cv2.imshow("Tracking", frame)
    
    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
