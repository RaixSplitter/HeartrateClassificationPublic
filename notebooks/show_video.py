# load video path
import cv2


video_path = "C:/Users/marku/Downloads/image_to_video.avi"

# load video
cap = cv2.VideoCapture(video_path)

# display video

print(cap.isOpened())
while cap.isOpened():
    ret, frame = cap.read()
    print(ret, frame)
    if not ret:
        break
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

