import numpy as np
import cv2
def get_frames(path):
  vidcap = cv2.VideoCapture(path)
  frames = []
  while True:
    ret, frame = vidcap.read()
    if not ret:
      break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
  vidcap.release()
  return np.array(frames)
 
VIDEO_KILLED_THE_RADIO_STAR  = "videos/mtv.mp4"
frames = get_frames(VIDEO_KILLED_THE_RADIO_STAR)
print(np.shape(frames))