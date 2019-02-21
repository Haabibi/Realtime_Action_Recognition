import cv2
import numpy as np
import os 


vid_list = sorted(os.listdir('./demo_video'))
sorted_vid_list = [ str(x+1)+'.h264' for x in range(len(vid_list))]

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (800,400))

for vid in sorted_vid_list:
    cap  = cv2.VideoCapture(os.getcwd() + '/demo_video/' + vid)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            print(vid)
            cv2.putText(frame, vid, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(56, 32, 32), 2)
            out.write(frame)
            
        else:
            break

cap.release()
out.release()
cv2.destroyAllWindows()
