
def streamimg(video_path, queue ): 
    video_data = open(video_path, 'r')
    data_loader = video_data.readlines()
    for video in data_loader:
        video_load = video.split(' ')
        cap = cv2.VideoCapture(video_load[0])
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                queue.put(frame)

