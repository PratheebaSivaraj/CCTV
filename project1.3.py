import pandas,cv2,time
import imutils
import warnings
import datetime


first_frame=None
status_list=[None,None]
times=[]
lastUploaded = datetime.datetime.now()
df=pandas.DataFrame(columns=["Start","End"])
time.sleep(2.5)
video=cv2.VideoCapture(0)
motionCounter = 0
while True:
    check,frame=video.read()#it is a Numpay array it represents the first image the vedio captures
    status=0
    timestamp = datetime.datetime.now()
    frame = imutils.resize(frame, width=1080)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray, (21,21),0)
    if first_frame is None:
        first_frame=gray.copy().astype("float")
        continue
    
    # accumulate the weighted average between the current frame and
	# previous frames, then compute the difference between the current
	# frame and running average
    cv2.accumulateWeighted(gray, first_frame, 0.5)
    delta_frame = cv2.absdiff(gray, cv2.convertScaleAbs(first_frame))
    
    #delta_frame=cv2.absdiff(first_frame,gray)
    thresh_delta=cv2.threshold(delta_frame,5,255,cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_delta,None,iterations=2)
    (cnts,image)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   
    for contour in cnts:
        if cv2.contourArea(contour)<1000:
            continue
        status=1
        (x,y,w,h)=cv2.boundingRect(contour)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
    cv2.imshow('Capturing img',frame)
    cv2.imshow('Capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thresh_delta',thresh_delta)
    
    
    
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    if status == 1:
        # check to see if enough time has passed between uploads
        if (timestamp - lastUploaded).seconds >= 3.0:
			# increment the motion counter
            motionCounter += 1
            #if motionCounter==1:
                #times.append(datetime.datetime.now())
            # check to see if the number of frames with consistent motion is
            # high enough
            if motionCounter >= 20:
                path = timestamp.strftime("%b-%d_%H_%M_%S" + ".jpg")
                cv2.imwrite(path, frame)
                times.append(datetime.datetime.now())
                lastUploaded = timestamp
                motionCounter = 0
            # otherwise, the room is not occupied
    else:
        motionCounter = 0
        
    
    key=cv2.waitKey(1)& 0xFF# this will create a new frame after 1 millissecond
    if key==ord('q'):
        break
    """
    status_list.append(status)
    status_list=status_list[-2:]
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.datetime.now())
    """
print(status_list)
print(times)
for i in range(0,len(times),2):
    try:
        df=df.append({'Start':times[i],"End":times[i+1]},ignore_index=True)
    except:
        print("one miss")
df.to_csv("Times.csv")
video.release()
cv2.destroyAllWindows()