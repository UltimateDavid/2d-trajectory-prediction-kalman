# USAGE
# python main.py

# Import the necessary packages
from webcamvideostream import WebcamVideoStream
from kalman import kalman
from fps import FPS
import numpy.linalg as la
import argparse
import imutils
import cv2 as cv
import numpy as np
import datetime


font = cv.FONT_HERSHEY_SIMPLEX

# Minimum area of an object to be recognized (to avoid noise)
min_area = 20

# Range of green color in HSV
lower = np.array([50,80,50])
upper = np.array([100,255,255])
lower = np.array([40,100,30])
upper = np.array([100,255,255])

#video_file = 'videos/example_05.avi'
video_file = 'videos/example_05.avi'
pause = False 

screen_ratio = 3/4
frame_w = 600
frame_h = int(frame_w * screen_ratio)

################### Kalman initialisation ###################
# Starting Values of the Kalman Filter

a = np.array([0, 900])

# State Matrix: x,y,vx,vy (positions and velocities)
mu = np.array([0,0,0,0])
# Covariance matrix: uncertainty
P = np.diag([1000,1000,1000,1000])**2

speed = 0.4
fps = 20 # Dit werkt nog niet helemaal, ik weet niet waarom het niet gelijkloopt
dt = 1/(fps/speed) # every timestep (delta t) 
noise = 3

sigmaM = 0.01 # model noise 
sigmaZ = 3*noise # average noise of the imaging process

Q = sigmaM**2 * np.eye(4) # Noise of F
R = sigmaZ**2 * np.eye(2) # Noise of H

F = np.array( 
		[1, 0, dt, 0,
		 0, 1, 0, dt,
		 0, 0, 1, 0,
		 0, 0, 0, 1 ]).reshape(4,4)

B = np.array( 
		[dt**2/2, 0, 
		 0, dt**2/2, 
		 dt, 0, 
		 0, dt]).reshape(4,2)

H = np.array(
	[1,0,0,0,
	 0,1,0,0]).reshape(2,4)

#res = [(mu,P)] # will be all the matrices in the Kalman Filter
res=[] 

# Storage of measurements
listCenterX=[]
listCenterY=[]
listPoints=[]

xe = None
ye = None

lastframe = None



################### START VIDEOSTREAM ###################
print("[INFO] sampling frames from webcam...")

# Check if we have a video or a webcam
if video_file is not None:
	stream = cv.VideoCapture(video_file)
	fps = stream.get(cv.CAP_PROP_FPS)
# otherwise, we are reading from a webcam
else:
	stream = WebcamVideoStream(src=0).start()
# measurements
streamfps = FPS().start()

out = cv.VideoWriter('outframe.avi',cv.VideoWriter_fourcc('M','J','P','G'), 15, (frame_w, frame_h))

################### START CAPTURING ###################
# loop over every frame
while(True):
	key = cv.waitKey(40) & 0xFF
	#if key== ord("c"): crop = True # Crop only to the region of interest
	if key == ord("p"): P = np.diag([100,100,100,100])**2 # Make the filter uncertain again
	if key == ord("q") or key == 27: break # quitting when ESCAPE or q is pressed
	if key == ord(" "): pause =not pause # pause when spacebar is pressed, unpause when pressed again
	if(pause): continue



	# grab the frame from the stream
	(grabbed, frame_read) = stream.read()
	# Check if it has been grabbed
	if grabbed is False:
		continue
	# Check if the frame is not None or empty
	#if not(isinstance(frame_read, np.ndarray)):
	#	break
	# Check if two frames are the same
	#if frame_read is lastframe:
	#	break
	frame = frame_read
	frame = imutils.resize(frame, width=frame_w, height=frame_h)



	########## FIND GREEN PIXELS ##########
	# Convert BGR to HSV
	hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
	
    # Threshold the HSV image to get only green colors
	mask = cv.inRange(hsv, lower, upper)




	########## FIND OBJECTS ##########
	# Loop over the contours to find all objects
	contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	contours = imutils.grab_contours(contours)
	#print(contours)

	# Find biggest contour
	num = 0
	biggreen = None
	max_area = 0
	for c in contours:
		area = cv.contourArea(c)
		if area==0:
			continue
		if area > max_area:
			max_area = area
			biggreen = c
			pos = num
		num += 1



	############## PROCESSING OBJECT ##############
	if (	num > 0 # Check if there is any green object 
			and int(cv.contourArea(biggreen)) > min_area # no noise
			and streamfps.frames() > 50): # Start the predicting at a particular point


		######## CALCULATE POSITION ########
		# Calculate bounding box
		#print(streamfps.frames())
		(x, y, w, h) = cv.boundingRect(biggreen)

		# Calculate x and y of the current observation
		xo = x + int(w/2)
		yo = y + int(h/2)
		error = w # Error: somewhere in this region is the center 




		######## KALMAN FILTER Position #########
		if yo < error: 
			mu,P,pred = kalman(mu,P,F,Q,B,a,None,H,R) 
			m="None" 
			mm=False
		else: 
			mu,P,pred = kalman(mu,P,F,Q,B,a,np.array([xo,yo]),H,R) 
			m="normal"
			mm=True
		
		if(mm): 
			listCenterX.append(xo) 
			listCenterY.append(yo)
		listPoints.append((xo,yo))
		res += [(mu,P)]

		# Calculating estimated position and uncertainty based on the state/covariance matrices
		xe = [mu[0] for mu,_ in res] 
		ye = [mu[1] for mu,_ in res] 
		xu = [2*np.sqrt(P[0,0]) for _,P in res] # uncertainty of estimated position
		yu = [2*np.sqrt(P[1,1]) for _,P in res] # uncertainty of estimated position




		######## KALMAN FILTER Prediction #########
		mu2 = mu 
		P2 = P 
		res2 = []
		
		for _ in range(int(fps*3)): 
			mu2,P2,pred2 = kalman(mu2,P2,F,Q,B,a,None,H,R) 
			res2 += [(mu2,P2)]

		xp = [mu2[0] for mu2,_ in res2] 
		yp = [mu2[1] for mu2,_ in res2] 
		xpu= [2*np.sqrt(P[0,0]) for _,P in res2] 
		ypu= [2*np.sqrt(P[1,1]) for _,P in res2]



	######## DRAW ########
	if xe is not None: #If there is any measurement
		cv.rectangle(frame, (x,  y), (x + w, y + h), (0, 255, 0), 2)
		cv.circle(frame, center=(xo, yo), radius=int((w+h)/24), color=(0, 0, 255), thickness=-1, lineType=8, shift=0)
		text = "Coordinates:" + str(xo) + ", " + str(yo)
		cv.putText(frame, format(text), (10, 20), font, 0.5, (0, 0, 255), 2)
		
		# Draw every measurement so far
		for n in range(len(listCenterX)): 
			cv.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3, (0, 255, 0),-1)
		# Draw the corrected measurements
		for n in range(len(xe)):
			uncertainty = (xu[n]+yu[n])/2 # Average uncertainty in position
			#cv.circle(frame,(int(xe[n]),int(ye[n])),int(uncertainty), (255, 255, 0),1)
			cv.circle(frame,(int(xe[n]),int(ye[n])),3, (0, 255, 255),-1)
		# Draw the predicted path
		for n in range(len(xp)): 
			uncertaintyP = (xpu[n]+ypu[n])/2 
			# Draw prediction (circles), with uncertainty as radius
			cv.circle(frame,(int(xp[n]),int(yp[n])),int(uncertaintyP), (0, 0, 255))
			cv.circle(frame,(int(xp[n]),int(yp[n])),3, (255, 255, 255),-1)
	



	########## DISPLAY ##########
	# check to see if the frame should be displayed to our screen
	cv.imshow('Frame', frame)
	cv.imshow('Vision', mask)
	out.write(frame)

	# update the FPS counter
	streamfps.update()
	lastframe = frame_read



################### CLEARING UP ###################
# stop the timer and display information

streamfps.stop()
print("[INFO] elasped time: {:.2f}".format(streamfps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(streamfps.fps_tot()))
print("Measurements: "+ str(listPoints))


out.release()
stream.release()
cv.destroyAllWindows()