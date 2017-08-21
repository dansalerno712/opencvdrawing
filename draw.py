import numpy as np
import cv2
import datetime

#global vars for drawing stuff
glLastCenter = None
glCurrentCenter = None
glDrawing = False

#dummy function that does nothing that gets passed into the sliders
def nothing(x):
	pass
	
#mouse callback function
#just toggles drawing on and off with left mouse clicks
def draw_line(event, x, y, flags, param):
	global glDrawing
	
	if event == cv2.EVENT_LBUTTONDOWN:
		glDrawing = True
		
	if event == cv2.EVENT_LBUTTONUP:
		glDrawing = False

#NOTE: img2 must not be bigger than img1
def overlay(img1, img2):
    #create roi in case they are not same size
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    #create mask and inverse mask of img2
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    #cv2.imshow('mask_inv', mask_inv)

    #black out the area of img2 in the roi
    img1_bg = cv2.bitwise_and(roi, roi, mask = mask_inv)

    #cv2.imshow('img1_bg', img1_bg)
    #get the filled in regions of img2
    img2_fg = cv2.bitwise_and(img2, img2, mask = mask)

    #cv2.imshow('img2_fg', img2_fg)
    #combine the fg and bg images
    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    return img1
	
# capture video, 0 means capture the primary camera, 1 or 2 or 3
# means other cameras
cap = cv2.VideoCapture(0)

#glImg is the image that the drawing occurs on. We set it to the same size
#as the image capture frame. This gets overlayed with the camera images later
glImg = np.zeros((int(cap.get(4)),int(cap.get(3)), 3), np.uint8)

#holds dummy image so the user can see their color
colorImg = np.zeros((200,500,3), np.uint8)

#holds image so the user can see their selector color
selectorImg = np.zeros((200, 500, 3), np.uint8)

#create the window, callbacks, and sliders
#windows
cv2.namedWindow('image')
cv2.namedWindow('settings')
cv2.namedWindow('selector')
cv2.namedWindow('mask')
#move windows
cv2.moveWindow('image', 0, 0)
cv2.moveWindow('mask', 0, int(cap.get(4)) + 30)
cv2.moveWindow('selector', int(cap.get(3) + 30), 0)
cv2.moveWindow('settings', int(cap.get(3) + 30), 500)
#callbacks
cv2.setMouseCallback('image', draw_line)
#trackbars
cv2.createTrackbar('R', 'settings', 255, 255, nothing)
cv2.createTrackbar('G', 'settings', 255, 255, nothing)
cv2.createTrackbar('B', 'settings', 255, 255, nothing)
cv2.createTrackbar('Line Width', 'settings', 10, 100, nothing)
cv2.createTrackbar('Hue', 'selector', 0, 180, nothing)
cv2.createTrackbar('Saturation', 'selector', 255, 255, nothing)
cv2.createTrackbar('Value', 'selector', 255, 255, nothing)
cv2.createTrackbar('Range', 'selector', 5, 20, nothing)

# loop until quit
while(True):
    # get a frame from the camera
    ret, frame = cap.read()

    #gauss blur
    gauss = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # convert to hsv
    hsv = cv2.cvtColor(gauss, cv2.COLOR_BGR2HSV)
    
    #get hsv values from the trackbars
    hue = cv2.getTrackbarPos('Hue', 'selector')
    saturation = cv2.getTrackbarPos('Saturation', 'selector')
    value = cv2.getTrackbarPos('Value', 'selector')
    selectorRange = cv2.getTrackbarPos('Range', 'selector')

    #calculate upper and lower ranges of the threhsolds
    lbr = hue - selectorRange
    ubr = hue + selectorRange
    #opencv hue values go from 0 to 180 and wrap, so this is the logic to handle the wrapping
    if lbr < 0:
        lbr = 180 - selectorRange + hue
        ubr = 180
    if ubr > 180:
        ubr = hue - 180 + selectorRange
        lbr = 0

    #define hsv ranges for object to track
    lowerColorThresh = np.array([lbr, saturation, value])
    upperColorThresh = np.array([ubr, 255, 255])

    #threshold the image and degrade slightly to get rid of small, non marker areas
    threshMask = cv2.inRange(hsv, lowerColorThresh, upperColorThresh)
    threshMask = cv2.erode(threshMask, None, iterations = 2)
    threshMask = cv2.dilate(threshMask, None, iterations = 2)

    cv2.imshow("mask", threshMask)

    # find countours
    bcontours = cv2.findContours(threshMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    #default of the center is none
    glLastCenter = glCurrentCenter
    glCurrentCenter = None
    #only do this if there is a contour to be found
    if len(bcontours) > 0:
        #find the largest contour (this should be the thing we are tracking)
        c = max(bcontours, key = cv2.contourArea)
        #calculate minimum enclosing cirlce
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        #get the moments of the contour in order to calculate the center of the contour
        M = cv2.moments(c)

        if M["m00"] == 0:
            glCurrentCenter = (0, 0)
        else:
            glCurrentCenter = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        #only draw circle for large enough radius
        if (radius >= 10):
            cv2.circle(frame, (int(x), int(y)), int(radius),
                (0, 255, 255), 2)
            cv2.circle(frame, glCurrentCenter, 5, (0, 0, 255), -1)
        else:
            glCurrentCenter = None

    #get trackbar values
    r = cv2.getTrackbarPos('R', 'settings')
    b = cv2.getTrackbarPos('B', 'settings')
    g = cv2.getTrackbarPos('G', 'settings')
    lineWidth = cv2.getTrackbarPos('Line Width', 'settings')

    #create a dummy image to show the color you are currently drawing with
    colorImg[:] = [b, g, r]

    #show selector img
    selectorImg[:] = [hue, saturation, value]
    selectorImg = cv2.cvtColor(selectorImg, cv2.COLOR_HSV2BGR)

    #draw on the screen
    if (glDrawing and glCurrentCenter != None and glLastCenter != None):
        cv2.line(glImg, glLastCenter, glCurrentCenter, (b, g, r), lineWidth)
    
    #cv2.imshow('glImg', glImg)
    #overlay the camera image and the drawn image
    final = overlay(frame, glImg)

    #redraw the dot at the center of the minimum enclosing circle
    if (glCurrentCenter != None):
        cv2.circle(final, glCurrentCenter, 5, (0, 0, 0), -1)

    #show image with drawings and the masked image. flip it so it is not mirrored
    cv2.imshow("image", cv2.flip(final, 1))

    #show color choice on settings frame
    cv2.imshow('settings', colorImg)

    #show selector
    cv2.imshow('selector', selectorImg)

    k = cv2.waitKey(1)
    
    # if they press q, exit
    if k & 0xFF == ord('q'):
        break
    #if they press esc, clear the drawings
    elif k == 27:
        glImg = np.zeros((int(cap.get(4)),int(cap.get(3)), 3), np.uint8)
    elif k == 0:
        glDrawing = not glDrawing
    elif k & 0xFF == ord('s'):
        date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        cv2.imwrite('images/screenshots/' + date + '.png', cv2.flip(final, 1))

# release camera and close windows
cap.release()
cv2.destroyAllWindows()


