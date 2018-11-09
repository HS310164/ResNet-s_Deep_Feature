# -*- coding: utf-8 -*-

import numpy as np
import os
import sys
import cv2

def detect(frame):

    th=7000
    hth=200
    #extra=100
    extra=30

    maskedframe = frame.copy()
    smooth = cv2.medianBlur(frame,7)
    hsv = cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(hsv[:,:,0],np.uint8)
    returnmask = np.zeros_like(hsv[:,:,0],np.uint8)
    bool = np.zeros_like(hsv,np.bool)

    #print(hsv[:,:,0])
    height,width = hsv.shape[:2]

    mask[(hsv[:,:,0]>=0) & (hsv[:,:,0]<=15) & (hsv[:,:,1]>=50) & (hsv[:,:,2]>=30)]=255
    """mask = cv2.morphologyEx(
		mask,
		cv2.MORPH_OPEN,
		kernel=np.ones((5,5), np.uint8),
		iterations=3
		)"""

    mask = cv2.erode(
        mask,
        kernel=np.ones((5,5), np.uint8),
        iterations=6
    )

    mask = cv2.dilate(
        mask,
        kernel=np.ones((5,5), np.uint8),
        iterations=20
    )

    binary = mask.copy()
    #print(binary)

    _,contours,_ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    Y=[]
    for i,cnt in enumerate(contours):
        #print(cnt)

        M = cv2.moments(cnt)
        cy = int(M['m01']/M['m00'])
        Y.append([cy,i])

    Y.sort(reverse=True)

    for _,i in Y[:3]:

        cnt = contours[i]
        
        area = cv2.contourArea(cnt)

        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        x,y,w,h = cv2.boundingRect(cnt)

        xx = x-extra if x-extra >= 0 else 0
        yy = y-extra if y-extra >= 0 else 0
        ww = w+extra*2 if x+w+extra*2 <= width else width-xx
        hh = h+extra*2 if y+h+extra*2 <= height else height-yy

        # if area>=th and y+h>=hth:
        if area>=th:
            
            bool[yy:yy+hh,xx:xx+ww,:] = 1
            # bool[y:y+h,x:x+w,:] = 1

    maskedframe[~bool] = 0
    returnmask[bool[:,:,0]] = 1


    return maskedframe,returnmask

def detect2(frame):

    th=7000
    hth=200
    extra=100

    maskedframe = frame.copy()
    smooth = cv2.medianBlur(frame,7)
    hsv = cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(hsv[:,:,0],np.uint8)
    returnmask = np.zeros_like(hsv[:,:,0],np.uint8)
    bool = np.zeros_like(hsv,np.bool)

    height,width = hsv.shape[:2]

    mask[(hsv[:,:,0]>=0) & (hsv[:,:,0]<=15) & (hsv[:,:,1]>=50) & (hsv[:,:,2]>=30)]=255

    # mask = cv2.morphologyEx(
    #     	mask,
    #     	cv2.MORPH_OPEN,
    #     	kernel=np.ones((5,5), np.uint8),
    #     	iterations=3
    #     	)

    mask = cv2.erode(
        mask,
        kernel=np.ones((5,5), np.uint8),
        iterations=3
    )

    mask = cv2.dilate(
        mask,
        kernel=np.ones((5,5), np.uint8),
        iterations=20
    )

    binary = mask.copy()

    multi_mask = np.stack((mask,)*3)
    multi_mask = multi_mask.transpose([1,2,0])
    bool[multi_mask==255]=True

    maskedframe[~bool] = 0
    returnmask[bool[:,:,0]] = 1


    return maskedframe,returnmask

def detect3(frame):

    th=7000
    hth=200
    extra=100

    maskedframe = frame.copy()
    smooth = cv2.medianBlur(frame,7)
    hsv = cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV)
    mask = np.zeros_like(hsv[:,:,0],np.uint8)
    returnmask = np.zeros_like(hsv[:,:,0],np.uint8)
    bool = np.zeros_like(hsv,np.bool)

    height,width = hsv.shape[:2]

    mask[(hsv[:,:,0]>=0) & (hsv[:,:,0]<=15) & (hsv[:,:,1]>=50) & (hsv[:,:,2]>=30)]=255

    """mask = cv2.morphologyEx(
		mask,
		cv2.MORPH_OPEN,
		kernel=np.ones((5,5), np.uint8),
		iterations=3
		)"""

    mask = cv2.erode(
        mask,
        kernel=np.ones((5,5), np.uint8),
        iterations=6
    )

    mask = cv2.dilate(
        mask,
        kernel=np.ones((5,5), np.uint8),
        iterations=20
    )

    binary = mask.copy()

    contours,_ = cv2.findContours(
        binary,
        cv2.cv.CV_RETR_EXTERNAL,
        cv2.cv.CV_CHAIN_APPROX_NONE
    )

    Y=[]
    for i,cnt in enumerate(contours):

        M = cv2.moments(cnt)
        cy = int(M['m01']/M['m00'])
        Y.append([cy,i])

    Y.sort(reverse=True)

    pos = []
    roi = []
    for _,i in Y[:2]:

        cnt = contours[i]
        
        area = cv2.contourArea(cnt)

        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        x,y,w,h = cv2.boundingRect(cnt)

        # xx = x-extra if x-extra >= 0 else 0
        # yy = y-extra if y-extra >= 0 else 0
        # ww = w+extra*2 if x+w+extra*2 <= width else width-xx
        # hh = h+extra*2 if y+h+extra*2 <= height else height-yy

        # if area>=th and y+h>=hth:
        if area>=th:
            
            roi.append([x,y,w,h])

            #bool[yy:yy+hh,xx:xx+ww,:] = 1
            bool[y:y+h,x:x+w,:] = 1

    maskedframe[~bool] = 0
    returnmask[bool[:,:,0]] = 1


    return maskedframe,roi

def detect_motion(prevs,now):

    return_now = now.copy()

    if prevs is None:
        return_now[:,:] = 0
        return return_now
        
    gprevs = cv2.cvtColor(prevs,cv2.COLOR_BGR2GRAY)
    gnow = cv2.cvtColor(now,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(gprevs,gnow, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    th = mag.max()*0.5
    bool = mag>th
    multi_bool = np.stack((bool,)*3)
    multi_bool = multi_bool.transpose([1,2,0])
    return_now[~multi_bool] = 0

    return return_now
