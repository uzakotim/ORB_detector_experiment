import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import math
import time

def lstCalculate3DPoint(P,CurPoint,camPos,scale):
    x_i = np.asmatrix([CurPoint[0],CurPoint[1],1]).T
    X_i = np.asmatrix(P).I*x_i
    point = [X_i[0][0]*scale + camPos[0],
                X_i[1][0]*scale + camPos[1],
                X_i[2][0]*scale + camPos[2]]
    return point

def arrayCalculateIntrinsicMatrix(width,height,fov):
    x = width/2
    y = height/2
    fov = fov*(math.pi/180)
    f_x = x/math.tan(fov/2)
    f_y = y/math.tan(fov/2)
    K = np.array([[f_x,0,x],[0,f_y,y],[0,0,1]])
    return K

def main():
    scale = 1
    cam_xyz = []
    lm_xyz = []
    camPos = [0,0,0]
    flagFirstPhoto = True

    prevImage = []

    orb = cv2.ORB_create()
    
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("TEST")

    while True:
        print("Detector task")
        ret, frame = cam.read()
        moved = True
        lm_xyz=[]
        cam_xyz=[]
        
        if not ret:
            print("Failed to grab frame")
            continue

        
        if  flagFirstPhoto:
            prevKeyPoints, prevDes = orb.detectAndCompute(frame, None)
            prevImage = frame
            demoImage = frame
            flagFirstPhoto = False
        
        if  not flagFirstPhoto:
            
            curKeyPoints, curDes = orb.detectAndCompute(frame,None)
        
            bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
            matches = bf.match(prevDes,curDes)
            matches = sorted(matches, key = lambda x: x.distance)

            demoImage = cv2.drawMatches(prevImage,prevKeyPoints,frame,curKeyPoints,matches[:10], prevImage, flags=2)
            
            lstPrevPoints= []
            lstCurPoints = []

            for mat in matches[:50]:
                prevImage_idx = mat.queryIdx
                curImage_idx = mat.trainIdx 

                (x1,y1) = prevKeyPoints[prevImage_idx].pt
                (x2,y2) = curKeyPoints[curImage_idx].pt

                lstPrevPoints.append([x1,y1])
                lstCurPoints.append([x2,y2])

                if (x1==x2):
                    moved = False
                    continue
            

            if moved:
                K = arrayCalculateIntrinsicMatrix(width = frame.shape[1],height = frame.shape[0], fov = 60)
        
                E, mask = cv2.findFundamentalMat(np.float32(lstCurPoints),np.float32(lstPrevPoints),cv2.FM_8POINT)
          
                points, R, t, mask = cv2.recoverPose(E,np.float32(lstCurPoints),np.float32(lstPrevPoints), K)
                R = np.asmatrix(R).I

                cam_xyz.append([camPos[0] + t[0], camPos[1]+t[1], camPos[2]+t[2]])

                C = np.hstack((R,t))
                P = np.asmatrix(K)*np.asmatrix(C)
                
                for i in range(len(lstCurPoints)):
                    lm_xyz.append(lstCalculate3DPoint(P,lstCurPoints[i],camPos,scale))
                
                camPos = [camPos[0]+t[0], camPos[1]+t[1],camPos[2]+t[2]]
                prevImage = frame
                prevKeyPoints = curKeyPoints
                prevDes = curDes

                arrLm_xyz = np.array(lm_xyz)
                arrCam_xyz = np.array(cam_xyz)
                
            else: continue
            
        cv2.imshow("TEST",demoImage)
        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        time.sleep(0.25)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

