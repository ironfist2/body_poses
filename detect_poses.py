import math
import numpy as np
import cv2
import pickle
from sklearn import tree
import time

#load trained model
pickle_file = open('train_model.pickle','rb')
train_model = pickle.load(pickle_file)['log']
cap = cv2.VideoCapture('cc.mp4')            #filename of your video input. (0) for webcam
template = cv2.imread('template.png',0)
fgbg = cv2.BackgroundSubtractorMOG()
face_cascade = cv2.CascadeClassifier('haarcascade.xml')
# out = cv2.VideoWriter('output4.avi',-1, 24, (640,480))  if you want to save the output uncomment.
poses = dict()
poses[0] = "Both hands up"
poses[1] = "Both hands straight at 90 degree with face"
poses[2] = "Both hands down"
poses[3] = "Left hand at 150 and Right at 90"
poses[4] = "Left hand at 90 and Right at 150"
poses[5] = "Left hand at 60 and Right at 90"
poses[6] = "Left hand at 60 and Right at 150"
poses[7] = "Left hand at 90 and Right at 60"
poses[8] = "Left hand at 150 and Right at 60"
tem = 0
theta = []
while(cap.isOpened()):                  #True for webcam
    ret, frame = cap.read()
    if ret == True:
        #################  BackGround Subtraction ###############
        start = time.time()
        frame = cv2.resize(frame,(640,480),interpolation=cv2.INTER_AREA)
        frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        kernel=np.ones((3,3),np.uint8)
        kernel2=np.ones((5,5),np.uint8)
        frame_gray = cv2.GaussianBlur(frame_gray,(5,5),0)
        fgmask = fgbg.apply(frame_gray,learningRate=0.0001)
        fgmask=cv2.GaussianBlur(fgmask,(5,5),0)
        fgmask=cv2.dilate(fgmask,kernel,iterations=5)
        fgmask=cv2.erode(fgmask,kernel,iterations=4)
        fgmask = cv2.dilate(fgmask,kernel2,iterations=5)
        fgmask = cv2.erode(fgmask,kernel2,iterations=4)
        ret,fgmask = cv2.threshold(fgmask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        fgmask_copy = fgmask.copy()
        # cv2.imshow('frame',fgmask)
        # k = cv2.waitKey(30) & 0xff
        # if k == 27:
        #     break
    
        temp = frame.copy()
        contours, hierarchy = cv2.findContours(fgmask_copy,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        all_cnts_area = []
        if len(contours)<1:
            continue
        for i in contours:
            area = cv2.contourArea(i)
            all_cnts_area.append(area)
        inp = np.argmax(all_cnts_area)
        x_,y_,w,h = cv2.boundingRect(contours[inp])
        cropped_foreground =  fgmask[y_:y_+h,x_:x_+w]
        cropped_frame = frame[y_:y_+h,x_:x_+w]
        cv2.imshow('vh',cropped_foreground)
        if cropped_foreground.shape[0] >= 260 and cropped_foreground.shape[1] >= 130:

        ####### Template matching Torso Detection #################    
            result = {}
            for i in range(150,260,20):                     #range of scales of template
                template = cv2.resize(template,(i/2,i),interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(cropped_foreground,template,cv2.TM_CCORR_NORMED)
                score = np.amax(res)
                result[score] = i
            ma = max(result.keys())
            size_ = result[ma]
            template = cv2.resize(template,(size_/2,size_),interpolation=cv2.INTER_AREA)
            m, n = template.shape[::-1]
            res = cv2.matchTemplate(cropped_foreground,template,cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            top_left = (top_left[0]+x_,top_left[1]+y_)
            bottom_right = (top_left[0] + m, top_left[1] + n)
            cv2.rectangle(temp,(top_left[0]+size_/6,top_left[1]),(top_left[0]+size_/3,top_left[1]+size_/3),255,2)
            cv2.rectangle(temp,(top_left[0],top_left[1]+size_/3),bottom_right,255,2)
            cv2.circle(temp,(top_left[0]+m/2,top_left[1]+n/2),3,(0,0,255),-1)
            # end = time.time()

            ################ Hands Detection,Face Detection  #############
            masked_image = cv2.bitwise_and(cropped_frame,cropped_frame,mask=cropped_foreground)
            faces = face_cascade.detectMultiScale(cropped_frame,scaleFactor = 1.05,minNeighbors=4,minSize=(20,20))
            if len(faces) != 0:
                (x,y,w,h) = faces[0]
                x = x + x_
                y = y + y_
                
                cv2.rectangle(temp,(x,y),(x+w,y+h),(255,0,0),2)
                if tem <= 10:
                    cropped_face = frame[y:y+h,x:x+w]
                    b,g,r = cv2.split(cropped_face)
                    blue_mean = np.mean(b)
                    green_mean = np.mean(g)
                    red_mean = np.mean(r)
                    skin_color = np.uint8([[[blue_mean,green_mean,red_mean]]])
                    hsv_skin = cv2.cvtColor(skin_color,cv2.COLOR_BGR2HSV)           #mean skin colour in hsv
                masked_image[y-y_:y-y_+h,x-x_:x-x_+w] = 255
                masked_image[:,top_left[0]-x_:bottom_right[0]-x_] = 255
                masked_hsv = cv2.cvtColor(masked_image,cv2.COLOR_BGR2HSV)
                mask_skin = cv2.inRange(masked_hsv,np.array([.25*hsv_skin[0][0][0],.65*hsv_skin[0][0][1],.85*hsv_skin[0][0][2]]).astype(np.float32),np.array([2*hsv_skin[0][0][0],255,255]).astype(np.float32))
                mask_skin = cv2.dilate(mask_skin,kernel,iterations=5)
                mask_skin = cv2.erode(mask_skin,kernel,iterations=2)
                ms = cv2.bitwise_and(masked_image,masked_image,mask = mask_skin)
                cv2.imshow('sd',ms)
                cnt,hec = cv2.findContours(mask_skin,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                all_cnts_area = []
                all_cnts_area_copy = []
                for i in cnt:
                    area = cv2.contourArea(i)
                    all_cnts_area.append(area)
                    all_cnts_area_copy.append(area)
                all_cnts_area.sort()
                if len(cnt) < 2:
                    continue
                for i in range(len(all_cnts_area)):
                    if all_cnts_area_copy[i] == all_cnts_area[-1]:
                        index1 = i                                        #####Hand1########
                    if all_cnts_area_copy[i] == all_cnts_area[-2]:        
                        index2 = i                                        #####Hand2########
                M1 = cv2.moments(cnt[index1])
                M2 = cv2.moments(cnt[index2])
                if M2['m00']==0 or M1['m00']==0:
                    continue
                c1x = int(M1['m10']/M1['m00']) 
                c1y = int(M1['m01']/M1['m00'])
                c2x = int(M2['m10']/M2['m00']) 
                c2y = int(M2['m01']/M2['m00'])
                c1x,c1y = c1x+x_,c1y+y_
                c2x,c2y = c2x+x_,c2y+y_
                c3x = x + w/2
                c3y = y + h/2
                cv2.circle(temp,(c1x,c1y),3,(0,0,255),-1)
                cv2.circle(temp,(c2x,c2y),3,(0,0,255),-1)
                cv2.circle(temp,(c3x,c3y),3,(0,0,255),-1)
                cv2.line(temp,(c1x,c1y),(top_left[0]+m/2,top_left[1]+n/2),(0,255,0),3)
                cv2.line(temp,(c2x,c2y),(top_left[0]+m/2,top_left[1]+n/2),(0,255,0),3)
                cv2.line(temp,(c3x,c3y),(top_left[0]+m/2,top_left[1]+n/2),(0,255,0),3)

            #     ###################  Angle Calculation ###################
                s1 = math.sqrt((c3y - (top_left[1] + n/2))**2 + (c3x - (top_left[0] + m/2))**2)
                s2 = math.sqrt((c1y - (top_left[1] + n/2))**2 + (c1x - (top_left[0] + m/2))**2)
                s3 = math.sqrt((c3y - c1y)**2 + (c3x - c1x)**2)
                theta1 =  math.degrees(math.acos((s1**2 + s2**2 - s3**2)/(2*s1*s2)))
                s1 = math.sqrt((c3y - (top_left[1] + n/2))**2 + (c3x - (top_left[0] + m/2))**2)
                s2 = math.sqrt((c2y - (top_left[1] + n/2))**2 + (c2x - (top_left[0] + m/2))**2)
                s3 = math.sqrt((c3y - c2y)**2 + (c3x - c2x)**2)
                theta2 =  math.degrees(math.acos((s1**2 + s2**2 - s3**2)/(2*s1*s2)))
                font = cv2.FONT_HERSHEY_SIMPLEX 
                
                ### if c1x < c2x -> theta1 is left angle , theata 2 right angle #### 
                if c1x < c2x :
                    left_theta = theta1
                    right_theta = theta2
                else:
                    left_theta = theta2
                    right_theta = theta1
                cv2.putText(temp,"left hand angle",(10,20),font,1,(255,255,255),2)
                cv2.putText(temp,"right hand angle",(300,20),font,1,(255,255,255),2)
                cv2.putText(temp,str(right_theta),(10,50), font, 1,(255,255,255),2)
                cv2.putText(temp,str(left_theta),(300,50), font, 1,(255,255,255),2)
                tem += 1
                if tem%24 == 0:
                    print poses[train_model.predict([[1,right_theta,left_theta,right_theta**2,left_theta**2]])[0]]
                
                # print (end-start)/
        cv2.imshow('frame4',temp)
        # out.write(temp)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break

cap.release()
# out.release()
cv2.destroyAllWindows()
