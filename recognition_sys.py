# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 16:49:41 2017

@author: zhang
"""
# %%
import numpy as np
import cv2
from skimage.feature import local_binary_pattern
import time
from sklearn.svm import SVC
from sklearn.externals import joblib
path_model='C:/Users/zhang/Dropbox/DIP/Project/'
clf = joblib.load(path_model+'projectmodel_friends_final3.pkl') 
pca = joblib.load(path_model+'projectpca_friends_final3.pkl') 
path='C:/Users/zhang/Dropbox/DIP/Project/'
face_cascade = cv2.CascadeClassifier(path+'haarcascade_frontalface_alt.xml')
def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]
#function to get the LBP feature
def get_lbp(gray):
    row=gray.shape[0]
    col=gray.shape[1]
    neighbors=24
    radius=3
    
    weight=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],
        [0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0]]
    #extract the LBP feature of the whole image
    lbp = local_binary_pattern(gray, 
                               neighbors,
                               radius, 
                               method="uniform")
    local_hist=[]
    for r in range(7):
        for c in range(7):
            #the range of the block
            r_start=r*int(row/7)
            c_start=c*int(col/7)
            if((r+1)*int(row/7)<=row):
                r_end=(r+1)*int(row/7)
            else:
                r_end=row
            if((c+1)*int(col/7)<=col):
                c_end=(c+1)*int(col/7)
            else:
                c_end=col
            if not weight[r][c]==0:
                #get the regional histogram
                (hist_temp, _) = np.histogram(lbp[r_start:r_end,c_start:c_end].ravel(),
                                              bins=np.arange(0, neighbors + 3),
                                              range=(0, neighbors + 2))
                #normalize the histogram
                hist_temp = hist_temp.astype("float")
                hist_temp /= (hist_temp.sum())
                local_hist=local_hist+list(hist_temp*weight[r][c])
    return local_hist
# %%
cap = cv2.VideoCapture(0)
i=0
n_class=13
start = time.clock()
count=[0]*n_class
total=0
text_space_hole=25
#start capture the video
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if(ret==False):
        print('no video captured')
        break

    # Display the resulting frame
    frame=cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    detected_face=np.zeros(frame.shape, np.uint8)
    
    # create a window to count the result
    result_window=np.ones(frame.shape, np.uint8)*255

    for (x,y,w,h) in faces:
        
        if faces.any():
            detected_face=frame[y: y + h, x: x + w]
            gray_face=cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
            resize_gray_face=cv2.resize(gray_face,(140,140))
            name_no=clf.best_estimator_.predict(pca.transform(get_lbp(resize_gray_face)))
            #print(name_no)
            elapsed = (time.clock() - start)
            frame = cv2.rectangle(frame,(x - 5,y - 5),(x+w + 10,y+h +10),(255,0,0),2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame,to_str(name_no),(x+w + 10,y+h +10), font, 1,(255,0,0),1,cv2.LINE_AA)
            count[name_no]=count[name_no]+1
            total=total+1
        else:
            detected_face=np.zeros(frame.shape, np.uint8)
            detected_face=cv2.resize(detected_face,(140,140))
            
    most_confident_no=count.index(max(count))
    #showing the recogniztion result
    for i in range(n_class):
        font = cv2.FONT_HERSHEY_SIMPLEX
        if(total==0):
            # the persons_no
            cv2.putText(result_window,
                        'the-'+to_str(i)+' person',
                        (50,100+i*text_space_hole), 
                        font, 
                        0.5,
                        (0,0,0),
                        1,
                        cv2.LINE_AA)
            cv2.putText(result_window,
                        to_str(0)+'%',
                        (300,100+i*text_space_hole), 
                        font, 
                        0.5,
                        (0,0,0),
                        1,
                        cv2.LINE_AA)
        else:
            cv2.putText(result_window,
                        'the-'+to_str(i)+' person',
                        (50,100+i*text_space_hole), 
                        font, 
                        0.5,
                        (0,0,0),
                        1,
                        cv2.LINE_AA)
            cv2.putText(result_window,
                        to_str(("%.2f" % (count[i]/total*100)))+'%',
                        (300,100+i*text_space_hole), 
                        font, 
                        0.5,
                        (0,0,0),
                        1,
                        cv2.LINE_AA)
    #conclude the result and show it on the screen
    if(total==0):
        cv2.putText(result_window,
                    'no person is detected',
                    (50,100+n_class*text_space_hole), 
                    font, 
                    0.5,
                    (0,0,0),
                    1,
                    cv2.LINE_AA)
    else:
        cv2.putText(result_window,
                    'I am '+to_str(("%.2f" % (max(count)/total*100)))+'% sure that you are the-'+to_str(most_confident_no)+' person',
                    (50,100+n_class*text_space_hole), 
                    font, 
                    0.5,
                    (0,0,0),
                    1,
                    cv2.LINE_AA)
    #press r to restart recogniztion
    if cv2.waitKey(1) & 0xFF == ord('r'):
        count=[0]*n_class
        total=0
    #press q to exit the system
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.imshow('frame',frame)
    cv2.imshow('result',result_window)
# When everything done, release the capture
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
