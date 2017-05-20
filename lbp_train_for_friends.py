# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:09:38 2017

@author: zhang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:26:28 2017

@author: zhang
"""
import cv2
from skimage.feature import local_binary_pattern
import numpy as np
from matplotlib import pyplot as plt
train_data=[]
train_labels = []
test_data=[]
test_labels = []
neighbors=24
radius=3

'''
        plt.hist(lbp.ravel(),len(hist),[0,len(hist)]); plt.show()
        lbp_im=np.array(lbp/10.0*255.0,dtype=np.uint8)
        cv2.imshow('gray_face',gray)
        cv2.imshow('lbp',lbp_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
# %%
sample=13
weight=[[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[1,1,1,1,1,1,1],[0,1,1,1,1,1,0],
        [0,1,1,1,1,1,0],[0,1,1,1,1,1,0],[0,1,1,1,1,1,0]]

block_size=20
re_size=140
for i in range(sample):
    no_folder=str(i)
    no_folder=no_folder.rjust(2, '0')
    for j in range(300):
        img_no=str(j+1)
        img_no=img_no.rjust(2, '0')
        save_path='C:/Users/zhang/Documents/DIP/project/dataset_for_friends/'+no_folder+'_'+img_no+'.jpg'
        gray = cv2.imread(save_path,0)
        #get LBP histogram
        if not gray==None:
            lbp = local_binary_pattern(gray,neighbors,radius, method="uniform")
            print(save_path)
            local_hist=[]
            for r in range(int(re_size/block_size)):
                for c in range(int(re_size/block_size)):
                    if not weight[r][c]==0:
                        
                        (hist_temp, _) = np.histogram(
                                                        lbp[r*block_size:(r+1)*block_size,c*block_size:(c+1)*block_size].ravel(),
                                                        bins=np.arange(0, neighbors + 3),
                                                        range=(0, neighbors + 2))
                        # normalize the histogram
                        hist_temp = hist_temp.astype("float")
                        hist_temp /= (hist_temp.sum())
                        local_hist=local_hist+list(hist_temp*weight[r][c])
            
            if j%9==0 or j%9==1:
                test_labels.append(i)
                test_data.append(local_hist)
            else:
                train_labels.append(i)
                train_data.append(local_hist)
# %%
from matplotlib import pyplot as plt
save_path='C:/Users/zhang/Documents/DIP/project/my_face_dataset/'+'35_04.jpg'
gray = cv2.imread(save_path,0)
#get LBP histogram
local_hist=[]
lbp=cv2.imread(save_path,0)
for r in range(int(re_size/block_size)):
    for c in range(int(re_size/block_size)):
        if not weight[r][c]==0:
            lbp[r*block_size:(r+1)*block_size,c*block_size:(c+1)*block_size]= local_binary_pattern(
                    gray[r*block_size:(r+1)*block_size,c*block_size:(c+1)*block_size], 
                    neighbors,
                    radius, 
                    method="uniform")
            (hist_temp, _) = np.histogram(lbp.ravel(),bins=np.arange(0, neighbors + 3),range=(0, neighbors + 2))
            # normalize the histogram
            hist_temp = hist_temp.astype("float")
            hist_temp /= (hist_temp.sum())
            local_hist=local_hist+list(hist_temp)
        else:
            lbp[r*block_size:(r+1)*block_size,c*block_size:(c+1)*block_size]=np.zeros(shape=(block_size,block_size))
#rescale the lbp into 0-255
r=2
c=4
block=lbp[r*block_size:(r+1)*block_size,c*block_size:(c+1)*block_size]
lbp=lbp.astype("float")/9*255
lbp=lbp.astype("uint8")
cv2.imshow('lbp_image',lbp[r*block_size:(r+1)*block_size,c*block_size:(c+1)*block_size])
plt.hist(block.ravel(),neighbors + 2,[0, neighbors + 2]); 
hist_temp, _ = np.histogram(block.ravel(),bins=np.arange(0, neighbors + 3),range=(0, neighbors + 2))
hist_temp = hist_temp.astype("float")
hist_normal =hist_temp/(hist_temp.sum())
print(hist_temp)
print(hist_normal)
plt.title('LBP histogram')
plt.xlabel('uniform values of LBP')
plt.ylabel('H')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
from sklearn.svm import LinearSVC
from sklearn import svm
best_c=0
best_acc=0
#68
for t in range(1,40):
    model = LinearSVC(C=t)
    model.fit(train_data, train_labels)
    acc=np.mean(model.predict(test_data) == test_labels)
    print(acc)
    if(acc>=best_acc):
        best_c=t
        best_acc=acc
model = svm.SVC(C=best_c)
model.fit(train_data, train_labels)
acc=np.mean(model.predict(test_data) == test_labels)
print('train_acc=',np.mean(model.predict(train_data) == train_labels))
prediction=model.predict(train_data)
# %%
from sklearn import svm
best_c=0
best_acc=0
for t in range(1,2):
    model = svm.SVC(C=t)
    model.fit(train_data, train_labels)
    acc=np.mean(model.predict(test_data) == test_labels)
    print(acc)
    if(acc>=best_acc):
        best_c=t/100
        best_acc=acc
model = svm.SVC(C=best_c)
model.fit(train_data, train_labels)
acc=np.mean(model.predict(train_data) == train_labels)
print(acc)
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
print("Fitting the classifier to the training set")
t0 = time.clock()
param_grid = {'C': [0.1,1,5,10,20,100,1000,10000],
              'gamma': [0.001,0.01,0.1,1,10,100], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(train_data, train_labels)
print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc=np.mean(clf.best_estimator_.predict(test_data) == test_labels)
print(acc)
acc_train=np.mean(clf.best_estimator_.predict(train_data) == train_labels)
print(acc_train)
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import time
print("Fitting the classifier to the training set")
t0 = time.clock()
param_grid = {'C': [0.1,0.5,1,5,10,100,1000,10000],
              }
clf = GridSearchCV(LinearSVC(), param_grid)
clf = clf.fit(train_data, train_labels)
print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc=np.mean(clf.best_estimator_.predict(test_data) == test_labels)
print(acc)
acc_train=np.mean(clf.best_estimator_.predict(train_data) == train_labels)
print(acc_train)
# %%
max_acc=0
# %%
n_components = 220
from sklearn.decomposition import PCA
import time
#print("Extracting the top %d eigenfaces from %d faces"
#      % (n_components, X_train.shape[0]))
t0 = time.clock()
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(train_data)
print("done in %0.3fs" % (time.clock() - t0))
#eigenfaces = pca.components_.reshape((n_components, h, w))
n_classes=51
#print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time.clock()
lbp_train_pca = pca.transform(train_data)
lbp_test_pca = pca.transform(test_data)
print("done in %0.3fs" % (time.clock() - t0))
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import time
from sklearn.metrics import confusion_matrix
print("Fitting the classifier to the training set")
t0 = time.clock()
param_grid = {'C': [0.01,0.1,1,7,10,30,100,1000],
              'gamma': [0.00001,0.0001,0.004,0.0005,0.006,0.001,0.01,0.1,1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(lbp_train_pca, train_labels)
print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc=np.mean(clf.best_estimator_.predict(lbp_test_pca) == test_labels)
print(acc)
acc_train=np.mean(clf.best_estimator_.predict(lbp_train_pca) == train_labels)
print(acc_train)
y_pred = clf.predict(lbp_test_pca)
confuse_mat=confusion_matrix(test_labels, y_pred, labels=range(n_classes))
print(confusion_matrix(test_labels, y_pred, labels=range(n_classes)))

max_acc=0
path='C:/Users/zhang/Documents/DIP/project'
# %%
from sklearn.externals import joblib
if(acc>max_acc):
    joblib.dump(clf, path+'model_friends_final3.pkl')
    max_acc=acc
    joblib.dump(pca,path+'pca_friends_final3.pkl')
# %%
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
import time
print("Fitting the classifier to the training set")
t0 = time.clock()
param_grid = {'C': [i/1000.0 for i in range(1,10)],
              }
clf = GridSearchCV(LinearSVC(), param_grid)
clf = clf.fit(lbp_train_pca, train_labels)
print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc=np.mean(clf.best_estimator_.predict(lbp_test_pca) == test_labels)
print(acc)
acc_train=np.mean(clf.best_estimator_.predict(lbp_train_pca) == train_labels)
print(acc_train)
max_acc=0
path='C:/Users/zhang/Documents/DIP/project'