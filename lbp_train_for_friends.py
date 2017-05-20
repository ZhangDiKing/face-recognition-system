import cv2
import time
import numpy as np
from skimage.feature         import local_binary_pattern
from matplotlib              import pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm             import SVC
from sklearn.svm             import LinearSVC
from sklearn.decomposition   import PCA
from sklearn.metrics         import confusion_matrix
from sklearn.externals       import joblib

train_data   = []
train_labels = []
test_data    = []
test_labels  = []
neighbors    = 24
radius       = 3
sample       = 13
block_size   = 20
im_size      = 140
class_size   = 300
weight       = [[1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1],
                [0,1,1,1,1,1,0],
                [0,1,1,1,1,1,0],
                [0,1,1,1,1,1,0],
                [0,1,1,1,1,1,0]]


for i in range(sample):
    #convert the number of folder to string
    no_folder = str(i)
    no_folder = no_folder.rjust(2, '0')
    
    for j in range(class_size):
        #convert the number of image to string
        img_no = str(j + 1)
        img_no = img_no.rjust(2, '0')
       
        save_path= 'C:/Users/zhang/Documents/DIP/project/dataset_for_friends/' + no_folder + '_' + img_no + '.jpg'
        gray     = cv2.imread(save_path, 0)
        
        #get LBP histogram
        if not gray == None:
            lbp        = local_binary_pattern(gray, neighbors, radius, method = "uniform")
            #print(save_path)
            local_hist = []
            for r in range(int(im_size / block_size)):
                for c in range(int(im_size / block_size)):
                    if not weight[r][c] == 0
                        (hist_temp, _) = np.histogram(lbp[r * block_size:(r + 1) * block_size, c * block_size:(c + 1)  *block_size].ravel(),
                                                      bins = np.arange(0, neighbors + 3),
                                                      range = (0, neighbors + 2))
                        # normalize the histogram
                        hist_temp      = hist_temp.astype("float")
                        hist_temp /    = (hist_temp.sum())
                        local_hist     = local_hist+list(hist_temp*weight[r][c])
            
            if j % 9 == 0 or j % 9 == 1:
                test_labels.append(i)
                test_data.append(local_hist)
            else:
                train_labels.append(i)
                train_data.append(local_hist)
                
#show the sample of LBP jpg
save_path  = 'sample_face.jpg'
gray       = cv2.imread(save_path,0)

#get LBP histogram
local_hist = []
lbp        = cv2.imread(save_path,0)

#rescale the lbp into 0-255
r     = 2
c     = 4
block = lbp[r * block_size : (r + 1) * block_size, c * block_size : (c + 1) * block_size]
lbp   = lbp.astype("float") / 9 * 255
lbp   = lbp.astype("uint8")

#show the LBP of the sample
cv2.imshow('lbp_image', lbp[r * block_size : (r + 1) * block_size, c * block_size : (c + 1) * block_size])
plt.title('LBP histogram')
plt.xlabel('uniform values of LBP')
plt.ylabel('H')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#print the histogram of certain region of the image
plt.hist(block.ravel(), neighbors + 2, [0, neighbors + 2]); 
hist_temp, _ = np.histogram(block.ravel(),bins=np.arange(0, neighbors + 3),range=(0, neighbors + 2))
hist_temp    = hist_temp.astype("float")
hist_normal  = hist_temp/(hist_temp.sum())
print(hist_temp)
print(hist_normal)

# start training the model
#SVM with rbf kernel
print("Fitting the classifier to the training set")
t0         = time.clock()
param_grid = {'C': [0.1, 1, 5, 10, 20, 100, 1000, 10000],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
clf        = GridSearchCV(SVC(kernel = 'rbf', class_weight = 'balanced'), param_grid)
clf        = clf.fit(train_data, train_labels)

print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc       = np.mean(clf.best_estimator_.predict(test_data) == test_labels)
acc_train = np.mean(clf.best_estimator_.predict(train_data) == train_labels)
print(acc)
print(acc_train)

#SVM with linear kernel
print("Fitting the classifier to the training set")
t0         = time.clock()
param_grid = {'C': [0.1,0.5,1,5,10,100,1000,10000]}
clf        = GridSearchCV(LinearSVC(), param_grid)
clf        = clf.fit(train_data, train_labels)

print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc       = np.mean(clf.best_estimator_.predict(test_data) == test_labels)
acc_train = np.mean(clf.best_estimator_.predict(train_data) == train_labels)
print(acc)
print(acc_train)

#PCA of the LBP
max_acc      = 0
n_components = 220
n_classes    = 11
t0           = time.clock()
pca          = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True).fit(train_data)
print("done in %0.3fs" % (time.clock() - t0))

#PCA transform of the training data and test data
t0 = time.clock()
lbp_train_pca = pca.transform(train_data)
lbp_test_pca = pca.transform(test_data)
print("done in %0.3fs" % (time.clock() - t0))

#SVM with RBF kernel after PCA
print("Fitting the classifier to the training set")
t0         = time.clock()
param_grid = {'C': [0.01, 0.1, 1, 7, 10, 30, 100, 1000],
              'gamma': [0.00001, 0.0001, 0.004, 0.0005, 0.006, 0.001, 0.01, 0.1, 1]}
clf        = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf        = clf.fit(lbp_train_pca, train_labels)
print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
acc         = np.mean(clf.best_estimator_.predict(lbp_test_pca) == test_labels)
acc_train   = np.mean(clf.best_estimator_.predict(lbp_train_pca) == train_labels)
y_pred      = clf.predict(lbp_test_pca)
confuse_mat = confusion_matrix(test_labels, y_pred, labels = range(n_classes))
print(acc)
print(acc_train)
print(confusion_matrix(test_labels, y_pred, labels = range(n_classes)))
path    = 'C:/Users/zhang/Documents/DIP/project'
max_acc = 0
if(acc > max_acc):
    joblib.dump(clf, path+'model_friends_final.pkl')
    joblib.dump(pca,path+'pca_friends_final.pkl')
    max_acc = acc
    

#SVM with linear kernel after PCA
print("Fitting the classifier to the training set")
t0          = time.clock()
param_grid  = {'C': [i / 1000.0 for i in range(1,10)]}
clf         = GridSearchCV(LinearSVC(), param_grid)
print("done in %0.3fs" % (time.clock() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
clf         = clf.fit(lbp_train_pca, train_labels)
acc         = np.mean(clf.best_estimator_.predict(lbp_test_pca) == test_labels)
acc_train   = np.mean(clf.best_estimator_.predict(lbp_train_pca) == train_labels)
y_pred      = clf.predict(lbp_test_pca)
confuse_mat = confusion_matrix(test_labels, y_pred, labels = range(n_classes))
print(acc)
print(acc_train)
print(confusion_matrix(test_labels, y_pred, labels = range(n_classes)))
path='C:/Users/zhang/Documents/DIP/project'
if(acc > max_acc):
    joblib.dump(clf, path+'model_friends_final2.pkl')
    max_acc = acc
    joblib.dump(pca,path+'pca_friends_final2.pkl')
