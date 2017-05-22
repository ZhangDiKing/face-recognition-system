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

#get regional LBP histogram from the gray image
def get_lbp(gray):
    row       = gray.shape[0]
    col       = gray.shape[1]
    neighbors = 24
    radius    = 3
    weight    = [[1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1],
                 [1,1,1,1,1,1,1],
                 [0,1,1,1,1,1,0],
                 [0,1,1,1,1,1,0],
                 [0,1,1,1,1,1,0],
                 [0,1,1,1,1,1,0]]

    #extract the LBP feature of the whole image
    lbp = local_binary_pattern(gray, 
                             neighbors,
                             radius, 
                             method="uniform")
    local_hist=[]
    for r in range(7):
        for c in range(7):
            
            #the range of the block
            r_start = r * int(row / 7)
            c_start = c * int(col / 7)
            
            if((r + 1) * int(row / 7) <= row):
                r_end = (r + 1) * int(row / 7)
            else:
                r_end = row
            if((c + 1) * int(col / 7) <= col):
                c_end = (c + 1) * int(col / 7)
            else:
                c_end = col
            if not weight[r][c] == 0:
                #get the regional histogram
                (hist_temp, _) = np.histogram(lbp[r_start:r_end, c_start:c_end].ravel(),
                                              bins=np.arange(0, neighbors + 3),
                                              range=(0, neighbors + 2))
                #normalize the histogram
                hist_temp = hist_temp.astype("float")
                hist_temp /= (hist_temp.sum())
                
            local_hist = local_hist + list(hist_temp * weight[r][c])
    return local_hist

#the funtion to show the LBP of the sample
def show_lbp_sample(save_path):
    local_hist = []
    neighbors  = 24
    radius     = 3
    r          = 2
    c          = 4
    block_size = 20
    
    #load image
    gray       = cv2.imread(save_path,0)

    #get LBP histogram
    lbp = local_binary_pattern(gray, 
                             neighbors,
                             radius, 
                             method="uniform")
    
    #rescale the lbp into 0-255
    lbp        = lbp.astype("float") / 9 * 255
    lbp        = lbp.astype("uint8")
    block      = lbp[r * block_size : (r + 1) * block_size, c * block_size : (c + 1) * block_size]
    
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
    return

#SVM with rbf kernel
def train_svm_rbf(train_data, test_data, train_labels, test_labels, param_grid, n_classes):    
    
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
    y_pred    = clf.predict(test_data)
    print('test accuracy=', acc)
    print('training accuracy=', acc_train)
    print(confusion_matrix(test_labels, y_pred, labels = range(n_classes)))
    return clf

#SVM with linear kernel
def train_svm_linear(train_data, test_data, train_labels, test_labels, param_grid, n_classes):    
    
    clf        = GridSearchCV(LinearSVC(), param_grid)
    clf        = clf.fit(train_data, train_labels)

    print("done in %0.3fs" % (time.clock() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    acc       = np.mean(clf.best_estimator_.predict(test_data) == test_labels)
    acc_train = np.mean(clf.best_estimator_.predict(train_data) == train_labels)
    print('test accuracy=', acc)
    print('training accuracy=', acc_train)
    print(confusion_matrix(test_labels, y_pred, labels = range(n_classes)))
    return clf, acc

def PCA_process(train_data, test_data, train_labels, test_labels, n_components):
    n_components = 220
    t0           = time.clock()
    pca          = PCA(n_components = n_components, svd_solver = 'randomized', whiten = True).fit(train_data)
    
    #PCA transform of the training data and test data
    t0 = time.clock()
    lbp_train_pca = pca.transform(train_data)
    lbp_test_pca = pca.transform(test_data)
    print("done in %0.3fs" % (time.clock() - t0))
    return lbp_train_pca, lbp_test_pca, pca


train_data   = []
train_labels = []
test_data    = []
test_labels  = []    
sample       = 13
class_size   = 300
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
            local_hist = get_lbp(gray)
            
            if j % 9 == 0 or j % 9 == 1:
                test_labels.append(i)
                test_data.append(local_hist)
            else:
                train_labels.append(i)
                train_data.append(local_hist)
                
#show the sample of LBP jpg
save_path  = 'sample_face.jpg'
show_lbp_sample(save_path)

# start training the model
n_classes=11
#SVM with rbf kernel
param_grid = {'C': [0.1, 1, 5, 10, 20, 100, 1000, 10000],
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
clf, acc   = train_svm_rbf(train_data, 
                           test_data, 
                           train_labels, 
                           test_labels, 
                           param_grid, 
                           n_classes)

#SVM with linear kernel
param_grid = {'C': [0.1,0.5,1,5,10,100,1000,10000]}
clf, acc   = train_svm_linear(train_data, 
                              test_data, 
                              train_labels, 
                              test_labels, 
                              param_grid, 
                              n_classes)

#PCA of the LBP
n_components = 220
lbp_train_pca, lbp_test_pca, pca = PCA_process(train_data, 
                                          test_data, 
                                          train_labels, 
                                          test_labels, 
                                          n_components)

#SVM with RBF kernel after PCA
param_grid = {'C': [0.01, 0.1, 1, 7, 10, 30, 100, 1000],
              'gamma': [0.00001, 0.0001, 0.004, 0.0005, 0.006, 0.001, 0.01, 0.1, 1]}
clf, acc   = train_svm_rbf(lbp_train_pca, 
                           lbp_test_pca, 
                           train_labels, 
                           test_labels, 
                           param_grid, 
                           n_classes)

#save model to certain path
path    = 'C:/Users/zhang/Documents/DIP/project'
max_acc = 0
if(acc > max_acc):
    joblib.dump(clf, path+'model_friends_final.pkl')
    joblib.dump(pca, path+'pca_friends_final.pkl')
    max_acc = acc
    

#SVM with linear kernel after PCA
param_grid  = {'C': [i / 1000.0 for i in range(1,10)]}
clf, acc    = train_svm_linear(lbp_train_pca, 
                               lbp_test_pca, 
                               train_labels, 
                               test_labels, 
                               param_grid, 
                               n_classes)

#save model to certain path
path='C:/Users/zhang/Documents/DIP/project'
if(acc > max_acc):
    joblib.dump(clf, path+'model_friends_final2.pkl')
    max_acc = acc
    joblib.dump(pca,path+'pca_friends_final2.pkl')
