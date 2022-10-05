import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.neighbors import KNeighborsClassifier

class NaiveBayes:

    def __init__(self,train_data,test_data):
       self.x1, self.y1 = train_data[:,:2], train_data[:,-1].astype('int')
       self.x2, self.y2 = test_data[:,:2], test_data[:,-1].astype('int')

    def plot_data(self):
        #x1,y1 is training data & labels
        #x2,y2 is test data & labels
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(1,2,1)
        ax1.scatter(self.x1[:,0], self.x1[:,1], s = 5, c =self.y1, cmap= plt.cm.rainbow,label = self.y1)
        ax1.set_title('Training data')
        ax1.set_xlabel('$x_{1}$')
        ax1.set_ylabel('$x_{2}$')
        ax1.set_label('0')

        ax2= fig.add_subplot(1,2,2)
        ax2.scatter(self.x2[:,0], self.x2[:,1], s = 5, c =self.y2, cmap= plt.cm.rainbow)
        ax2.set_title('Test data')
        ax2.set_xlabel('$x_{1}$')
        ax2.set_ylabel('$x_{2}$')

        plt.show()

    def log_gaussian_dist(self,x,mean,std):
        output = np.log((1/np.sqrt(2*math.pi)*std) * np.exp(-0.5 * ((x-mean)/std)**2))
        return output 

    def naive_bayes(self,x,y):
        posterior = []
        labels = np.unique(y)
        for y_ in labels:
            x_indices = np.where(y==y_)[0]
            prior = (x_indices.size/y.size)
            x_ = x[x_indices]
            mean_ = np.mean(x_,axis=0)
            std_ = np.std(x_,axis = 0)
            dist_ = self.log_gaussian_dist(x,mean_,std_)
            loglikelihood = np.sum(dist_,axis=-1)
            posterior.append(loglikelihood + np.log(prior))
        
        return posterior

    def prediction(self,posterior):

        #posterior[0] -> label 0 , posterior[1] -> 1
        pred = (posterior[0] < [posterior[1]]).astype(int)
        return pred

    def error(self,pred,gt):

        num_sample = gt.shape[0]
        correct = np.count_nonzero(pred==gt) 
        
        return correct/num_sample

    def __call__(self):

        #plot data
        self.plot_data()

        posterior_train = self.naive_bayes(self.x1,self.y1)
        pred_train = self.prediction(posterior_train)
        posterior_test = self.naive_bayes(self.x2,self.y2)
        pred_test = self.prediction(posterior_test)

        # Compute the error
        train_accuracy= self.error(pred_train,self.y1)
        test_accuracy = self.error(pred_test,self.y2)

        print("Train accuracy : {} , Test accuracy : {}".format(train_accuracy,test_accuracy))


class LinearRegression:

    def __init__(self,train_data,test_data):
        self.x1, self.y1 = train_data[:,:2], train_data[:,-1].astype('int')
        self.x2, self.y2 = test_data[:,:2], test_data[:,-1].astype('int')
        self.system = None
    
    def determine_sys(self,x):

        m, d = x.shape[0], x.shape[1]

        if m > d:
            self.system = "Over_det"
        elif m < d:
            self.system = "Under_det"
        else:
            self.system = "Even_det"
        
    
    def lr_params(self,x,y):

        self.determine_sys(x)

        if self.system == "Even_det":
            x_inv = np.linalg.inv(x)
            w_hat = np.matmul(x_inv,y)
        elif self.system == "Under_det":
            in_ = np.linalg.inv(np.matmul(x,x.T)) #(XX^T)^-1
            w_hat = x.T @ in_ @ y
        elif self.system == "Over_det":
            rank_x = np.linalg.matrix_rank(x)
            rank_xy = np.linalg.matrix_rank(np.concatenate((x,np.expand_dims(y,axis=-1)),axis=-1))
            in_ = np.linalg.inv(np.matmul(x.T,x)) #(X^TX)^-1
            w_hat = in_ @ x.T @ y
        
        return w_hat
    
    def plot_decision_boundary(self,w_hat):
        xx1, yy1 = np.meshgrid(np.arange(self.x1[:, 0].min()-0.2,self.x1[:, 0].max()+0.2,0.1), 
                            np.arange(self.x1[:, 1].min()-0.2,self.x1[:, 1].max()+0.2,0.1))
        grid1 = np.c_[xx1.ravel(), yy1.ravel()]
        probs1 = np.matmul(grid1,w_hat).reshape(xx1.shape)
        probs1 = (probs1 > 0.5).astype(int)

        xx2, yy2 = np.meshgrid(np.arange(self.x2[:, 0].min()-0.2,self.x2[:, 0].max()+0.2,0.1), 
                            np.arange(self.x2[:, 1].min()-0.2,self.x2[:, 1].max()+0.2,0.1))

        grid2 = np.c_[xx2.ravel(), yy2.ravel()]
        probs2 = np.matmul(grid2,w_hat).reshape(xx2.shape)
        probs2 = (probs2 > 0.5).astype(int)

        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(1,2,1)
        ax1.pcolormesh(xx1,yy1,probs1,cmap=plt.cm.rainbow)
        ax1.scatter(self.x1[:,0], self.x1[:,1], s = 5, c =self.y1, cmap= plt.cm.rainbow, edgecolors='k')
        ax1.set_title('Training data')
        ax1.set_xlabel('$x_{1}$')
        ax1.set_ylabel('$x_{2}$')

        ax2= fig.add_subplot(1,2,2)
        ax2.pcolormesh(xx2,yy2,probs2,cmap=plt.cm.rainbow)
        ax2.scatter(self.x2[:,0], self.x2[:,1], s = 5, c =self.y2, cmap= plt.cm.rainbow, edgecolors='k')
        ax2.set_title('Test data')
        ax2.set_xlabel('$x_{1}$')
        ax2.set_ylabel('$x_{2}$')

        plt.show()
    
    def __call__(self):

        # Compute linear regression params
        w_hat = self.lr_params(self.x1,self.y1)

        pred_train = np.matmul(self.x1,w_hat)
        acc_mask = (pred_train > 0.5).astype(int)
        train_acc = np.count_nonzero(acc_mask==self.y1) / len(self.y1) * 100

        pred_test = np.matmul(self.x2,w_hat)
        acc_mask = (pred_test > 0.5).astype(int)
        test_acc = np.count_nonzero(acc_mask==self.y2) / len(self.y2) * 100

        print("Train accuracy : {} , Test accuracy : {}".format(train_acc,test_acc))

        self.plot_decision_boundary(w_hat)

class KNNClassifier:

    def __init__(self,train_data,test_data,k_values):
        self.x1, self.y1 = train_data[:,:2], train_data[:,-1].astype('int')
        self.x2, self.y2 = test_data[:,:2], test_data[:,-1].astype('int')
        self.k = k_values
        self.weights = ['uniform','distance']
    
    def error(self,pred,gt):
        num_sample = gt.shape[0]
        correct = np.count_nonzero(pred==gt)
        
        return 1 - correct/num_sample
    
    def plot(self,error1,error2,error3,error4):

        fig = plt.figure(figsize=(5,5))
        x_values = np.array(self.k)
        plt.plot(x_values,error1,'ro--',label='Train Error (w:uniform)')
        plt.plot(x_values,error2,'bo--',label='Test Error (w:uniform)')
        plt.plot(x_values,error3,'go--',label='Train Error (w:distance)')
        plt.plot(x_values,error4,'co--',label='Test Error (w:distance)')
        plt.title("KNN Error")
        plt.xlabel("k-values")
        plt.ylabel("Error")
        plt.xticks(self.k)
        plt.grid()
        plt.legend()
        plt.show()

    
    def __call__(self):

        errors_train = []
        errors_test = []

        for w in self.weights:
            for k in self.k:
                knn_classifier = KNeighborsClassifier(n_neighbors=k,weights=w)
                knn_classifier.fit(self.x1,self.y1) #Train
                pred_train = knn_classifier.predict(self.x1)
                error_train = self.error(pred_train,self.y1)
                errors_train.append(error_train)

                pred_test = knn_classifier.predict(self.x2)
                error_test = self.error(pred_test,self.y2)
                errors_test.append(error_test)
        
        error_train_uniform, error_train_distance = errors_train[:5] , errors_train[5:]
        error_test_uniform, error_test_distance = errors_test[:5] , errors_test[5:]
        self.plot(error_train_uniform,error_test_uniform,error_train_distance,error_test_distance)


if __name__ == '__main__':

    train_data = np.loadtxt('train.txt')
    test_data = np.loadtxt('test.txt')

    naive_bayse = NaiveBayes(train_data,test_data)
    naive_bayse()
    linear_regression_ = LinearRegression(train_data,test_data)
    linear_regression_()
    k_values = [1,5,10,15,20]
    knn_classifier = KNNClassifier(train_data,test_data,k_values)
    knn_classifier()
    


