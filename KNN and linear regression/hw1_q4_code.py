import numpy as np
from matplotlib import pyplot as plt

data_train = {'X':np.genfromtxt('data/data_train_X.csv',delimiter=','),'t':np.genfromtxt('data/data_train_y.csv',delimiter=',')}
data_test = {'X':np.genfromtxt('data/data_test_X.csv',delimiter=','),'t':np.genfromtxt('data/data_test_y.csv',delimiter=',')}

def shuffle_data(data):
    l1 = data['X']
    l2 = data['t']
    l2=np.array(l2).reshape(len(l2),1)
    tmp = np.concatenate([l2,l1],1)
    np.random.shuffle(tmp)
    l2, l1 = np.split(tmp,[1],1)
    return {'X':l1,'t':l2}

def split_data(data, num_folds, fold):
    num_row = data['X'].shape[0]
    start = (fold-1)*(num_row//num_folds)
    end = fold*(num_row//num_folds)
    data_fold = {'X':data['X'][start:end],'t':data['t'][start:end]}
    tmp1 = [data['X'][:start],data['t'][:start]]
    tmp2 = [data['X'][end:],data['t'][end:]]
    data_rest = {'X':np.concatenate([tmp1[0],tmp2[0]]),'t':np.concatenate([tmp1[1],tmp2[1]])}
    return data_fold, data_rest


def train_model(data, lambd):
    X = data['X']
    X_transpose = np.transpose(X)
    D = X.shape[0]
    I = np.identity(X.shape[1])
    return np.dot(np.linalg.inv((np.dot(X_transpose,X)+np.dot(lambd*D,I))),np.dot(X_transpose,data['t']))

def predict(data, model):
    return np.dot(data['X'],model)

def loss(data, model):
    res = predict(data, model)
    i = res - data['t']
    N = res.shape[0]
    res = np.dot(np.transpose(i),i)/(2*N)
    return res[0,0]

def cross_validation(data, num_folds, lambd_seq):
    cv_error = []
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(1,num_folds+1):
            val_cv, train_cv = split_data(data,num_folds,fold)
            model = train_model(train_cv,lambd)
            cv_loss_lmd += loss(val_cv,model)
        cv_error.append(cv_loss_lmd/num_folds)
    return cv_error

def loss_for_each_lambd_4c(training_data,test_data,lambd_seq):
    training_error = []
    test_error = []
    training_data = shuffle_data(training_data)
    test_data = shuffle_data(test_data)
    for i in lambd_seq:
        model = train_model(training_data,i)
        training_error.append(loss(training_data,model))
        test_error.append(loss(test_data,model))
    return training_error, test_error






lmd_seq = []
for i in range(50):
    lmd_seq.append(0.00005+(0.005-0.00005)*i/50)
l3 = cross_validation(data_test,5,lmd_seq)
l4 = cross_validation(data_test,10,lmd_seq)
plt.figure(figsize=(12, 6))
l1,l2 = loss_for_each_lambd_4c(data_train,data_test,lmd_seq)
plt.plot(range(1,51),l1,label='training error')
plt.plot(range(1, 51), l2,label='testing error')
plt.plot(range(1,51),l3,label='cross validation with folds number 5')
plt.plot(range(1, 51), l4,label='cross validation with folds number 10')
plt.legend()
plt.xticks(range(1,51))
plt.show()
