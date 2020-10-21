import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from matplotlib import pyplot as plt




def load_data(f1, f2):
    '''
    get two input file f1, f2, divide the entire data in to three parts, training set, validation set, testing set
    '''
    df1 = pd.read_table(f1, header=None)
    df1['label'] = 0
    df2 = pd.read_table(f2, header=None)
    df2['label'] = 1
    # concat two file in one dataframe
    df = pd.concat([df1, df2])
    df.columns = ['title', 'label']
    df_x = df['title']
    df_y = df['label']
    # change our data into array form for faster calculation
    cv = CountVectorizer()
    X = np.array(df_x)
    Y = np.array(df_y)
    # fit our title data with CountVectorizer for future vectorize and prevent from peeking into our
    # validation set and train set
    cv.fit(X)
    # First divide into three data set
    X_train, X_t, Y_train, Y_t = train_test_split(X, Y, test_size=0.3)
    X_velidate, X_test, Y_velidate, Y_test = train_test_split(X_t, Y_t, test_size=0.5)
    # Then vectorize title information
    X_train = cv.transform(X_train)
    X_test = cv.transform(X_test)
    X_velidate = cv.transform(X_velidate)
    return X_train, X_velidate, X_test, Y_train, Y_velidate, Y_test

def select_knn_model(f1, f2):
    l1 = []
    l2 = []
    open(f1)
    open(f2)
    # create our training set, validation set, testing set
    X_train, X_velidate, X_test, Y_train, Y_velidate, Y_test = load_data(f1, f2)
    for i in range(1, 21):
        KN = KNeighborsClassifier(n_neighbors=i,metric='cosine')
        KN.fit(X_train, Y_train)
        # print out validation accuracy
        y_expect = Y_velidate
        y_pred = KN.predict(X_velidate)
        print("=" * 60)
        print("number of neighbors = ", i)
        res = metrics.accuracy_score(y_expect,y_pred)
        print('validation accuracy = ',res)
        l1.append(res)
        # print out training accuracy
        y_expect = Y_train
        y_pred = KN.predict(X_train)
        res2 = metrics.accuracy_score(y_expect, y_pred)
        print('training accuracy = ', res2)
        l2.append(res2)
    plt.xticks(range(1,21))
    plt.yticks([i/10 for i in range(0,11)])
    plt.plot(range(1,21),l1,label='validation accuracy')
    plt.plot(range(1, 21), l2,label='training accuracy')
    plt.legend()
    plt.show()
    # get the highest accuracy value k to test our testing set
    most_accurate_k = l1.index(max(l1))+1
    KN = KNeighborsClassifier(n_neighbors=most_accurate_k,metric='cosine')
    KN.fit(X_train, Y_train)
    y_expect = Y_test
    y_pred = KN.predict(X_test)
    print()
    print('most accurate k = ',most_accurate_k)
    print('testing accuracy = ',metrics.accuracy_score(y_expect,y_pred))


f1 = 'data/clean_fake.txt'
f2 = 'data/clean_real.txt'
select_knn_model(f1,f2)

# t = ['cat', 'bulldozer', 'cat cat cat']
# cv = CountVectorizer()
# X = cv.fit_transform(t)
# X = X.toarray()
# print(X)
