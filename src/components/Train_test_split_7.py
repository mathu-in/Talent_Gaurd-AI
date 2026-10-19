import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(X , y , test_size=0.2, random_state=42):

    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state, stratify=y)
    #since we have imbalanced dataset during split stratify = y is given
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_test)
    print("7. Train-Test split is done")
    return x_train,x_test,y_train,y_test #mention or not it returns tuple.


if __name__ == '__main__':
    split_data()