import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz


def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:,:2], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# dot -Tpng id3_tree.dot -o tree.png
tree_pic = export_graphviz(clf, out_file="id3_tree.dot")