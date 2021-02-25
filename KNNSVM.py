import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv()


x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_tr ,x_te, y_tr, y_te = train_test_split(x, y, test_size = 0.20, random_state = 0)


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p=2)
classifier.fit(x_tr,y_tr)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'linear', random_state = 0)
#classifier.fit(x_tr,y_tr)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'poly', random_state = 0)
#classifier.fit(x_tr,y_tr)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'rbf', random_state = 0)
#classifier.fit(x_tr,y_tr)

#from sklearn.svm import SVC
#classifier = SVC(kernel = 'sigmoid', random_state = 0)
#classifier.fit(x_tr,y_tr)

from sklearn.metrics import accuracy_score,confusion_matrix
y_pred = classifier.predict(x_te)
acrscr = accuracy_score(y_te,y_pred)
confmtx = confusion_matrix(y_te,y_pred)
print(acrscr)

from mlxtend.plotting import plot_confusion_matrix
plot_confusion_matrix(confmtx)
plt.show()
