
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library  
import matplotlib.pyplot as plt

# veri setini okuyoruz
data = pd.read_csv('5k-continuous (1).csv')
#%%

col = data.columns       # veri setindeki sütunları yazdırıyoruz
print(col)

data["class"] = [1 if i.strip() == "spammer" else 0 for i in data['class']]

y = data['class']                         # spammer or non-spammer 
list1 = ['class']
x = data.drop(list1,axis = 1 )

#%%

ax = sns.countplot(y,label="Count")       # M = 212, B = 357
spammer, nonspammer = y.value_counts()
print('Number of spammer: ',spammer)
print('Number of non-spammer : ',nonspammer)

#%%

# corralation matrisi
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(x.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

#%%

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis, LocalOutlierFactor
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# veri setini test ve train olarak ayırıyoruz
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)


#%%

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accKNN = accuracy_score(y_test, y_pred)
score = knn.score(x_test, y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",accKNN)


cm_2 = confusion_matrix(y_test,y_pred)
sns.heatmap(cm_2,annot=True,fmt="d")

corr_matrix = data.corr()


cor_target = abs(corr_matrix["class"])
relevant_features = cor_target[cor_target>0.1]
relevant_features

to_drop = cor_target[cor_target<0.1]


row_names = to_drop.index
row_names_list = list(row_names) # Ayırt edici özellikler (feature selection)
row_names_list.append('class')
y = data['class'].values
X = data.drop(row_names_list, axis=1).values



#%%

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train,y_train)
y_pred=dtree.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accDT = accuracy_score(y_test, y_pred)
score = knn.score(x_test, y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic dtree Acc: ",accDT)

cm_2 = confusion_matrix(y_test,y_pred)
sns.heatmap(cm_2,annot=True,fmt="d")


#%%

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train,y_train)
y_pred=rfc.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
accRF = accuracy_score(y_test, y_pred)
score = knn.score(x_test, y_test)
print("Score: ",score)
print("CM: ",cm)
print("Basic rfc Acc: ",accRF)

cm_2 = confusion_matrix(y_test,y_pred)
sns.heatmap(cm_2,annot=True,fmt="d")

#%%

plt.title ("Kullanılan Yöntemlerin ACC değerlerinin Karşılaştırılması")
plt.xlabel("Kullanılan Yöntemler")
plt.ylabel("ACC Değerleri")

x_label='accKNN','accDT','accRF'
y_label= [accKNN,accDT,accRF]


plt.bar(x_label,y_label)
plt.legend()
plt.show()


#%% Aşağıda tespit ettiğimiz işimize yaramayan sütunları kaldırıyoruz.(feature selection)
data.drop('account_age',axis=1, inplace=True)
data.drop('no_follower',axis=1, inplace=True)
data.drop('no_following',axis=1, inplace=True)
data.drop('no_userfavourites',axis=1, inplace=True)
data.drop('no_lists',axis=1, inplace=True)
data.drop('no_retweets',axis=1, inplace=True)
data.drop('no_urls',axis=1, inplace=True)
data.drop('no_digits',axis=1, inplace=True)

#%% yeni data sütunlarımız
col2=data.columns

#%% train ve test için güncel x ve y değerlerimizi belirliyoruz.
y_yeni = data['class']                         # spammer or non-spammer 
list1 = ['class']
x_yeni = data.drop(list1,axis = 1 )

#%%
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_yeni, y_yeni, test_size=0.3, random_state=42)
#%% silinen sütunlardan sonra KNN algoritması

knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(x_train_new, y_train_new)
y_pred = knn.predict(x_test_new)
cm = confusion_matrix(y_test_new, y_pred)
accKNN = accuracy_score(y_test_new, y_pred)
score = knn.score(x_test_new, y_test_new)
print("Score: ",score)
print("CM: ",cm)
print("Basic KNN Acc: ",accKNN)

#%% silinen sütunlardan sonra DecisionTree algoritması

from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(x_train_new,y_train_new)
y_pred=dtree.predict(x_test_new)
cm = confusion_matrix(y_test_new, y_pred)
accDT = accuracy_score(y_test_new, y_pred)
score = knn.score(x_test_new, y_test_new)
print("Score: ",score)
print("CM: ",cm)
print("Basic dtree Acc: ",accDT)

#%% silinen sütunlardan sonra RandomForest algoritması

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
rfc.fit(x_train_new,y_train_new)
y_pred=rfc.predict(x_test_new)
cm = confusion_matrix(y_test_new, y_pred)
accRF = accuracy_score(y_test_new, y_pred)
score = knn.score(x_test_new, y_test_new)
print("Score: ",score)
print("CM: ",cm)
print("Basic rfc Acc: ",accRF)

#%%

plt.title ("Kullanılan Yöntemlerin ACC değerlerinin Karşılaştırılması")
plt.xlabel("Kullanılan Yöntemler")
plt.ylabel("ACC Değerleri")

x_label='accKNN','accDT','accRF'
y_label= [accKNN,accDT,accRF]


plt.bar(x_label,y_label)
plt.legend()
plt.show()


