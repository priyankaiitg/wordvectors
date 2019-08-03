import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import re
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from keras import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import to_categorical

embeddings_index = dict()
f = open('glove.6B.300d.txt')
for line in f:
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()


def post_embeddings(post):
    prov2= re.sub(r'[“€â.|,?!)(1234567890:/-]', '', post)
    prov3 = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', prov2)
    prov4 = re.sub(r'[|||)(?.,:1234567890!]',' ',prov3)
    prov5 = re.sub(' +',' ', prov4)
    prov6 = prov5.split(" ")
    prov7 = Counter(prov6)
    temp_prov = []
    for w in prov7:
        if(w in embeddings_index):
            temp_prov.append(embeddings_index[w] * prov7[w])
    if temp_prov:
        return np.array(np.mean(temp_prov, axis=0))
    else:
        return np.zeros(300)
    #return np.array[np.mean([embeddings_index[w] * prov7[w] for w in prov7 if w in embeddings_index] or [np.zeros(300)], axis=0)]

df = pd.read_csv('mbti/mbti_1.csv')


df['embeddings'] = df['posts'].apply(post_embeddings)
#print(df['embeddings'].head(10))

df[['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41','42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61','62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81','82','83','84','85','86','87','88','89','90','91','92','93','94','95','96','97','98','99','100','101','102','103','104','105','106','107','108','109','110','111','112','113','114','115','116','117','118','119','120','121','122','123','124','125','126','127','128','129','130','131','132','133','134','135','136','137','138','139','140','141','142','143','144','145','146','147','148','149','150','151','152','153','154','155','156','157','158','159','160','161','162','163','164','165','166','167','168','169','170','171','172','173','174','175','176','177','178','179','180','181','182','183','184','185','186','187','188','189','190','191','192','193','194','195','196','197','198','199','200','201','202','203','204','205','206','207','208','209','210','211','212','213','214','215','216','217','218','219','220','221','222','223','224','225','226','227','228','229','230','231','232','233','234','235','236','237','238','239','240','241','242','243','244','245','246','247','248','249','250','251','252','253','254','255','256','257','258','259','260','261','262','263','264','265','266','267','268','269','270','271','272','273','274','275','276','277','278','279','280','281','282','283','284','285','286','287','288','289','290','291','292','293','294','295','296','297','298','299','300']]=pd.DataFrame(df.embeddings.values.tolist(), index= df.index)


df['words_per_comment'] = df['posts'].apply(lambda x: len(x.split())/50)
df['http_per_comment'] = df['posts'].apply(lambda x: x.count('http')/50)
df['music_per_comment'] = df['posts'].apply(lambda x: x.count('music')/50)
df['question_per_comment'] = df['posts'].apply(lambda x: x.count('?')/50)
df['img_per_comment'] = df['posts'].apply(lambda x: x.count('jpg')/50)
df['excl_per_comment'] = df['posts'].apply(lambda x: x.count('!')/50)
df['ellipsis_per_comment'] = df['posts'].apply(lambda x: x.count('...')/50)



map1 = {"I": 0, "E": 1}
map2 = {"N": 0, "S": 1}
map3 = {"T": 0, "F": 1}
map4 = {"J": 0, "P": 1}
df['I-E'] = df['type'].astype(str).str[0]
df['I-E'] = df['I-E'].map(map1)
df['N-S'] = df['type'].astype(str).str[1]
df['N-S'] = df['N-S'].map(map2)
df['T-F'] = df['type'].astype(str).str[2]
df['T-F'] = df['T-F'].map(map3)
df['J-P'] = df['type'].astype(str).str[3]
df['J-P'] = df['J-P'].map(map4)

X = df.drop(['type','posts','I-E','N-S','T-F','J-P','embeddings'],axis=1).values
labels = df['type'].values
le = preprocessing.LabelEncoder()
le.fit(['INFJ','ENTP','INTP','INTJ','ENTJ','ENFJ','INFP','ENFP','ISFP','ISTP','ISFJ','ISTJ','ESTP','ESFP','ESTJ','ESFJ'])
y = le.transform(labels)
y = to_categorical(y)
#print (X.shape)
#print (y.shape)
#print(y)


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=5)


mbtimodel = Sequential()
mbtimodel.add(Dense(40, input_dim=307, activation='relu'))
mbtimodel.add(Dense(16, activation='softmax'))
mbtimodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#mbtimodel.summary()
mbtimodel.fit(X_train,y_train, batch_size=10, epochs=100)

Y_prediction = mbtimodel.predict(X_test)
Y_prediction=np.argmax(Y_prediction, axis=1) 
y_test = np.argmax(y_test, axis=1)

#random_forest = RandomForestClassifier(n_estimators=100)

#random_forest.fit(X_train,y_train)

#Y_prediction = random_forest.predict(X_test)

#random_forest.score(X_train,y_train)

#acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
#print(round(acc_random_forest,2,), "%")

ascore = accuracy_score(y_test, Y_prediction)
print(ascore)


