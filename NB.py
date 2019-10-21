#(11/20 homework)使用 Naive Bayes 進行資料分析

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#載入數據集
avocado=pd.read_csv("avocado.csv")
x = pd.DataFrame(avocado,columns=["Total Volume","AveragePrice"])

#資料預處理
label_Encoder=preprocessing.LabelEncoder()
y=label_Encoder.fit_transform(avocado["type"])

#split our data
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33,random_state=0)

# Initialize our classifier
gnb=GaussianNB()

# Train our classifier
model = gnb.fit(train_x, train_y)
print("model:",model)

# Make predictions
preds = gnb.predict(test_x)
#預測值
print("預測值:\n",preds)

# make predictions
expected = y
predicted = model.predict(x)
# summarize the fit of the model
print("分類報告:\n",metrics.classification_report(expected, predicted))
print("混淆矩陣:\n",metrics.confusion_matrix(expected, predicted))

# Evaluate accuracy
print("準確率:",accuracy_score(test_y, preds))

