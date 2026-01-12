import pandas as pd 
data={
    "hours_studied" : [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
       "passed" : [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          1, 1, 1, 0, 1, 1, 1, 1, 0, 1]
}

df=pd.DataFrame(data)

x=df[["hours_studied"]]
y=df["passed"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.3,random_state=42
)

from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2)
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)

from sklearn.linear_model import LogisticRegression
result=LogisticRegression()
result.fit(x_train_poly,y_train)

prediction=result.predict(x_test_poly)
prob=result.predict_proba(x_test_poly)
print(prediction)
print(prob)

print(y_test.values)


from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,prediction)
ac=accuracy_score(y_test,prediction)

print("Cnfusion_matrix :",cm)
print("Accuracy_score",ac)


