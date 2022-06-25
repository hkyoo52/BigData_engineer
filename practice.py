from cmath import nan
import pandas as pd
from sklearn import datasets
import numpy as np

# 1번
'''
housing = datasets.load_boston()
X=pd.DataFrame(housing['data'],columns=housing['feature_names'])
y=pd.DataFrame(housing['target'],columns=['target'])
new_df = X.sort_values(by='CRIM',ascending=False).reset_index().iloc[:10]
new_df['CRIM']=new_df['CRIM'].iloc[-1]
new_df = new_df[['CRIM','AGE']]
a=np.mean(new_df['CRIM'] * (new_df['AGE']>=80))
print(new_df)
print(a)
'''

# 2번
'''
housing = datasets.load_boston()
X=pd.DataFrame(housing['data'],columns=housing['feature_names'])
for i in range(1,500,10):
    X['INDUS'].iloc[i] = nan
X['new_INDUS'] = X['INDUS'].fillna(X['INDUS'].median())
print(X['new_INDUS'].std())
print(X['INDUS'].std())
'''

# 3번
'''
housing = datasets.load_boston()
X=pd.DataFrame(housing['data'],columns=housing['feature_names'])
print(np.sum(X['INDUS'] * (X['INDUS'] > (X['INDUS'].mean() + 1.5 * X['INDUS'].std()))))
'''

# 2유형
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
housing = datasets.load_boston()
X=pd.DataFrame(housing['data'],columns=housing['feature_names'])
y=pd.DataFrame(housing['target'],columns=['target'])

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=52)
min_max = MinMaxScaler()
min_max.fit(X_train)
min_max_train = min_max.transform(X_train)
min_max_test = min_max.transform(X_test)
model = LinearRegression()
model.fit(min_max_train,y_train)
pred = model.predict(min_max_test)
print(pred)
