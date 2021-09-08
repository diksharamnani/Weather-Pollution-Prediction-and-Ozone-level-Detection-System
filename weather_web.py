#!/usr/bin/env python
# coding: utf-8

# In[138]:


# Importing necessary libraries to conduct our analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
# Ignore harmless warnings
import warnings
import pickle
warnings.filterwarnings("ignore")
from IPython.display import HTML,display

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('E:\TE_Project\datacsv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[139]:


#Reading the dataset into object 'df' using pandas:
df= pd.read_csv('E:/TE_Project/datacsv/India.csv',index_col='date_time',parse_dates=True)


# In[140]:


df.head(5)


# In[141]:


df.describe()


# In[142]:


df=df[['tempC']]


# In[143]:


df


# In[144]:



df=df.resample(rule='MS').mean()


# In[145]:


df.tail(7)


# In[146]:


df['tempC']==df.mean(axis=1)


# In[147]:


ax=df[['tempC']].plot(figsize=(20,10),grid=True,lw=2,color='Red')
ax.autoscale(enable=True, axis='both', tight=True)


# In[148]:


df_2019=df['2019-01-01':'2020-01-01']
df_2019.head()


# In[149]:


df_2019.isna().sum()


# In[150]:


temp_2019=df_2019.mean(axis=0)


# In[151]:


plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
bplot = sns.boxplot( data=df_2019,  width=0.75,palette="GnBu_d")
plt.ylabel('temp');
bplot.grid(True)


# In[152]:


from statsmodels.tsa.seasonal import seasonal_decompose
India_temp=df['tempC']
result=seasonal_decompose(India_temp,model='multiplicative')
result.plot();


# In[153]:


from matplotlib import dates
ax=result.seasonal.plot(xlim=['2018-01-01','2020-02-10'],figsize=(20,8),lw=2)
ax.yaxis.grid(True)
ax.xaxis.grid(True)


# In[154]:


#Formatting necessary to Prophet:
India_temp=India_temp.reset_index()
India_temp.columns=['ds','y']
India_temp=India_temp.set_index('ds')


# In[155]:


train=India_temp[:-24]
test=India_temp[-24:-12]


# In[156]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)


# In[157]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[158]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[159]:


#To give an idea of what generator file holds:
X,y = generator[0]


# In[160]:


# We can see that the x array gives the list of values that we are going to predict y of:
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[161]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[162]:


# defining the model(note that  I am using a very basic model here, a 2 layer model only):
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()


# In[163]:


# Fitting the model with the generator object:
model.fit_generator(generator,epochs=250)


# In[164]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[165]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    
    current_pred = model.predict(current_batch)[0]
    
    
    test_predictions.append(current_pred) 
    
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[166]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[167]:


test['Predictions'] = true_predictions


# In[168]:


test.plot(figsize=(12,8))
plt.plot(true_predictions)


# In[169]:


import numpy as np
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(test['y'],test['Predictions']))
print('RMSE = ',RMSE)
print('India_temp=',India_temp['y'].mean())


# In[170]:


scaler.fit(India_temp)
scaled_India_temp=scaler.transform(India_temp)


# In[171]:


generator = TimeseriesGenerator(scaled_India_temp, scaled_India_temp, length=n_input, batch_size=1)


# In[172]:


model.fit_generator(generator,epochs=250)
test_predictions = []

first_eval_batch = scaled_India_temp[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    
    current_pred = model.predict(current_batch)[0]
    
    
    test_predictions.append(current_pred) 
    
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

true_predictions = scaler.inverse_transform(test_predictions)
true_predictions=true_predictions.flatten()
true_preds=pd.DataFrame(true_predictions,columns=['Forecast'])
true_preds=true_preds.set_index(pd.date_range('2020-08-01',periods=12,freq='MS')) 
print(true_preds)  
pickle.dump(true_preds, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))

# %%
