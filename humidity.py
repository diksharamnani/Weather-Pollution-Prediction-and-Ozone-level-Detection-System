#!/usr/bin/env python
# coding: utf-8

# In[55]:


# Importing necessary libraries to conduct our analysis
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
# Ignore harmless warnings
import warnings
warnings.filterwarnings("ignore")
from IPython.display import HTML,display

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('E:\TE_Project\datacsv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[56]:


#Reading the dataset into object 'df' using pandas:
df= pd.read_csv('E:/TE_Project/datacsv/India.csv',index_col='date_time',parse_dates=True)


# In[57]:


df.head(5)


# In[58]:


df.describe()


# In[59]:


df=df[['humidity']]


# In[60]:


df


# In[61]:



df=df.resample(rule='MS').mean()


# In[62]:


df.tail(7)


# In[63]:


df['humidity']==df.mean(axis=1)


# In[64]:


ax=df[['humidity']].plot(figsize=(20,10),grid=True,lw=2,color='Red')
ax.autoscale(enable=True, axis='both', tight=True)


# In[65]:


df_2019=df['2019-01-01':'2020-01-01']
df_2019.head()


# In[66]:


df_2019.isna().sum()


# In[67]:


humidity_2019=df_2019.mean(axis=0)


# In[68]:


plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
bplot = sns.boxplot( data=df_2019,  width=0.75,palette="GnBu_d")
plt.ylabel('humidity');
bplot.grid(True)


# In[69]:


from statsmodels.tsa.seasonal import seasonal_decompose
India_humidity=df['humidity']
result=seasonal_decompose(India_humidity,model='multiplicative')
result.plot();


# In[70]:


from matplotlib import dates
ax=result.seasonal.plot(xlim=['2018-01-01','2020-02-10'],figsize=(20,8),lw=2)
ax.yaxis.grid(True)
ax.xaxis.grid(True)


# In[71]:


#Formatting necessary to Prophet:
India_humidity=India_humidity.reset_index()
India_humidity.columns=['ds','y']
India_humidity=India_humidity.set_index('ds')


# In[72]:


train=India_humidity[:-24]
test=India_humidity[-24:-12]


# In[73]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)


# In[74]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[75]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[76]:


#To give an idea of what generator file holds:
X,y = generator[0]


# In[77]:


# We can see that the x array gives the list of values that we are going to predict y of:
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[78]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[79]:


# defining the model(note that  I am using a very basic model here, a 2 layer model only):
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()


# In[80]:


# Fitting the model with the generator object:
model.fit_generator(generator,epochs=250)


# In[81]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[82]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    
    current_pred = model.predict(current_batch)[0]
    
    
    test_predictions.append(current_pred) 
    
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[83]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[84]:


test['Predictions'] = true_predictions


# In[85]:


test.plot(figsize=(12,8))
plt.plot(true_predictions)


# In[86]:


import numpy as np
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(test['y'],test['Predictions']))
print('RMSE = ',RMSE)
print('India_humidity=',India_humidity['y'].mean())


# In[87]:


scaler.fit(India_humidity)
scaled_India_humidity=scaler.transform(India_humidity)


# In[88]:


generator = TimeseriesGenerator(scaled_India_humidity, scaled_India_humidity, length=n_input, batch_size=1)


# In[89]:


model.fit_generator(generator,epochs=250)


# In[90]:


test_predictions = []

first_eval_batch = scaled_India_humidity[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    
    current_pred = model.predict(current_batch)[0]
    
    
    test_predictions.append(current_pred) 
    
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[91]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[92]:


true_predictions=true_predictions.flatten()


# In[93]:


true_preds=pd.DataFrame(true_predictions,columns=['Forecast'])
true_preds=true_preds.set_index(pd.date_range('2020-08-01',periods=12,freq='MS'))


# In[94]:


true_preds


# In[95]:


plt.figure(figsize=(15,8))
plt.grid(True)
plt.plot( true_preds['Forecast'])
plt.plot( India_humidity['y'])

print(true_preds)  
pickle.dump(true_preds, open('humid.pkl','wb'))

# Loading model to compare the results
humid = pickle.load(open('humid.pkl','rb'))
# In[ ]:




