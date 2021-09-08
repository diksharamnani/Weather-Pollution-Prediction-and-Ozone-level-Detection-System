#!/usr/bin/env python
# coding: utf-8

# In[73]:


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

for dirname, _, filenames in os.walk('E:\TE_Project\Pollution'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[74]:


#Reading the dataset into object 'df' using pandas:
df= pd.read_csv('E:/TE_Project/Pollution/city_day.csv',parse_dates=True)
df['Date'] = pd.to_datetime(df['Date'])


# In[75]:


df.head(5)


# In[76]:


df.describe()


# In[77]:


df=df[['City','Date','AQI','AQI_Bucket']]


# In[78]:


cities=pd.unique(df['City'])
column1= cities+'_AQI'
column2=cities+'_AQI_Bucket'
columns=[*column1,*column2]
len(column1)


# In[79]:


final_df=pd.DataFrame(index=np.arange('2015-01-01','2020-07-02',dtype='datetime64[D]'),columns=column1)
print(final_df.shape)


# In[80]:


arr=dict()
for i in range(len(cities)):
    arr[cities[i]] = 0
    

for i in range(len(cities)):
    for j in range(29531):
        if(cities[i]==df['City'][j]):
            arr[cities[i]]+=1
            
            
print(arr)


# In[81]:


for city,i in zip(cities,final_df.columns):
    n=len(np.array(df[df['City']==city]['AQI']))
#     print(n)
    final_df[i][-n:]=np.array(df[df['City']==city]['AQI'])


# In[82]:


final_df=final_df.astype('float64')
final_df=final_df.resample(rule='MS').mean()


# In[83]:


final_df.tail(7)


# In[84]:


final_df.head()


# In[85]:


final_df=final_df.astype('float64')
final_df=final_df.resample(rule='MS').mean()


# In[86]:


final_df.tail(7)


# In[87]:


final_df['India_AQI']=final_df.mean(axis=1)


# In[88]:


ax=final_df[['India_AQI']].plot(figsize=(12,8),grid=True,lw=2,color='Red')
ax.autoscale(enable=True, axis='both', tight=True)


# In[89]:


df_2019=final_df['2019-01-01':'2020-01-01']
df_2019.head()


# In[90]:


df_2019.isna().sum()


# In[91]:


df_2019=df_2019.drop(['Aizawl_AQI','Ernakulam_AQI','Kochi_AQI'],axis=1)


# In[92]:


AQI_2019=df_2019.mean(axis=0)


# In[93]:


plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
bplot = sns.boxplot( data=df_2019,  width=0.75,palette="GnBu_d")
plt.ylabel('AQI');
bplot.grid(True)


# In[94]:


plt.figure(figsize=(20,8))
plt.xticks(rotation=90)
plt.ylabel('AQI')
bplot=sns.barplot(AQI_2019.index, AQI_2019.values,palette="GnBu_d")


# In[95]:


from statsmodels.tsa.seasonal import seasonal_decompose
India_AQI=final_df['India_AQI']
result=seasonal_decompose(India_AQI,model='multiplicative')
result.plot();


# In[96]:


from matplotlib import dates
ax=result.seasonal.plot(xlim=['2018-01-01','2020-02-10'],figsize=(20,8),lw=2)
ax.yaxis.grid(True)
ax.xaxis.grid(True)


# In[97]:


# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima;                              # for determining ARIMA orders


# In[98]:


auto_arima(y=India_AQI,start_p=0,start_P=0,start_q=0,start_Q=0,seasonal=True, m=12).summary()


# In[99]:


len(India_AQI)


# In[100]:


#dividing into train and test:
train=India_AQI[:41]
test=India_AQI[42:54]


# In[101]:


# Forming the model:
model=SARIMAX(train,order=(0,1,2),seasonal_order=(1,0,1,12),)
results=model.fit()
results.summary()


# In[102]:


#Obtaining predicted values:
predictions = results.predict(start=42, end=53, typ='levels').rename('Predictions')


# In[103]:


#Plotting predicted values against the true values:
predictions.plot(legend=True)
test.plot(legend=True);


# In[104]:


from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(predictions,test))
print('RMSE = ',RMSE)
print('Mean AQI',test.mean())


# In[105]:


#dividing into train and test:
train=India_AQI[:53]
test=India_AQI[54:]
# Forming the model:
model=SARIMAX(train,order=(0,1,2),seasonal_order=(1,0,1,12),)
results=model.fit()
results.summary()
#Obtaining predicted values:
predictions = results.predict(start=54, end=66, typ='levels').rename('Predictions')
#Plotting predicted values against the true values:
predictions.plot(legend=True)
test.plot(legend=True);


# In[106]:


#Finding RMSE:
from sklearn.metrics import mean_squared_error
RMSE=np.sqrt(mean_squared_error(predictions,test))
print('RMSE = ',RMSE)
print('Mean AQI',test.mean())


# In[107]:


# Forming the model:
model=SARIMAX(India_AQI,order=(0,1,2),seasonal_order=(1,0,1,12))
results=model.fit()
results.summary()
#Obtaining predicted values:
predictions = results.predict(start=67, end=100, typ='levels').rename('Predictions')
#Plotting predicted values against the true values:
predictions.plot(legend=True)
India_AQI.plot(legend=True,figsize=(12,8),grid=True);


# In[43]:


#Formatting necessary to Prophet:
India_AQI=India_AQI.reset_index()
India_AQI.columns=['ds','y']
India_AQI=India_AQI.set_index('ds')


# In[44]:


train=India_AQI[:-24]
test=India_AQI[-24:-12]


# In[45]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)


# In[46]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[47]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[48]:


#To give an idea of what generator file holds:
X,y = generator[0]


# In[49]:


# We can see that the x array gives the list of values that we are going to predict y of:
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[50]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[51]:


# defining the model(note that  I am using a very basic model here, a 2 layer model only):
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()


# In[52]:


# Fitting the model with the generator object:
model.fit_generator(generator,epochs=250)


# In[54]:


loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[55]:


test_predictions = []

first_eval_batch = scaled_train[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    
    current_pred = model.predict(current_batch)[0]
    
    
    test_predictions.append(current_pred) 
    
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[56]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[57]:


test['Predictions'] = true_predictions


# In[58]:


test.plot(figsize=(12,8))
plt.plot(true_predictions)


# In[64]:


RMSE=np.sqrt(mean_squared_error(test['y'],test['Predictions']))
print('RMSE = ',RMSE)
print('India_AQI=',India_AQI['y'].mean())


# In[61]:


scaler.fit(India_AQI)
scaled_India_AQI=scaler.transform(India_AQI)


# In[62]:


generator = TimeseriesGenerator(scaled_India_AQI, scaled_India_AQI, length=n_input, batch_size=1)


# In[63]:


model.fit_generator(generator,epochs=250)


# In[65]:


test_predictions = []

first_eval_batch = scaled_India_AQI[-n_input:]
current_batch = first_eval_batch.reshape((1, n_input, n_features))

for i in range(len(test)):
    
    
    current_pred = model.predict(current_batch)[0]
    
    
    test_predictions.append(current_pred) 
    
    
    current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)


# In[66]:


true_predictions = scaler.inverse_transform(test_predictions)


# In[67]:


true_predictions=true_predictions.flatten()


# In[68]:


true_preds=pd.DataFrame(true_predictions,columns=['Forecast'])
true_preds=true_preds.set_index(pd.date_range('2020-08-01',periods=12,freq='MS'))


# In[69]:


true_preds
print(true_preds)  
pickle.dump(true_preds, open('AQI.pkl','wb'))

# Loading model to compare the results
AQI = pickle.load(open('AQI.pkl','rb'))


# In[ ]:





# In[ ]:




