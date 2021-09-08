#!/usr/bin/env python
# coding: utf-8

# In[27]:


import pandas as pd
df=pd.read_csv('E:\\TE_Project\\datacsv\\India.csv',index_col='date_time',parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()
df


# In[28]:


df=df[['tempC']]
df


# In[29]:


df['tempC'].plot(figsize=(12,5))


# In[30]:


train=df[:-24]
test=df[-24:-12]


# In[31]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train)


# In[32]:


scaled_train = scaler.transform(train)
scaled_test = scaler.transform(test)


# In[33]:


from keras.preprocessing.sequence import TimeseriesGenerator
n_input = 24
n_features = 1
generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)


# In[34]:


#To give an idea of what generator file holds:
X,y = generator[0]


# In[35]:


# We can see that the x array gives the list of values that we are going to predict y of:
print(f'Given the Array: \n{X.flatten()}')
print(f'Predict this y: \n {y}')


# In[36]:


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


# In[37]:


# defining the model(note that  I am using a very basic model here, a 2 layer model only):
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.summary()


# In[ ]:


# Fitting the model with the generator object:
model.fit_generator(generator,epochs=25)


# In[26]:


import matplotlib.pyplot as plt
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)),loss_per_epoch)


# In[ ]:




