from keras.models import Sequential
from keras.layers import LSTM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[61]:


Data = [[[(i+j)/100] for i in range(5)] for j in range(100)]
target = [(i+5)/100 for i in range(100)]


# In[62]:


Data = np.array(Data , dtype = float)
target = np.array(target , dtype = float)


# In[63]:


Data.shape


# In[64]:


target.shape


# In[65]:


x_train, x_test, y_train, y_test = train_test_split(Data, target, test_size = 0.2, random_state = 4)


# In[88]:


model = Sequential()


# In[89]:


model.add(LSTM((2), batch_input_shape = (None, 5, 1), return_sequences = True))
model.add(LSTM((1), return_sequences = False))


# In[90]:


model.compile(optimizer= 'adam', metrics= ['accuracy'], loss= 'mean_absolute_error')


# In[91]:


model.summary()


# In[92]:


history = model.fit(x_train, y_train, epochs=300, validation_data=(x_test, y_test))


# In[93]:


result = model.predict(x_test)


# In[94]:


plt.scatter(range(20), result, c='r')
plt.scatter(range(20), y_test, c='g')
plt.show()


# In[95]:


plt.plot(history.history['loss'])
plt.show()


# In[ ]:




