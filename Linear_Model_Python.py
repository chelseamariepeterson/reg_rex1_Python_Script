#!/usr/bin/env python
# coding: utf-8

# In[19]:

#**import dependencies**
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance
import sys
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[20]:

#bring in data
dataset = pd.read_csv(sys.argv[1])


# In[65]:

# fun stuff
dataset.shape


# In[66]:


dataset.describe()


# In[67]:

#**Plot 1**
dataset.plot(x='x', y='y', style='o')  
plt.title('Linear Model')  
plt.xlabel('X')  
plt.ylabel('Y')  
plt.show()

#Save my Plot 1
plt.savefig('py_orig.png')

# In[68]:


x = dataset['x'].values.reshape(-1,1)
y = dataset['y'].values.reshape(-1,1)


# In[78]:


x_test = x
y_test = y 
x_train = x
y_train = y


# In[79]:


regressor = LinearRegression()  
regressor.fit(x_train, y_train) 


# In[80]:


#To retrieve the intercept:
print(regressor.intercept_)

#For retrieving the slope:
print(regressor.coef_)


# In[81]:


y_pred = regressor.predict(x_test)


# In[82]:


df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df


# In[84]:

#**Plot 2**
plt.scatter(x_test, y_test,  color='gray')
plt.plot(x_test, y_pred, color='red', linewidth=2)
plt.show()


# In[ ]:
#Save my Plot2
plt.savefig('py_lm.png')

