#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


data=pd.read_csv('iris.csv')


# In[5]:


data.head()


# In[6]:


y=data[['Species']]


# In[8]:


x=data[['Sepal.Length']]


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


dct=DecisionTreeClassifier()


# In[13]:


dct.fit(x_train,y_train)


# In[14]:


y_pred=dct.predict(x_test)


# In[15]:


y_test.head()


# In[16]:


y_pred[0:5]


# In[17]:


from sklearn.metrics import confusion_matrix


# In[18]:


confusion_matrix(y_test,y_pred)


# In[19]:


(19+8+11)/(19+2+0+4+8+7+1+8+11)


# In[20]:


x=data[['Sepal.Width']]


# In[22]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[23]:


dct.fit(x,y)


# In[24]:


y_pred=dct.predict(x_test)


# In[26]:


y_test.head()


# In[27]:


y_pred[0:5]


# In[28]:


confusion_matrix(y_test,y_pred)


# In[29]:


(15+16+7)/(15+1+4+2+16+6+6+3+7)


# In[ ]:




