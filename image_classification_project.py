#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


data=pd.read_csv(r"C:\Users\thama\Downloads\mnist-in-csv.zip")


# In[17]:


data.head()


# In[23]:


a=data.iloc[2,1:].values


# In[24]:


a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[25]:


df_x=data.iloc[:,1:]
df_y=data.iloc[:,0]


# In[26]:


x_train, x_test, y_train, y_test=train_test_split(df_x,df_y,test_size=0.2,random_state=4)


# In[27]:


x_train.head()


# In[28]:


y_train.head()


# In[29]:


rf=RandomForestClassifier(n_estimators=100)


# In[30]:


rf.fit(x_train,y_train)


# In[31]:


pred=rf.predict(x_test)


# In[32]:


pred


# In[35]:


s=y_test.values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
      count+=1
        


# In[36]:


count


# In[37]:


len(pred)


# In[38]:


11622/12000


# In[ ]:




