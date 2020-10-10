#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


# In[2]:


url = 'https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'


# In[3]:


Data = pd.read_csv(url,error_bad_lines=False)


# In[4]:


Data


# # Lets Visualise our Data

# In[77]:


sns.set_style("darkgrid")
sns.lmplot(x='Hours',y='Scores',data=Data)


# In[14]:


sns.distplot(Data.Hours)


# In[15]:


sns.distplot(Data.Scores)


# Let's prepare out data for model fitting

# In[36]:


X = Data.iloc[:, :-1].values
y = Data.iloc[:, 1].values  


# In[37]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# Training the Algorithm

# In[38]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


# In[61]:


y_pred = regressor.predict(X_test)


# In[60]:


X_test.reshape(1,-1)


# In[70]:


hour = [[9.5]]
prediction = regressor.predict(hour)
print("Score of a person who study 9.5 hr/day:" ' ', prediction)


# In[73]:


from sklearn import metrics


# In[75]:


metrics.mean_absolute_error(y_test, y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




