#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


titanic_data=sns.load_dataset("titanic")


# In[5]:


titanic_data.head()


# # Exploratory Analysis

# In[8]:


titanic_data.shape


# In[10]:


titanic_data.info()


# In[12]:


pd.isnull(titanic_data).sum()


# # Heatmap 

# In[19]:


sns.heatmap(titanic_data.isnull(), cmap='viridis', cbar=True)
plt.show()


# # Heatmap showing null values in Deck and Age columns

# In[29]:


sns.set_style('whitegrid')
sns.countplot(x='survived',data=titanic_data,hue='survived')
plt.xlabel('Survived (0 = No, 1 = Yes)')


# # More than 300 people can survive while about 500 people can't survived

# In[27]:


sns.set_style('whitegrid')
sns.countplot(x='survived',hue='sex',data=titanic_data)
plt.xlabel('Survived (0 = No, 1 = Yes)')


# # More than 100 male and 200 female can survive but less than 100 female and 500 male can't survive 

# In[35]:


import seaborn as sns
import pandas as pd
titanic_data = sns.load_dataset('titanic')
sns.set_style('whitegrid')
titanic_data['age_group'] = pd.cut(titanic_data['age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
sns.set_style('whitegrid')
sns.countplot(x='survived', hue='age_group', data=titanic_data,)
plt.xlabel('Survived (0 = No, 1 = Yes)')


# # Survival rate with Age

# In[38]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('whitegrid')
sns.countplot(x='survived', hue='pclass', data=titanic_data)
plt.xlabel('Survived (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.title('Survival Count Based on Passenger Class')
plt.show()


# # Survival rate with Passenger Claass

# In[41]:


import seaborn as sns
import matplotlib.pyplot as plt
titanic_data['family_size'] = titanic_data['sibsp'] + titanic_data['parch'] + 1
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))
sns.countplot(x='family_size', hue='survived', data=titanic_data)
plt.xlabel('Family Size')
plt.ylabel('Count')
plt.title('Survival Count Based on Family Size')
plt.show()


# # Survived factor with family size

# # Conclusion:

# In[ ]:


According to the Analysis;
Sociodemographic factors like Age, clas and family size has great impact on survival rate.
Survival rate of female is more than male.
Survival and demise rate both are high in the age goup of Adult.
Passenger class 1 has more survival rate as compared to others.    
Smaller families have better chances of survival.

