
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)
df=pd.read_csv("D:\Personal\BA\sample files\diabetes.csv")
#
df.shape
# have 768 row with 9 column


# In[30]:


df.dtypes 
# All data types are numbers


# In[28]:


df.head()


# In[8]:


df.tail()


# In[3]:


df.describe()
# Appox 35% of patients are diabetes


# In[19]:


#normalize our data in order to visualize it better
df1= (df - df.mean()) / (df.max() - df.min())
df1.describe()


# In[32]:


df1.corr()


# In[46]:


df1.plot.box()
# plot box shoing the correlation


# In[9]:


co = df.corr()
sns.heatmap(co, annot=True)
# correlation map - shows Glucose as the highes probability with 47%


# In[17]:


df.hist(figsize=(12,8),bins=20)


# In[10]:


sns.pairplot(df)


# In[14]:


x=df['Age']
y=df['Glucose']
z=df['Outcome']
df.corr()
colors = {1:'red',0:'green'}
plt.scatter(x,y,c=z.apply(lambda x: colors[x]))
plt.xlabel('Age')
plt.ylabel('Glucose')
plt.show()


# In[15]:


x=df['Glucose']
y=df['BMI']
z=df['Outcome']
df.corr()
colors = {1:'red',0:'green'}
plt.scatter(x,y,c=z.apply(lambda x: colors[x]))
plt.xlabel('Glucose')
plt.ylabel('BMI')
plt.show()


# In[16]:


x=df['SkinThickness']
y=df['BMI']
z=df['Outcome']
df.corr()
colors = {1:'red',0:'yellow'}
plt.scatter(x,y,c=z.apply(lambda x: colors[x]))
plt.xlabel('SkinThickness')
plt.ylabel('BMI')
plt.show()


# In[18]:


#Bin variables
import numpy as np
df=pd.read_csv("D:\Personal\BA\sample files\diabetes.csv")


# In[26]:


df['pwBin'] = np.where(df['Glucose'] <=125, '<=125','>125')
df


# In[27]:


pd.pivot_table(df, values="Glucose", index=["pwBin"], columns="Outcome", aggfunc = "count",fill_value=0)


# In[28]:


train, test = np.split(df.sample(frac=1), [int(.8*len(df))])


# In[29]:


train


# In[30]:


test


# In[31]:


sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.show()


# In[32]:


sns.boxplot(x="Outcome", y="pwBin", data=df)
plt.show()


# In[33]:


sns.pairplot(df, hue="Outcome", size=3)
plt.show()


# In[34]:


pd.pivot_table(train, values="Glucose", index=["pwBin"], columns="Outcome", aggfunc = "count",fill_value=0)


# In[35]:


pd.pivot_table(test, values="Glucose", index=["pwBin"], columns="Outcome", aggfunc = "count",fill_value=0)

