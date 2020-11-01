#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning

# In[116]:


import pandas as pd
import numpy as np
df_comp=pd.read_csv('C:\\Users\\lenovo\\Desktop\\Spark Foundation EDA\\companies.csv',encoding= "ISO-8859-1")
df_map=pd.read_csv('C:\\Users\\lenovo\\Desktop\\Spark Foundation EDA\\mapping.csv',encoding= "ISO-8859-1")
df_rou2=pd.read_csv('C:\\Users\\lenovo\\Desktop\\Spark Foundation EDA\\rounds2.csv',encoding = "ISO-8859-1")


# Taking a look at the companies data set

# In[117]:


df_comp.info()


# In[118]:


df_comp.head()


# Removing the first row column

# In[119]:


df_comp.drop('Unnamed: 0',axis=1, inplace=True)


# In[120]:


df_comp.head(5)


# In[121]:


df_rou2.info()


# In[122]:


df_rou2.head()


# Making the Primary keys uniform in both the data sets.

# In[123]:


df_comp['permalink']=df_comp['permalink'].str.lower()
df_rou2['company_permalink']=df_rou2['company_permalink'].str.lower()


# Checking that the meta data of all the companies in round 2 is available with us.

# In[124]:


len(df_comp['permalink'])


# In[125]:


len(df_rou2['company_permalink'])


# In[126]:


len(df_rou2['company_permalink'].unique())


# In[127]:


df_rou2.loc[~df_rou2['company_permalink'].isin(df_comp['permalink'])]


# This is a problem we usally face due to problem while encoding and decoding data.

# Since we want the meta data of all the companies performing a left join to retain all the data of the company data set.

# In[128]:


master_frame=pd.merge(df_comp, df_rou2, how='left', left_on='permalink', right_on='company_permalink')


# In[129]:


master_frame.info()


# In[130]:


master_dataframe.head()


# In[131]:


master_frame.isnull().sum()


# Since we have a lot of misisng values

# In[132]:


master_frame.isnull().sum()*100/len(master_frame['permalink'])


# All these rows would not help in our Analysis.

# In[133]:


master_frame.drop('homepage_url',axis=1,inplace=True)
master_frame.drop('funding_round_code',axis=1,inplace=True)
master_frame.drop('founded_at',axis=1,inplace=True)
master_frame.drop('state_code',axis=1,inplace=True)


# In[134]:


master_frame.info()


# The column raised_amount_usd of atmost significance to us.

# In[136]:


from scipy.stats import kurtosis 
master_frame['raised_amount_usd'].kurtosis()


# This is as far from a normal distribution as anything could be.
# Since teh values only account for 17% of the value, rather than replacing with the mean let's just drop the values.

# In[137]:


master_frame.drop(master_frame[master_frame['raised_amount_usd'].isnull()].index, inplace = True)


# In[138]:


master_frame.isnull().sum()


# # Investment Type Analysis

# In[139]:


master_frame['funding_round_type'].unique()


# In[140]:


master_frame.groupby('funding_round_type').mean().sort_values(by='raised_amount_usd',ascending=False)


# Since The money that is supposed to be invested in should be between 5million-15million USD, the best appropriate funding type is venture.
# Hypotheisis testing wont work since we dont have a normal distribution.

# In[141]:


df_venture= master_frame[master_frame["funding_round_type"]=="venture"]


# In[142]:


df_venture.info()


# In[143]:


df_venture.head()


# # Country

# WE'll decide what Country the investment is to made based on the past capital invsetment trend of the Country.

# In[144]:


df_venture.groupby('country_code').sum().sort_values(by='raised_amount_usd', ascending=False)


# We can see that the top three countries with the maximum invesment in the past.

# In[145]:


df_country=pd.DataFrame(master_frame[master_frame['country_code'].isin(['USA','CHN','GBR'])])


# In[146]:


df_country.info()


# In[147]:


df_country.head()


# # Sector Analysis

# In[148]:


df_map.info()


# In[150]:


df_map.head(0)


# In[51]:


df_map = pd.melt(df_map, id_vars =['category_list'], value_vars =['Automotive & Sports',
                                                              'Cleantech / Semiconductors','Entertainment',
                                                             'Health','Manufacturing','News, Search and Messaging','Others',
                                                             'Social, Finance, Analytics, Advertising']) 


# In[52]:


df_map.head(5)


# Null values are of no use here, hence dropping them.

# In[54]:


df_map.dropna(inplace=True)


# In[55]:


df_map.head(10)


# In[57]:


df_map = df_map[df_map.value == 1]


# In[58]:


df_map.head(5)


# In[60]:


df_map.drop('value', axis=1,inplace=True)
df_map.rename(columns={'category_list':'primary_sector','variable':'main_sector'},inplace=True)


# In[65]:


df_map.head(7)


# Since the category List has more than one values associated with a country so considering just the first one.
# Hence cleaning the data accordingly.

# In[66]:


df_country['primary_sector'] = df_country['category_list'].str.split('|', n = 2, expand = True)[[0]]


# In[67]:


df_country.head()


# Since we want all the information on the country data set, performing a left join on the tables.

# In[68]:


df_country=pd.merge(df_country,df_map,how='left',on='primary_sector')


# In[69]:


df_country.head()


# Now we have our df_country data frame with only the country we want to invest in with the type of funding and also the main 8 sectors to which the companies belong to.
# Applying the condition for the invesment amount and divind the country sets further on the basis of region

# In[70]:


df_fund_USA=df_country[(df_country['country_code'] == 'USA') & (df_country.raised_amount_usd > 5000000.0) & (df_country.raised_amount_usd < 15000000.0)]


# In[71]:


df_fund_USA.info()


# In[72]:


df_fund_USA.head()


# In[75]:


df_fund_USA.sort_values(by='raised_amount_usd',ascending=False)


# In[78]:


df_fund_USA.groupby('main_sector').sum().sort_values(by='raised_amount_usd',ascending=False)


# In[89]:


df_fund_USA.groupby('main_sector').mean().sort_values(by='raised_amount_usd',ascending=False)


# In[79]:


df_fund_CHN=df_country[(df_country['country_code'] == 'CHN') & (df_country.raised_amount_usd > 5000000.0) & (df_country.raised_amount_usd < 15000000.0)]


# In[80]:


df_fund_CHN.info()


# In[81]:


df_fund_CHN.head()


# In[82]:


df_fund_CHN.sort_values(by='raised_amount_usd',ascending=False)


# In[87]:


df_fund_CHN.groupby('main_sector').sum().sort_values(by='raised_amount_usd',ascending=False)


# In[88]:


df_fund_CHN.groupby('main_sector').mean().sort_values(by='raised_amount_usd',ascending=False)


# In[90]:


df_fund_GBR=df_country[(df_country['country_code'] == 'GBR') & (df_country.raised_amount_usd > 5000000.0) & (df_country.raised_amount_usd < 15000000.0)]


# In[91]:


df_fund_GBR.info()


# In[92]:


df_fund_GBR.head()


# In[93]:


df_fund_GBR.groupby('main_sector').sum().sort_values(by='raised_amount_usd',ascending=False)


# In[94]:


df_fund_GBR.groupby('main_sector').mean().sort_values(by='raised_amount_usd',ascending=False)


# In[ ]:




