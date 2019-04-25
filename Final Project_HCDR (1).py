#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


import numpy as np


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib


# In[4]:


import seaborn as sns


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import warnings


# In[7]:


warnings.filterwarnings('ignore')


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


import gc


# In[10]:


import os


# In[11]:


print(os.listdir("C:/Users/choti/Documents/National University/ANA 665/PROJECT"))


# In[12]:


import pandas as pd


# In[13]:


Train_data = pd.read_csv('C:/Users/choti/Documents/National University/ANA 665/PROJECT/application_train.csv')


# In[14]:


pd.crosstab(Train_data.TARGET, Train_data.NAME_CONTRACT_TYPE, dropna=False, normalize='all')


# In[15]:


print('Training data shape: ', Train_data.shape)


# In[16]:


Train_data.head()


# In[17]:


Test_data = pd.read_csv('C:/Users/choti/Documents/National University/ANA 665/PROJECT/application_test.csv')


# In[18]:


print('Testing data shape: ', Test_data.shape)


# In[19]:


Test_data.head()


# In[20]:


Train_data['TARGET'].value_counts()


# In[21]:


Train_data['TARGET'].plot.hist();

#Missing Values
# In[22]:


def mis_values(df):
    mis_value = df.isnull().sum() 
    mis_value_per = 100 * df.isnull().sum() / len(df)
    mis_value_column = pd.concat([mis_value, mis_value_per], axis=1)
    mis_val_tab_rename_cols = mis_value_column.rename(columns = {0 : 'Missing Values', 1 : '% of Total Missing Values'})
    mis_val_tab_rename_cols = mis_val_tab_rename_cols[mis_val_tab_rename_cols.iloc[:,1] != 0].sort_values('% of Total Missing Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"  
           
           "There are " + str(mis_val_tab_rename_cols.shape[0]) +
           " cols that have missing values.")
    return mis_val_tab_rename_cols                   
        


# In[23]:


missing_values = mis_values(Train_data)


# In[24]:


print(missing_values)


# In[25]:


def mis_values_test(df):
    mis_value_test = df.isnull().sum() 
    mis_value_per_test = 100 * df.isnull().sum() / len(df)
    mis_value_column_test = pd.concat([mis_value_test, mis_value_per_test], axis=1)
    mis_val_tab_rename_cols_test = mis_value_column_test.rename(columns = {0 : 'Missing Values', 1 : '% of Total Missing Values'})
    mis_val_tab_rename_cols_test = mis_val_tab_rename_cols_test[mis_val_tab_rename_cols_test.iloc[:,1] != 0].sort_values('% of Total Missing Values', ascending=False).round(1)
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"  
           
           "There are " + str(mis_val_tab_rename_cols_test.shape[0]) +
           " cols that have missing values.")
    return mis_val_tab_rename_cols_test         


# In[26]:


missing_values_test = mis_values_test(Test_data)


# In[27]:


print(missing_values_test)


# In[28]:


Train_data.TARGET.value_counts(normalize=True)


# In[29]:


Train_data.dtypes.value_counts()


# In[30]:


pd.crosstab(Train_data.TARGET, Train_data.NAME_CONTRACT_TYPE, dropna=False, normalize='all')


# In[31]:


target_distribution = Train_data['TARGET'].value_counts()
target_distribution.plot.pie(figsize=(12, 12),
                             title='Target Distribution',
                             fontsize=10, 
                             legend=True, 
                             autopct=lambda v: "{:0.1f}%".format(v))


# In[32]:


pd.crosstab(Train_data.TARGET, Train_data.CODE_GENDER, dropna=False)


# In[33]:


print('There are {0} people with realty. {1}% of them repay loans.'.format(Train_data[Train_data.FLAG_OWN_REALTY == 'Y'].
shape[0], np.round(Train_data[Train_data.FLAG_OWN_REALTY == 'Y'].TARGET.value_counts(normalize=True).values[1], 3) * 100))
print('There are {0} people with cars. {1}% of them repay loans.'.format(Train_data[Train_data.FLAG_OWN_CAR == 'Y'].shape[0], 
np.round(Train_data[Train_data.FLAG_OWN_CAR == 'Y'].TARGET.value_counts(normalize=True).values[1], 4) * 100))
print('Average age of the car is {:.2f} years.'.format(Train_data.groupby(['FLAG_OWN_CAR'])['OWN_CAR_AGE'].mean().values[1]))


# In[34]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[35]:


import seaborn as sns


# In[36]:


import matplotlib.pyplot as plt


# In[37]:


sns.countplot(Train_data['NAME_HOUSING_TYPE'])
sns.set_style('whitegrid');
plt.xticks(rotation=45);
plt.title('Counts of housing type')


# In[38]:


Train_data['contact_info'] = Train_data['FLAG_MOBIL'] + Train_data['FLAG_EMP_PHONE'] + Train_data['FLAG_WORK_PHONE'] + Train_data['FLAG_CONT_MOBILE'] + Train_data['FLAG_PHONE'] + Train_data['FLAG_EMAIL']
sns.countplot(Train_data['contact_info'])
sns.set_style('whitegrid');
plt.title('Count of ways to contact client');


# In[39]:


Train_data.loc[Train_data['OBS_30_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_30_CNT_SOCIAL_CIRCLE'] = '1+'
Train_data.loc[Train_data['DEF_30_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_30_CNT_SOCIAL_CIRCLE'] = '1+'
Train_data.loc[Train_data['OBS_60_CNT_SOCIAL_CIRCLE'] > 1, 'OBS_60_CNT_SOCIAL_CIRCLE'] = '1+'
Train_data.loc[Train_data['DEF_60_CNT_SOCIAL_CIRCLE'] > 1, 'DEF_60_CNT_SOCIAL_CIRCLE'] = '1+'


# In[40]:


fig, ax = plt.subplots(figsize = (30, 8), dpi = 100)
plt.subplot(1, 4, 1)
sns.countplot(Train_data['OBS_30_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 2)
sns.countplot(Train_data['DEF_30_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 3)
sns.countplot(Train_data['OBS_60_CNT_SOCIAL_CIRCLE']);
plt.subplot(1, 4, 4)
sns.countplot(Train_data['DEF_60_CNT_SOCIAL_CIRCLE']);


# In[41]:


non_zero_good_price = Train_data[Train_data['AMT_GOODS_PRICE'].isnull() == False]
credit_to_good_price = non_zero_good_price['AMT_CREDIT'] / non_zero_good_price['AMT_GOODS_PRICE']
plt.boxplot(credit_to_good_price);
plt.title('Credit amount to goods price.');


# In[42]:


sns.boxplot(Train_data['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL boxplot');


# In[43]:


sns.boxplot(Train_data[Train_data['AMT_INCOME_TOTAL'] < np.percentile(Train_data['AMT_INCOME_TOTAL'], 80)]['AMT_INCOME_TOTAL'], color='Orange');
plt.title('AMT_INCOME_TOTAL boxplot on data within 80 percentile');


# In[44]:


Train_data.groupby('TARGET').agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count']})


# In[45]:


plt.hist(Train_data['AMT_INCOME_TOTAL']);
plt.title('AMT_INCOME_TOTAL histogram');


# In[46]:


plt.hist(Train_data[Train_data['AMT_INCOME_TOTAL'] < np.percentile(Train_data['AMT_INCOME_TOTAL'], 80)]['AMT_INCOME_TOTAL'], edgecolor = 'black', color = 'Purple');
plt.title('AMT_INCOME_TOTAL histogram on data within 80 percentile');


# In[47]:


plt.hist(np.log1p(Train_data['AMT_INCOME_TOTAL']), edgecolor = 'black', color='Green');
plt.title('AMT_INCOME_TOTAL histogram on data with log1p transformation');


# In[48]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[49]:


import seaborn 


# In[50]:


sns.distplot(np.log1p(Train_data['AMT_INCOME_TOTAL']), hist=True, kde=True, 
             bins=int(180/10), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})


# In[51]:


sns.boxplot(Train_data['AMT_CREDIT'], orient='v', color = 'Pink');
plt.title('AMT_CREDIT boxplot');


# In[52]:


sns.boxplot(Train_data[Train_data['AMT_CREDIT'] < np.percentile(Train_data['AMT_CREDIT'], 80)]['AMT_CREDIT'], orient='v', color = 'Blue');
plt.title('AMT_CREDIT boxplot on data within 80 percentile');


# In[53]:


Train_data.groupby('TARGET').agg({'AMT_CREDIT': ['mean', 'median', 'count']})


# In[54]:


plt.hist(Train_data['AMT_CREDIT'], edgecolor = 'black', color = 'Green');
plt.title('AMT_CREDIT histogram');


# In[55]:


plt.hist(Train_data[Train_data['AMT_CREDIT'] < np.percentile(Train_data['AMT_CREDIT'], 80)]['AMT_CREDIT'], edgecolor = 'black', color = 'Blue');
plt.title('AMT_CREDIT histogram on data within 80 percentile');


# In[56]:


plt.hist(np.log1p(Train_data['AMT_CREDIT']), edgecolor = 'black', color = 'Pink');
plt.title('AMT_CREDIT histogram on data with log1p transformation');


# In[57]:


sns.distplot(np.log1p(Train_data['AMT_CREDIT']), hist=True, kde=True, 
             bins=int(180/10), color = 'deeppink', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 1})


# In[58]:


Train_data['age'] = Train_data['DAYS_BIRTH'] / -365
plt.hist(Train_data['age'], edgecolor = 'black', color = 'deeppink');
plt.title('Histogram of age in years.');


# In[59]:


Train_data.loc[Train_data['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
Train_data['years_employed'] = Train_data['DAYS_EMPLOYED'] / -365
plt.hist(Train_data['years_employed'], edgecolor ='black', color = 'Tomato');
plt.title('Length of working at current workplace in years.');


# In[60]:


Train_data.groupby(['NAME_INCOME_TYPE']).agg({'years_employed': ['mean', 'median', 'count', 'max'], 'age': ['median']})


# In[61]:


sns.countplot(x="NAME_INCOME_TYPE", hue = 'TARGET', data=Train_data);
plt.xticks(rotation=90);


# In[62]:


Train_data.groupby(['NAME_EDUCATION_TYPE', 'NAME_INCOME_TYPE']).agg({'AMT_INCOME_TOTAL': ['mean', 'median', 'count', 'max']})


# In[63]:


sns.countplot(x="NAME_EDUCATION_TYPE", hue = 'TARGET', data=Train_data);
plt.xticks(rotation=90);


# In[64]:


corr = Train_data.corr()
plt.figure(figsize=[25, 25])
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()


# In[65]:


# Plot distribution of multiple features, with TARGET = 1/0 on the same graph
def plot_b_distribution_comp(var,nrow=2):
    
    i = 0
    t1 = Train_data.loc[Train_data['TARGET'] != 0]
    t0 = Train_data.loc[Train_data['TARGET'] == 0]

    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(nrow,2,figsize=(12,6*nrow))

    for feature in var:
        i += 1
        plt.subplot(nrow,2,i)
        sns.kdeplot(t1[feature], bw=0.5,label="TARGET = 1")
        sns.kdeplot(t0[feature], bw=0.5,label="TARGET = 0")
        plt.ylabel('Density plot', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        locs, labels = plt.xticks()
        plt.tick_params(axis='both', which='major', labelsize=12)
    plt.show();


# In[66]:


sns.distplot(np.log1p(Train_data['AMT_INCOME_TOTAL']), hist=True, kde=True, 
             bins=int(180/10), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2})


# In[67]:


var = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']
plot_b_distribution_comp(var, nrow=2)


# # Heatmap of correlations

# In[68]:


var = ['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']
ext_data = Train_data[var]
ext_data_corrs = ext_data.corr();
plt.figure(figsize = (10, 8))
sns.heatmap(ext_data_corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)
plt.title('Correlation Heatmap');


# In[69]:


correlations = Train_data.corr()['TARGET'].sort_values()


# In[70]:


print('Most Positive Correlations:\n', correlations.tail(15))
print('\nMost Negative Correlations:\n', correlations.head(15))


# In[71]:


#Is the relationship between total income and credit amount linear?
sns.regplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=Train_data[Train_data.TARGET == 0])


# In[72]:


sns.regplot(x="AMT_INCOME_TOTAL", y="AMT_ANNUITY", data=Train_data[Train_data.TARGET == 0], color = 'Black')


# In[73]:


sns.regplot(x="AMT_INCOME_TOTAL", y="AMT_GOODS_PRICE", data=Train_data[Train_data.TARGET == 0])


# In[ ]:


plt.figure(figsize=[8, 8])
sns.regplot(x="AMT_CREDIT", y="AMT_GOODS_PRICE", data=Train_data[Train_data.TARGET == 0])


# In[70]:


Train_data_corr = Train_data[['TARGET','DAYS_BIRTH','REGION_RATING_CLIENT_W_CITY','REGION_RATING_CLIENT',
                          'DAYS_LAST_PHONE_CHANGE','FLOORSMAX_AVG','DAYS_EMPLOYED','EXT_SOURCE_1', 
                          'EXT_SOURCE_2', 'EXT_SOURCE_3']].copy()


# In[71]:


target = Train_data['TARGET']
target.shape


# In[72]:


# Calculate correlations
corr = Train_data_corr.corr()
# Heatmap
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True, linewidths=.2, cmap="YlGnBu");


# In[73]:


# Align the training and testing data, keep only columns present in both dataframes
Train_data = Train_data.drop('TARGET', axis=1) #drop target variable from training dataset
Train_data, Test_data = Train_data.align(Test_data, join = 'inner', axis = 1)


# In[74]:


Train_data.shape, Test_data.shape, target.shape


# In[75]:


Train_data['training_set'] = True 
Test_data['training_set'] = False


# In[76]:


df_full = pd.concat([Train_data, Test_data]) #concatenate both dataframes
df_full = df_full.drop('SK_ID_CURR', axis=1) #drop SK_ID_CURR variable
df_full.shape


# In[77]:


print('Size of Full dataset df_full is: {}'.format( df_full.shape))


# # Encoding Categorical Variables

# In[78]:


le = LabelEncoder()

df_full.dtypes.value_counts()


# In[79]:


le_count = 0
for col in df_full.columns[1:]:
    if df_full[col].dtype == 'object':
        if len(list(df_full[col].unique())) <= 2:
            le.fit(df_full[col])
            df_full[col] = le.transform(df_full[col])
            le_count += 1
print('{} columns were label encoded.'.format(le_count))


# In[80]:


# convert rest of categorical variable into dummy
df_full = pd.get_dummies(df_full)


# In[81]:


print('Size of Full Encoded Dataset', df_full.shape)


# # Feature Engineering

# In[82]:


Train_data['TARGET'] = target
df_doc_corr = Train_data[['TARGET','FLAG_DOCUMENT_2','FLAG_DOCUMENT_3','FLAG_DOCUMENT_4',
                        'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8', 
                        'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 
                        'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16',
                        'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20',
                        'FLAG_DOCUMENT_21']].copy()


# In[83]:


# Find correlations with the target and sort
correlations = df_doc_corr.corr()['TARGET'].sort_values()
print('Most Positive Correlations: \n', correlations.tail(5))
print('\nMost Negative Correlations: \n', correlations.head(5))


# In[84]:


# Calculate correlations
corr = df_doc_corr.corr()
# Heatmap
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True, linewidths=.2, cmap="YlGnBu");


# In[86]:


from sklearn.preprocessing import MinMaxScaler, Imputer
imputer = Imputer(strategy = 'median') # Median imputation of missing values
scaler = MinMaxScaler(feature_range = (0, 1)) # Scale each feature to 0-1


# In[87]:


for column in df_full.columns:
    df_full[[column]] = imputer.fit_transform(df_full[[column]])
    df_full[[column]] = scaler.fit_transform(df_full[[column]])


# In[92]:


df_full = df_full.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 
                        'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 
                        'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 
                        'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15'], axis=1)


# In[91]:


def missing_val_ratio(df):
    perc_na = (df.isnull().sum()/len(df))*100
    ratio_na = perc_na.sort_values(ascending=False)
    missing_data_table = pd.DataFrame({'% of Total Values' :ratio_na})
    return missing_data_table


# In[92]:


df_full_miss = missing_val_ratio(df_full)
df_full_miss.head()


# # Data Preparation

# In[93]:


Train_data = df_full[df_full['training_set']==True]
Train_data = Train_data.drop('training_set', axis=1)
Test_data = df_full[df_full['training_set']==False]
Test_data = Test_data.drop('training_set', axis=1)


# In[94]:


print('Size of training_set: ', Train_data.shape)
print('Size of testing_set: ', Test_data.shape)from sklearn.ensemble import RandomForestClassifier
# Train on the training data
rf_model = RandomForestClassifier() 
rf_model.fit(df_train, target)


# In[95]:


print('Size of target: ', target.shape)
print('Size of original data_test: ', Test_data.shape)


# # Logistic Regression

# In[103]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# Make the model with the specified regularization parameter
log_reg = LogisticRegression(random_state = 42)
# Train on the training data
log_reg.fit(Train_data, target)
log_reg_predict = log_reg.predict(Test_data)
print(log_reg.predict)
#roc_auc_score(y_test, log_reg_predict)


# In[127]:


y_pred = log_reg.predict(Train_data)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(Train_data, target)))


# # Random Forest

# In[118]:


from sklearn.ensemble import RandomForestClassifier
# Train on the training data
rf_model = RandomForestClassifier() 
rf_model.fit(Train_data, target)


# In[120]:


random_forest_pred = rf_model.predict_proba(Test_data)[:, 1]
print(random_forest_pred)


# In[111]:


# Train on the training data
opt_rf_model = RandomForestClassifier(n_estimators=200, 
                                      min_samples_split=10, 
                                      min_samples_leaf=5, 
                                      n_jobs=-1, 
                                      random_state=42) 
opt_rf_model.fit(Train_data, target)
opt_RF_pred = opt_rf_model.predict_proba(Test_data)[:, 1]
print(opt_RF_pred)

