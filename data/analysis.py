#!/usr/bin/env python
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


df = pd.read_csv('/Users/huyue/Desktop/UCI_Credit_Card.csv')


# In[44]:


print('Shape:', df.shape)
print('Duplicated rows:', df.duplicated().sum())
print('Missing per column:\n', df.isnull().sum().sort_values(ascending=False))


# In[45]:


df.columns = df.columns.str.lower()


# In[46]:


y = df['default.payment.next.month']
X = df.drop(['default.payment.next.month'], axis=1)


# In[53]:


plt.style.use('seaborn')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.histplot(df['age'], bins=30, kde=True, ax=axes[0,0])
axes[0,0].set_title('Age Distribution')

sns.histplot(df['limit_bal'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Credit Limit Distribution')

sns.histplot(df['bill_amt1']/1e4, kde=True, ax=axes[1,0])
axes[1,0].set_title('Latest Bill Amount (10k)')

default_rate = y.mean()
axes[1,1].pie([1-default_rate, default_rate],
              labels=['No Default', 'Default'],
              autopct='%.1f%%', startangle=90)
axes[1,1].set_title('Overall Default Rate')

plt.tight_layout()

plt.show()


# In[56]:


fig, ax = plt.subplots(1, 3, figsize=(15, 4))

sns.barplot(x='sex', y=y, data=df, ax=ax[0])
ax[0].set_title('Default Rate by Sex (1Men 2Women)')

sns.barplot(x='education', y=y, data=df, ax=ax[1])
ax[1].set_title('Default Rate by Education')

sns.barplot(x='marriage', y=y, data=df, ax=ax[2])
ax[2].set_title('Default Rate by Marriage')

plt.tight_layout()
plt.savefig('02_default_by_cats.png', dpi=300)


# In[57]:


num_cols = df.select_dtypes(include=np.number).columns
corr = df[num_cols].corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corr, cmap='coolwarm', square=True, cbar_kws={'shrink': .8})
plt.title('Correlation Matrix')
plt.savefig('03_corr.png', dpi=300)


# In[59]:


use_cols = ['limit_bal', 'sex', 'education', 'marriage', 'age',
            'pay_0', 'bill_amt1', 'pay_amt1']
X_small = df[use_cols]
X_train, X_test, y_train, y_test = train_test_split(
    X_small, y, test_size=0.3, random_state=42, stratify=y)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

print('------Logistic Regression baseline------')
print(classification_report(y_test, y_pred))
print('ROC-AUC:', roc_auc_score(y_test, y_prob))


# In[61]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC - Logistic Regression baseline')
plt.legend()
plt.tight_layout()
plt.savefig('04_roc.png', dpi=300)
plt.show()

