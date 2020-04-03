import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import time, warnings
import datetime as dt
# Import data (took from Kaggle)
data = pd.read_excel('Online Retail.xlsx')
# drop the row missing customer ID 
data = data[data.CustomerID.notnull()]
data = data.sample(frac = .3).reset_index(drop = True)
data.head()


# extract year, month and day
data['InvoiceDay'] = data.InvoiceDate.apply(lambda x: dt.datetime(x.year, x.month, x.day))
data.head()

# print the time period
print('Min : {}, Max : {}'.format(min(data.InvoiceDay), max(data.InvoiceDay)))

# pin the last date
pin_date = max(data.InvoiceDay) + dt.timedelta(1)


# Create total spend dataframe
data['TotalSum'] = data.Quantity * data.UnitPrice
data.head()


# calculate RFM values
rfm_table = data.groupby('CustomerID').agg({
    'InvoiceDate' : lambda x: (pin_date - x.max()).days,
    'InvoiceNo' : 'count', 
    'TotalSum' : 'sum'})
# rename the columns
rfm_table.rename(columns = {'InvoiceDate' : 'Recency', 
                      'InvoiceNo' : 'Frequency', 
                      'TotalSum' : 'Monetary'}, inplace = True)
rfm_table.head()


# create labels and assign them to tree percentile groups 
r_labels = range(4, 0, -1)
r_groups = pd.qcut(rfm_table.Recency, q = 4, labels = r_labels)
f_labels = range(1, 5)
f_groups = pd.qcut(rfm_table.Frequency, q = 4, labels = f_labels)
m_labels = range(1, 5)
m_groups = pd.qcut(rfm_table.Monetary, q = 4, labels = m_labels)

# make a new column for group labels
rfm_table['R'] = r_groups.values
rfm_table['F'] = f_groups.values
rfm_table['M'] = m_groups.values
# sum up the three columns
rfm_table['RFM_Segment'] = rfm_table.apply(lambda x: str(x['R']) + str(x['F']) + str(x['M']), axis = 1)
rfm_table['RFM_Score'] = rfm_table[['R', 'F', 'M']].sum(axis = 1)
rfm_table.head()


# assign labels from total score
score_labels = ['Bronze' , 'Silver', 'Gold']
score_groups = pd.qcut(rfm_table.RFM_Score, q = 3, labels = score_labels)
rfm_table['RFM_Level'] = score_groups.values
rfm_table.head()


# define function for the values below 0
def neg_to_zero(x):
    if x <= 0:
        return 1
    else:
        return x
# apply the function to Recency and MonetaryValue column 
rfm_table['Recency'] = [neg_to_zero(x) for x in rfm_table.Recency]
rfm_table['Monetary'] = [neg_to_zero(x) for x in rfm_table.Monetary]
# unskew the data
rfm_log = rfm_table[['Recency', 'Frequency', 'Monetary']].apply(np.log, axis = 1).round(3)


# In real world data there are other parameters that affect clusters like age , sex ,occupation , etc. of customer
# Since we do not have that in our Dataset so we're gonna add them ourselves


#Adding more columns:
rfm_table['Age'] = np.random.randint(20, 60, rfm_table.shape[0])
occ=["Doctor","Engineer","Businessmen","Army","Other"]
rfm_table["Occupation"] = np.random.choice(occ,len(rfm_table),p=[0.05,0.2,0.5,0.05,.2])
sex=["Male","Female"]
rfm_table["Sex"] = np.random.choice(sex,len(rfm_table),p=[0.4,0.6])



from sklearn.preprocessing import StandardScaler
# scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_log)
# transform into a dataframe
rfm_scaled = pd.DataFrame(rfm_scaled, index = rfm_table.index, columns = rfm_log.columns)


#Label encoding Occupation/Sex for visualization
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
rfm_table['Sex_encoded'] = le.fit_transform(rfm_table['Sex'])
rfm_table['Occupation_encoded'] = le.fit_transform(rfm_table['Occupation'])
#Visualizing the  3d graph
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(rfm_table['RFM_Score'],rfm_table['Age'], rfm_table['Occupation_encoded'], c='b', marker='o')

ax.set_xlabel('RFM Score')
ax.set_ylabel('Age')
ax.set_zlabel('Occupation_encoded')

plt.show()

#######################creatinf rfm_2

rfm_encoded = rfm_table[['RFM_Score','Age','Occupation_encoded', 'Sex_encoded']].copy()
#######################






# Importing Libraries
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns


# standardizing data
columns_to_normalize = ['RFM_Score','Age']
rfm_encoded[columns_to_normalize] = rfm_encoded[columns_to_normalize].apply(lambda x: (x - x.mean()) / np.std(x))

matrix = rfm_encoded.as_matrix()

# Running K-Prototype clustering
kproto = KPrototypes(n_clusters=3, init='Cao')
clusters = kproto.fit_predict(matrix, categorical=[2])

print(kproto.cluster_centroids_)
print(kproto.cost_)

rfm_encoded['cluster_id'] = clusters

# add cluster_id column to rfm data frame for better understanding
rfm_table['cluster_number']=rfm_encoded['cluster_id'].values

#Checking cluster count
cluster_count = pd.DataFrame(rfm_encoded['cluster_id'].value_counts())
print(cluster_count)

sns.barplot(x=cluster_count.index, y=cluster_count['cluster_id'])

# Results are not the same as the result which is found just from RFM score
# Since the data is not real 