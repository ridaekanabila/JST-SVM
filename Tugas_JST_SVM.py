#!/usr/bin/env python
# coding: utf-8

# In[10]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm

data1 = pd.read_excel('Data_JST_SVM_1.xlsx')
x1 = data1['x1']
x2 = data1['x2']
x_training = np.array(list(zip(x1,x2)))
y_training = data1['y']
print(data1)

nama_kelas = ['-1','+1']

i_plus = y_training[y_training>0].index
i_min = y_training[y_training<0].index

plt.figure()
plt.scatter(x_training[i_min,0],x_training[i_min,1],c='b',s=100)
plt.scatter(x_training[i_plus,0],x_training[i_plus,1],c='r',s=100)
plt.legend(nama_kelas)
plt.title('SVM Linier')


# In[18]:


plt.figure()
plt.scatter(x_training[i_min,0],x_training[i_min,1],c='b',s=100)
plt.scatter(x_training[i_plus,0],x_training[i_plus,1],c='r',s=100)
plt.legend(nama_kelas)
plt.title('SVM Linier')

svc = svm.SVC(C=5, kernel='linear').fit(x_training,y_training)
w = svc.coef_
intercept = svc.intercept_
a = -w[0,0] / w[0,1]
b = -intercept[0] / w[0,1]
print('y = '+str(a)+' * x + '+str(b))

x_coba = np.arange(-1,2,0.1)
y_coba = np.multiply(a,x_coba)+b
plt.plot(x_coba,y_coba)


# In[7]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm

data2 = pd.read_excel('Data_JST_SVM_2.xlsx')
x1 = data2['x1']
x2 = data2['x2']
x_training = np.array(list(zip(x1,x2)))
y_training = data2['y']
print(data2)

nama_kelas = ['-1','+1']

i_plus = y_training[y_training>0].index
i_min = y_training[y_training<0].index

plt.figure()
plt.scatter(x_training[i_min,0],x_training[i_min,1],c='b',s=100)
plt.scatter(x_training[i_plus,0],x_training[i_plus,1],c='r',s=100)
plt.legend(nama_kelas)
plt.title('SVM Linier')


# In[8]:


plt.figure()
plt.scatter(x_training[i_min,0],x_training[i_min,1],c='b',s=100)
plt.scatter(x_training[i_plus,0],x_training[i_plus,1],c='r',s=100)
plt.legend(nama_kelas)
plt.title('SVM Linier')

svc = svm.SVC(C=5, kernel='linear').fit(x_training,y_training)
w = svc.coef_
intercept = svc.intercept_
a = -w[0,0] / w[0,1]
b = -intercept[0] / w[0,1]
print('y = '+str(a)+' * x + '+str(b))

x_coba = np.arange(2,6,0.1)
y_coba = np.multiply(a,x_coba)+b
plt.plot(x_coba,y_coba)


# In[ ]:




