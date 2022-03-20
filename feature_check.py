#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pickle
import matplotlib.pyplot as plt


# In[25]:


pk = pickle.load(open("/home/ubuntu7/data/disk0/alphafold_check/testout/P27695/result_model_1.pkl","rb"));


# In[26]:


pk.keys()


# In[39]:


plt.plot(pk['experimentally_resolved']['logits'][:,0:3]);


# In[17]:


pk['experimentally_resolved']['logits'].shape


# In[28]:


plt.plot(pk['plddt']);


# In[ ]:




