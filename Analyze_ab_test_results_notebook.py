#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# 
# You may either submit your notebook through the workspace here, or you may work from your local machine and submit through the next page.  Either way assure that your code passes the project [RUBRIC](https://review.udacity.com/#!/projects/37e27304-ad47-4eb0-a1ab-8c12f60e43d0/rubric).  **Please save regularly.**
# 
# This project will assure you have mastered the subjects covered in the statistics lessons.  The hope is to have this project be as comprehensive of these topics as possible.  Good luck!
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  It is important that you get some practice working with the difficulties of these 
# 
# In this project, I worked to understand the results of an A/B test run by an e-commerce website. My goal is to work to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[1]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import statsmodels.api as sm
#We are setting the seed to assure you get the same answers on quizzes as we set up
random.seed(42)


# `1.` Now, read in the `ab_data.csv` data. Store it in `df`.  **Use your dataframe to answer the questions in Quiz 1 of the classroom.**
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[2]:


df = pd.read_csv('ab_data.csv')
df.head()


# b. Use the cell below to find the number of rows in the dataset.

# In[3]:


df.shape


# c. The number of unique users in the dataset.

# In[4]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[5]:


conv_users = df.query('converted == 1').user_id.nunique() / df.user_id.nunique()
conv_users


# e. The number of times the `new_page` and `treatment` don't match.

# In[6]:


num = df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]
num


# f. Do any of the rows have missing values?

# In[7]:


df.info()


# In[8]:


print('there is no missing data') 


# `2.` For the rows where **treatment** does not match with **new_page** or **control** does not match with **old_page**, we cannot be sure if this row truly received the new or old page.  Use **Quiz 2** in the classroom to figure out how we should handle these rows.  
# 
# a. Now use the answer to the quiz to create a new dataset that meets the specifications from the quiz.  Store your new dataframe in **df2**.

# In[9]:


df2 = df.copy()
df2 = df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == True]


# In[10]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[11]:


df2.user_id.nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[12]:


re_user_id = (df2[df2.user_id.duplicated()].user_id.iloc[0])
re_user_id


# c. What is the row information for the repeat **user_id**? 

# In[13]:


df2[df2.user_id == re_user_id]


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[14]:


df2.drop(1899, inplace=True)
df2[df2.user_id == re_user_id]


# `4.` Use **df2** in the cells below to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[15]:


conv = df2.converted.mean()
conv


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[16]:


P_C = df2.query('group == "control"').converted.mean()
#P_C = P_C.converted.mean()
P_C 


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[17]:


P_T = df2.query('group == "treatment"').converted.mean()
P_T


# d. What is the probability that an individual received the new page?

# In[18]:


P_N = df2.query('landing_page == "new_page"').user_id.nunique() / df2.user_id.nunique()
P_N


# e. Consider your results from parts (a) through (d) above, and explain below whether you think there is sufficient evidence to conclude that the new treatment page leads to more conversions.

# **Your answer goes here.**
# 
# 

# #### Considering the probability of converting the individuals of both  control group and treatment group, we note that there is a convergence between the probability, and that the probability ratio is not enough to do the new treatment page.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **Put your answer here.**
# #### In the event that we consider that the, The old page is better, it means that the conversion rate of the old page is greater than or equal to the conversion rate of the new page, and this is what we mean by (the null hypothesis)
# 
# 
# #### As for the alternative hypotheses, it means that the conversion rate of the new page is greater than the conversion rate of the old page.

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **conversion rate** for $p_{new}$ under the null? 

# In[19]:


p_new = conv
p_new


# b. What is the **conversion rate** for $p_{old}$ under the null? <br><br>

# In[20]:


p_old = conv
p_old


# c. What is $n_{new}$, the number of individuals in the treatment group?

# In[21]:


n_new = df2.query('landing_page == "new_page"').shape[0]
n_new


# d. What is $n_{old}$, the number of individuals in the control group?

# In[22]:


n_old = df2.query('landing_page == "old_page"').shape[0]
n_old


# e. Simulate $n_{new}$ transactions with a conversion rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[23]:


npage_conv = np.random.choice([0, 1], size=n_new, p=[(1 - p_new), p_new])
p_new2 = npage_conv.mean()
p_new2


# f. Simulate $n_{old}$ transactions with a conversion rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[24]:


opage_conv = np.random.choice([0, 1], size=n_old, p=[(1 - p_old), p_old])
p_old2 = opage_conv.mean()
p_old2


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[42]:


p_new2 - p_old2 


# h. Create 10,000 $p_{new}$ - $p_{old}$ values using the same simulation process you used in parts (a) through (g) above. Store all 10,000 values in a NumPy array called **p_diffs**.

# In[43]:


p_diffs = []

p_diffs = []
for _ in range(10000):
    npage_conv = np.random.choice([0, 1], size=n_new, p=[(1 - p_new), p_new]).mean()
    opage_conv = np.random.choice([0, 1], size=n_old, p=[(1 - p_old), p_old]).mean()
    p_diffs.append(npage_conv - opage_conv)


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[44]:


Observed_diffs = P_T - P_C
p_diffs = np.asarray(p_diffs)
plt.hist(p_diffs)

plt.axvline(Observed_diffs, color='r')


# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[45]:


proportion = (p_diffs >= Observed_diffs).mean()
proportion


# k. Please explain using the vocabulary you've learned in this course what you just computed in part **j.**  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# **Put your answer here.**
# #### Simplified, we are trying to calculate the value of P, which means the probability of not observing any difference between the conversion rates of both the control and treatment groups.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[46]:


convert_old = df2.query('group == "control" & converted == 1')['converted'].count()
convert_new =  df2.query('group == "treatment" & converted == 1')['converted'].count()

convert_old, convert_new, n_old, n_new


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](https://docs.w3cub.com/statsmodels/generated/statsmodels.stats.proportion.proportions_ztest/) is a helpful link on using the built in.

# In[47]:


z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')
z_score, p_value


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# **Put your answer here.**
# 
# ####  z-score : Means the number of standard deviations from (Observed_diff).
# ####  p-value = 0.905
# #### Also, both A and B agree with the results in .

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you achieved in the A/B test in Part II above can also be achieved by performing regression.<br><br> 
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# **Put your answer here.**
# 
# #### Logistic regression is a good option for this case.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives. However, you first need to create in df2 a column for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[48]:


df2['intercept'] = 1
df2['ab_page'] = 1
t_ab_page = df2[df2['group']=='control'].index
df2.loc[t_ab_page, "ab_page"] = 0
df2.head()


# c. Use **statsmodels** to instantiate your regression model on the two columns you created in part b., then fit the model using the two columns you created in part **b.** to predict whether or not an individual converts. 

# In[49]:


lm = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
result = lm.fit()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# In[50]:


result.summary2()


# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in **Part II**?<br><br>  **Hint**: What are the null and alternative hypotheses associated with your regression model, and how do they compare to the null and alternative hypotheses in **Part II**?

# **Put your answer here.**
# 
# #### the p-value associated with ab_page  : 0.1899
# #### The reason why the P-value in Part III is different from the P-value in Part II, is because we rely on logistic regression.
# 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# **Put your answer here.**
# #### I think that time is an influential factor in the transformation of the individual, as it is considered an important factor, I do not think that there are clear complications to add to it.

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives in. You will need to read in the **countries.csv** dataset and merge together your datasets on the appropriate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy variables.** Provide the statistical output as well as a written response to answer this question.

# In[51]:


dfCountries = pd.read_csv('countries.csv')
dfCountries.head()


# In[52]:


new_country_df = dfCountries.join(pd.get_dummies(dfCountries['country']))
new_country_df.head()


# In[53]:


new_country_df.country.unique()


# In[54]:


df3 = df2.set_index('user_id').join(new_country_df.set_index('user_id'))
df3.head()


# In[55]:


lm = sm.Logit(df3['converted'], df3[['intercept', 'UK', 'US']])
results = lm.fit()
results.summary2()


# #### In this case we cannot reject the null hypothesis, since the p-value of US and UK are greater than 0.05.

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[56]:


df3['UK_ab_page'] = df3['UK'] * df3['ab_page']
df3['US_ab_page'] = df3['US'] * df3['ab_page']
df3.head()


# In[57]:


lm = sm.Logit(df3['converted'], df3[['intercept', 'ab_page', 'US', 'UK', 'US_ab_page', 'UK_ab_page']])
results = lm.fit()
results.summary2()


# #### Since the p-values of all the variables are greater than 0.05, this means that the country has no effect on the conversion.

# ## Conclusion :
# #### In conclusion, the final analysis of this case indicates that we do not have analyzes and high numbers that assure us that the new page leads to more conversions than the old page. So after working, analyzing and deducing I see that the new page should be deleted and replaced with another new page with different features to test and see if it leads to more conversions.

# In[58]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])


# In[ ]:




