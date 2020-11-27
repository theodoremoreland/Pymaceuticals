#!/usr/bin/env python
# coding: utf-8

# ## Conclusions:
# 
# - Ramicane had the greatest affect in reducing tumor growth between drugs. Ramicane is one of only two drugs that acheived an overall tumor reduction.
# 
# - Mice given Propriva were the least likely to survive the 45 days; these mice had a survival rate of 26%, which is 22% below the median.
# 
# - Ramicane maintained the fewest number of metastatic sites across the 45 days, ending with approximately 200% less than Mice treated with Ketapril.
# 

# In[1]:

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


drug_data_path = "./data/mouse_drug_data.csv"
trial_data_path = "./data/clinicaltrial_data.csv"
drug_df = pd.read_csv(drug_data_path)
trials_df = pd.read_csv(trial_data_path)
mice_trials_df = pd.merge(trials_df, drug_df, on="Mouse ID")
drugs = [
    "Capomulin"
    , "Ceftamin"
    , "Infubinol"
    , "Ketapril"
    , "Naftisol"
    , "Placebo"
    , "Propriva"
    , "Ramicane"
    , "Stelasyn"
    , "Zoniferol"
]
linewidth = 0.3
marker = '^'


# In[4]:


mice_trials_df.head()


# In[5]:


avg_tumor_volume = pd.DataFrame(
    {
        "Average Tumor Volume (mm3)": mice_trials_df.groupby(["Drug", "Timepoint"]).mean()["Tumor Volume (mm3)"]
    }
).reset_index()


# ## Tumor Response to Treatment

# In[6]:


avg_tumor_volume.head()


# In[7]:


# Store the Standard Error of Tumor Volumes Grouped by Drug and Timepoint
mice_trials_by_drug_and_timepoint = mice_trials_df.groupby(["Drug", "Timepoint"])
tumor_volume_samples = [sample for sample in mice_trials_by_drug_and_timepoint["Tumor Volume (mm3)"]]

drug_samples = [sample[0][0] for sample in tumor_volume_samples]
timepoints = [sample[0][1] for sample in tumor_volume_samples]
means = [sample[1].mean() for sample in tumor_volume_samples]
sems = [sample[1].sem() for sample in tumor_volume_samples]

# Convert to DataFrame
standard_error = pd.DataFrame(
    {
        "Drug": drug_samples
        , "Timepoint": timepoints
        , "Tumor Volume (Standard Error)": sems
    }
)


# In[8]:


standard_error.head()


# In[9]:


# Minor Data Munging to Re-Format the Data Frames
avg_tumor_volume_pivot = pd.pivot_table(
    avg_tumor_volume
    , values="Average Tumor Volume (mm3)"
    , index="Timepoint"
    , columns="Drug"
)


# In[10]:


# Preview that Reformatting worked
avg_tumor_volume_pivot


# In[11]:


try:
    avg_tumor_volume.insert(3, "Tumor Volume (Standard Error)", sems)
except:
    print("The 'avg_tumor_volume' DataFrame already has a 'Standard Error' column.")

# Generate the Plot (with Error Bars)
plt.figure(figsize=(10,8))

for drug in drugs:
    avg_tumor_volume_while_on_drug = avg_tumor_volume[avg_tumor_volume.Drug == drug]
    plt.errorbar(
        avg_tumor_volume_while_on_drug["Timepoint"]
        , avg_tumor_volume_while_on_drug["Average Tumor Volume (mm3)"]
        , yerr=avg_tumor_volume_while_on_drug["Tumor Volume (Standard Error)"]
        , linewidth=linewidth
        , marker=marker
    )

plt.grid(True)
plt.xlabel("Time (Days)")
plt.ylabel("Average Tumor Volume (mm3)")
plt.legend(avg_tumor_volume.Drug.unique(), loc="upper left")

# Save the Figure
plt.savefig("./presentation/1.png")


# In[12]:


# Store the Mean Met. Site Data Grouped by Drug and Timepoint 
met_samples = [sample for sample in mice_trials_by_drug_and_timepoint["Metastatic Sites"]]

drug_samples = [sample[0][0] for sample in met_samples]
timepoints = [sample[0][1] for sample in met_samples]
means = [sample[1].mean() for sample in met_samples]
sems = [sample[1].sem() for sample in met_samples]

# Convert to DataFrame
avg_met = pd.DataFrame(
    {
        "Drug": drug_samples
         , "Timepoint": timepoints
         , "Metastatic Sites (Average)": means
    }
)


# In[13]:


# Store the Standard Error associated with Met. Sites Grouped by Drug and Timepoint 
standard_error_met = pd.DataFrame(
    {
        "Drug": drug_samples
        , "Timepoint": timepoints
        , "Metastatic Sites (Standard Error)": sems
    }
)


# ## Metastatic Response to Treatment

# In[14]:


avg_met.head()


# In[15]:


standard_error_met.head()


# In[16]:


# Minor Data Munging to Re-Format the Data Frames
standard_error_met_pivot = pd.pivot_table(
    standard_error_met
    , values="Metastatic Sites (Standard Error)"
    , index="Timepoint"
    , columns="Drug"
)


# In[17]:


# Preview that Reformatting worked
standard_error_met_pivot


# In[18]:


try:
    avg_met.insert(3, "Metastatic Sites (Standard Error)", sems)
except:
    print("The 'avg_met' DataFrame already has a 'Standard Error' column.")

# Generate the Plot (with Error Bars)
plt.figure(figsize=(10,8))

for drug in drugs:
    avg_met_while_on_drug = avg_met[avg_met.Drug == drug]
    plt.errorbar(
        avg_met_while_on_drug["Timepoint"]
        , avg_met_while_on_drug["Metastatic Sites (Average)"]
        , yerr=avg_met_while_on_drug["Metastatic Sites (Standard Error)"]
        , linewidth=linewidth
        , marker=marker
    )

plt.grid(True)
plt.xlabel("Treatment Duration (Days)")
plt.ylabel("Metastatic Sites")
plt.legend(avg_met.Drug.unique(), loc="upper left")

# Save the Figure
plt.savefig("./presentation/2.png")


# In[19]:


# Store the Count of Mice Grouped by Drug and Timepoint (W can pass any metric)
mice = mice_trials_by_drug_and_timepoint    .count()    .reset_index()["Mouse ID"]

# Convert to DataFrame
mice_survival_over_time_df = pd.DataFrame(
    {
        "Drug": drug_samples
        , "Timepoint": timepoints
        , "Mice": mice
    }
)


# In[20]:


# Minor Data Munging to Re-Format the Data Frames
mice_survival_over_time_df_pivot = pd.pivot_table(
    mice_survival_over_time_df
    , values="Mice"
    , index="Timepoint"
    , columns="Drug"
)


# ## Survival Rates

# In[21]:


# Preview DataFrame
mice_survival_over_time_df.head()


# In[22]:


# Preview pivoted DataFrame
mice_survival_over_time_df_pivot


# In[23]:


# Generate the Plot (Accounting for percentages)
plt.figure(figsize=(10,8))

for drug in drugs:
    starting_number_of_mice = mice_survival_over_time_df_pivot[drug][0]
    plt.plot(
        100 * mice_survival_over_time_df_pivot[drug] / starting_number_of_mice
        , linewidth=linewidth
        , marker=marker)

plt.grid(True)
plt.xlabel("Time (Days)")
plt.ylabel("Survival Rate (%)")
plt.legend(avg_met.Drug.unique(), loc="lower left")

# Save the Figure
plt.savefig("./presentation/3.png")

# Show the Figure
plt.show()


# In[24]:


percentage_change_list = []

# Calculate the percent changes for each drug and display data
print("Drug")
for drug in drugs:
    averages_list = list(avg_tumor_volume[(avg_tumor_volume.Drug == drug)]["Average Tumor Volume (mm3)"])
    percentage_change = 100 * (averages_list[-1] - averages_list[0]) / averages_list[0]
    print(drug, "  ", percentage_change)
    percentage_change_list.append(percentage_change)
    
# Store all relevant percent changes into a Tuple
tumor_volume_change_tup = tuple(percentage_change_list)


# ## Summary Bar Graph

# In[25]:


plt.figure(figsize=(12,8))
plt.bar(
    avg_met.Drug.unique()
    , tumor_volume_change_tup
    , color=["g","r", "r", "r", "r", "r", "r", "g", "r", "r"]
)
plt.grid(True, color="black", linestyle="dashed")
plt.title("Tumor Change Over 45 Day Treatment")
plt.ylabel("% Tumor Volume Change")

def plot_tumor_volume_change_text(tup):
    for i in range(len(tup)):
        if tup[i] > 0:
            x = i
            label = "% " + str(round(tup[i]))
            plt.text(x, 3, label, ha='center', va='top', color="white", fontsize="xx-large")
        else:
            x = i
            label = "% " + str(round(tup[i]))
            plt.text(x, -3, label, ha='center', va='top', color="white", fontsize="xx-large")
            
plot_tumor_volume_change_text(tumor_volume_change_tup)

# Save the Figure
plt.savefig("./presentation/4.png")

