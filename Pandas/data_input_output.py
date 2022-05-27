import pandas as pd
#
# df = pd.read_csv(
#     '/home/robert/Documents/PyThorch_Bootcamp/PYTORCH_NOTEBOOKS/00-Crash-Course-Topics/01-Crash-Course-Pandas/example.csv')
# # print(df)
#
# new_df = df[['a', 'b']]
# # print(new_df)
#
# new_df.to_csv('my_new.csv')  # to make a new file csv
#
# read_excel = pd.read_excel(
#     '/home/robert/Documents/PyThorch_Bootcamp/PYTORCH_NOTEBOOKS/00-Crash-Course-Topics/01-Crash-Course-Pandas/Excel_Sample.xlsx')
# # print(read_excel)
#
# mylist_of_table = pd.read_html('https://www.fdic.gov/resources/resolutions/bank-failures/failed-bank-list/')
# print(mylist_of_table)




##########   Exercices   ##########

df = pd.read_csv('/home/robert/Documents/PyThorch_Bootcamp/PYTORCH_NOTEBOOKS/00-Crash-Course-Topics/01-Crash-Course-Pandas/bank.csv')
print(df)
# print(len(df))
#TASK: Display the first 5 rows of the data set

df_5 = df.iloc[:5]
# print(df_5)

# TASK: What is the average (mean) age of the people in the dataset?

# print(df['age'].mean())

# TASK: What is the marital status of the youngest person in the dataset?

var = df['age'].idxmin()
# print(var)
varr = df[df['age'] == 19]
# print(varr)

# TASK: How many unique job categories are there?

unique_job = len(df['job'].unique())
# print(unique_job)

# TASK: How many people are there per job category? (Take a peek at the expected output)

jobs = df['job'].value_counts()
# print(jobs)

# **TASK: What percent of people in the dataset were married? **

married_people = df['marital'].value_counts()['married']/len(df)*100
# print(married_people)

# TASK: There is a column labeled "default". Use pandas' .map() method to create a new column called "default code"
# which contains a 0 if there was no default, or a 1 if there was a default. Then show the head of the dataframe with
# this new column.
df['default code'] = df['default'].map({'no':0,'yes':1})
# print(df)

# TASK: What was the longest lasting duration?

long_duration = df['duration'].max()
# print(long_duration)

# TASK: What is the most common education level for people who are unemployed?

common_education = df['education'].value_counts()
# print(common_education)

# TASK: What is the average (mean) age for being unemployed?

unemployed = df[df['job'] == 'unemployed']['age'].mean()
print(unemployed)
