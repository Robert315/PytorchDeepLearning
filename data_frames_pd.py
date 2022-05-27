import pandas as pd
import numpy as np
from numpy.random import randn

np.random.seed(101)

rand_mat = randn(5, 4)
# print(rand_mat)

df = pd.DataFrame(data=rand_mat, index='A B C D E '.split(),
                  columns='W X Y Z'.split())  # .split to make a list easy/quickly
# print(df)
one_column = df['W']
# print(one_column)
my_list = ['W', 'Y']  #
# print(df[my_list])

df['NEW_COLUMN'] = df['W'] + df['X'] + df['Y'] + df['Z']  # to make a new column with Pandas
# print(df)

# to drop a column, axis=0 for rows and 1 for columns. inplace=True is for permanently delete.
drop_column = df.drop('NEW_COLUMN', axis=1, inplace=True)

# print(df)

df_loc = df.loc['A']  # to extract data from a rows
# print(df_loc)

df_iloc = df.iloc[2]
# print(df_iloc)

square_loc = df.loc[['A', 'B']][['X', 'Y']]  # to extract data specify
# print(square_loc)

cond1 = df['W'] > 0
cond2 = df['Y'] > 1

cond1_and_cond2 = df[cond1 & cond2]  # '&' it`s 'and'... '|' it`s 'or'
# print(cond1_and_cond2)


reset_index = df.reset_index()  # restore index
# print(reset_index)

new_ind = 'CA NY WY OR CO'.split()
# print(new_ind)

df['States'] = new_ind
df.set_index('States', inplace=True)  # set above index
# print(df)
#
# print(df.info()) ### info
# print(df.describe(())) ###info

ser_w = df['W'] > 0
# print(ser_w.value_counts()) # .value_counts() tell us how many are True and how many are False


##################  GroupBy ##################

### It allows you to split your data into separate groups to perform computations for better analysis.

data = {'Company': ['GOOG', 'GOOG', 'MSFT', 'MSFT', 'FB', 'FB'],
        'Person': ['Sam', 'Charlie', 'Amy', 'Vanessa', 'Carl', 'Sarah'],
        'Sales': [200, 120, 340, 124, 243, 350]
        }
data_frame = pd.DataFrame(data)
# print(data_frame)

data_frame_groupby = data_frame.groupby('Company')  # you can use the .groupby() method to group rows together based off of a column name.

group_by_sales = data_frame.groupby('Company').mean()  # call aggregate methods off the object
# print(group_by_sales)


##################  Pandas Operations ##################

d_frame = pd.DataFrame({'col1': [1, 2, 3, 4],
                        'col2': [444, 555, 666, 444],
                        'col3': ['abc', 'def', 'ghi', 'xyz']
                        })

unique = d_frame['col2'].unique()
# print(unique)

value_counter = d_frame['col2'].value_counts()  # numara de cate ori sunt in interiorul listei elementele
# print(value_counter)

new_df = d_frame[(d_frame['col1'] > 2) & (d_frame['col2'] == 444)]
# print(d_frame)
# print(new_df)

def times_two(number):
    return number*2

d_frame['new'] = d_frame['col1'].apply(times_two) # apply a function in a data frame
print(d_frame)

del d_frame['new']
print(d_frame)
































