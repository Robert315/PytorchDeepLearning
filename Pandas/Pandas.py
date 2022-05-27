import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']
lst = [10, 20, 30]
arr = np.array(lst)
d = {'a': 10, 'b': 20, 'c': 30}

pd.Series(data=arr, index=labels)

ser1 = pd.Series([1, 2, 3, 4],index=['USA','Romania','Croatia', 'Polonia'])
# print(ser1)
# print(ser1['USA'])

ser2 = pd.Series([1, 4, 6, 2],index=['USA','Romania', 'Italia', 'Polonia'])

ser3 = ser1 + ser2
# print(ser3)


