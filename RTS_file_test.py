"""
manage the output of vicon.
"""
# coding: utf-8

import re
import pandas as pd

# user-defined
filename = 'out.txt'
outfile = 'test.csv'

# condition = ['Object2']
condition = ['xc_right_hand']
# condition = ['SBOX']

with open(filename, 'r') as f:
    raw_rows = f.readlines()

# 要求数据大于3个刚体
first = raw_rows.index('\n')
second = raw_rows[first + 1:].index('\n')

distance = second + 1
strip_tail = (len(raw_rows) - 1) % distance
rows = raw_rows[2:-strip_tail]

temp = rows[:distance]

# end_index = None
# for i,e in enumerate(temp):
#     if e.startswith('Ron-output file'):
#         end_index = i
#         break
# # value_num = end_index - 1
# value_num = 4
value_num = distance - 1

result = [e for i, e in enumerate(rows) if (i % distance <= value_num) and (i % distance > 0)]
split_result = [re.split(r'[,()]', row)[:10] for row in result]
add_zero_result = [row[:5] + row[6:] for row in split_result]

title = ['time', 'name'] + list('xyzabcw')
df = pd.DataFrame(add_zero_result, columns=title)

df_out = df[df.name.isin(condition)]
df_out.to_csv(outfile, index=False)
