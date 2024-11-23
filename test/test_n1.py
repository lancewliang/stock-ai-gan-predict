import pandas as pd
 
# 建立示例DataFrame
df = pd.DataFrame({
    'A': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    'C': [1, 1, 3, 1, 2, 2, 1, 3],
    'D': [10, 10, 30, 10, 20, 20, 20, 30]
})
   
# 根據索引去重
df_unique = df.drop_duplicates(inplace=True)  # 保留第一次出現的重複數據行
 
print(df)




# 创建两个DataFrame
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
 
# 垂直拼接
result = pd.concat([df1, df2], axis=0)

print(result)