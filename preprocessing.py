import pandas as pd

df1 = pd.read_csv("./fwd32_linear.csv")
df2 = pd.read_csv("./models_flops.csv")

df1 = df1[['Model', '32-bit', '16-bit', '8-bit']]
df2 = df2[['Model', '32-bit', '16-bit', '8-bit']]

indexes_to_remove = []

# 데이터들 중에서 경계선에 있는 얘들만
for i in range(0, df1.shape[0]):
    for j in range(0, df1.shape[0]):
        if ((df1.loc[i,'32-bit'] < df1.loc[j, '32-bit']) & (df2.loc[i, '32-bit'] > df2.loc[j, '32-bit'])):
            indexes_to_remove.append(i)
            break

df1_filtered = df1.drop(indexes_to_remove)
df2_filtered = df2.drop(indexes_to_remove)

# 너무 낮은 accuracy model 없애기
too_much_low_accuracy = df1_filtered[df1_filtered['8-bit'] <= 0.6].index
df1_filtered = df1_filtered.drop(too_much_low_accuracy)
df2_filtered = df2_filtered.drop(too_much_low_accuracy)

# 너무 큰 FLOPs model 없애기
#too_large_FLOPs = df2_filtered[df2_filtered['32-bit'] >= 6].index
#df1_filtered = df1_filtered.drop(too_large_FLOPs)
#df2_filtered = df2_filtered.drop(too_large_FLOPs)

df1_filtered.to_csv("/home/ubuntu/experiment1/filtered_fwd32_linear.csv")
df2_filtered.to_csv("/home/ubuntu/experiment1/filtered_models_flops.csv")
