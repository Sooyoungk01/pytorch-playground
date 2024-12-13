import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("./filtered_fwd32_linear.csv")
df2 = pd.read_csv("./filtered_models_flops.csv")

acc_32 = df['32-bit'].to_numpy()
flops_32 = df2['32-bit'].to_numpy()
acc_16 = df['16-bit'].to_numpy()
flops_16 = df2['16-bit'].to_numpy()
acc_8 = df['8-bit'].to_numpy()
flops_8 = df2['8-bit'].to_numpy()


def sort_data_by_flops(flops, accuracy):
    sorted_indices = np.argsort(flops)
    return flops[sorted_indices], accuracy[sorted_indices]


flops_32_sorted, acc_32_sorted = sort_data_by_flops(flops_32, acc_32)
flops_16_sorted, acc_16_sorted = sort_data_by_flops(flops_16, acc_16)
flops_8_sorted, acc_8_sorted = sort_data_by_flops(flops_8, acc_8)


plt.scatter(flops_32_sorted, acc_32_sorted, color='orange', label='32-bit')
plt.scatter(flops_16_sorted, acc_16_sorted, color='blue', label='16-bit')
plt.scatter(flops_8_sorted, acc_8_sorted, color='yellow', label='8-bit')
plt.plot(flops_32_sorted, acc_32_sorted, color='orange', label='32-bit')
plt.plot(flops_16_sorted, acc_16_sorted, color='blue', label='16-bit')
plt.plot(flops_8_sorted, acc_8_sorted, color='yellow', label='8-bit')

plt.xlabel('GFLOPs')
plt.ylabel('Top-1 Accuracy')
plt.title('Impact of Quantization on Model Performance')
plt.legend()

plt.xscale('log')

plt.savefig('sample_graph5.png', dpi=300, bbox_inches='tight')