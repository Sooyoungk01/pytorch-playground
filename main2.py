import models as m
import cal_flops as c
import pandas as pd

def main2():

    df = pd.read_csv("/home/ubuntu/experiment1/models_flops.csv")
    df = df[['Model', '32-bit', '16-bit', '8-bit']]

    row_lists = []

    for i in range(31, len(m.model_lists)):
        model = m.model_lists[i]
        
        #cal_flops(model, quant_method, param_bits, bn_bits, overflow_rate, batch_size):
        result = c.cal_flops(model, 'linear', 32, 32, 0, 1)

        flops = result.split()
        if (flops[1] == 'MFLOPS'):
            flops_32 = float(flops[0]) / 1000
        elif (flops[1] == 'GFLOPS'):
            flops_32 = float(flops[0])
        else:
            flops_32 = "이게 뭐야"
        flops_16 = flops_32 / 2
        flops_8 = flops_16 / 2
        

        row = {'Model': model, '32-bit': flops_32, '16-bit': flops_16, '8-bit': flops_8}
        row_lists.append(row)
        df2 = pd.DataFrame(row_lists)

        total_df = pd.concat([df, df2], ignore_index=True)
        total_df.to_csv("/home/ubuntu/experiment1/models_flops.csv")


if __name__ == '__main__':
    main2()
