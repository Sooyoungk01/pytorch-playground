import models as m
import imagenet as ig
from utee import misc, quant, selector
import new_quantize as nq
import timm
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict
import pandas as pd

def main():

    df = pd.read_csv("/home/ubuntu/experiment1/fwd32_log.csv")
    df = df[['Model', 'Quant Method', '32-bit', '16-bit', '8-bit']]

    row_lists = []

    for i in range(53, len(m.model_lists)):
        model = m.model_lists[i]
        # new_quantize(model, quant_method, param_bits, fwd_bits, bn_bits, overflow_rate, batch_size):
        acc1_32, _ = nq.new_quantize(model, 'log', 32, 32, 32, 0.0, 10)
        acc1_16, _ = nq.new_quantize(model, 'log', 16, 32, 32, 0.0, 10)
        acc1_8, _ = nq.new_quantize(model, 'log', 8, 32, 32, 0.0, 10)

        row = {'Model': model, 'Quant Method': ['linear'], '32-bit': acc1_32, '16-bit': acc1_16, '8-bit': acc1_8}
        row_lists.append(row)
        df2 = pd.DataFrame(row_lists)

        total_df = pd.concat([df, df2], ignore_index=True)
        total_df.to_csv("/home/ubuntu/experiment1/fwd32_log.csv")


if __name__ == '__main__':
    main()
