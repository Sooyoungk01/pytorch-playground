import models as m
from calflops import calculate_flops
from utee import misc, quant, selector
import timm
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict


def cal_flops(model, quant_method, param_bits, bn_bits, overflow_rate, batch_size):

    input_shape = (batch_size, 3, 224, 224)

    model_raw = timm.create_model(model, pretrained=True).cuda()

    # quantize parameters
    if param_bits < 32:
        state_dict = model_raw.state_dict()
        state_dict_quant = OrderedDict()
        sf_dict = OrderedDict()
        for k, v in state_dict.items():
            if 'running' in k:
                if bn_bits >=32:
                    print("Ignoring {}".format(k))
                    state_dict_quant[k] = v
                    continue
                else:
                    bits = bn_bits
            else:
                bits = param_bits

            if quant_method == 'linear':
                sf = bits - 1. - quant.compute_integral_part(v, overflow_rate=overflow_rate)
                v_quant  = quant.linear_quantize(v, sf, bits=bits)
            elif quant_method == 'log':
                v_quant = quant.log_minmax_quantize(v, bits=bits)
            elif quant_method == 'minmax':
                v_quant = quant.min_max_quantize(v, bits=bits)
            else:
                v_quant = quant.tanh_quantize(v, bits=bits)
            state_dict_quant[k] = v_quant
            print(k, bits)
        model_raw.load_state_dict(state_dict_quant)


    flops, macs, params = calculate_flops(model=model_raw, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)

    print(model + "FLOPs: %s    MACs:%s    Params:%s    \n" %(flops, macs, params))

    return flops

