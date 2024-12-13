import models as m
import imagenet as ig
from utee import misc, quant, selector
import timm
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from collections import OrderedDict


def new_quantize(model, quant_method, param_bits, fwd_bits, bn_bits, overflow_rate, batch_size):

    gpu = '0'
    seed = 117
    n_sample = 20
    ngpu = len(gpu)
    input_size = 224
    model_root = '~/experiment1/models/'
    data_root = '~/experiment1/data'
    assert quant_method in ['linear', 'minmax', 'log', 'tanh']

    assert torch.cuda.is_available(), 'no cuda'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    # load model and dataset fetcher
    model_raw = timm.create_model(model, pretrained=True).cuda()
    ds_fetcher = ig.dataset.get
    is_imagenet = True

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

    # quantize forward activation
    if fwd_bits < 32:
        model_raw = quant.duplicate_model_with_quant(model_raw, bits=fwd_bits, overflow_rate=overflow_rate,
                                                     counter=n_sample, type=quant_method)
        print(model_raw)
        val_ds_tmp = ds_fetcher(10, data_root=data_root, train=False, input_size=input_size)
        misc.eval_model(model_raw, val_ds_tmp, ngpu=1, n_sample=n_sample, is_imagenet=is_imagenet)

    # eval model
    val_ds = ds_fetcher(batch_size, data_root=data_root, train=False, input_size=input_size)
    acc1, acc5 = misc.eval_model(model_raw, val_ds, ngpu=ngpu, is_imagenet=is_imagenet)

    # print sf
    print(model)
    res_str = ", quant_method={}, param_bits={}, bn_bits={}, fwd_bits={}, overflow_rate={}, acc1={:.4f}, acc5={:.4f}".format(
        quant_method, param_bits, bn_bits, fwd_bits, overflow_rate, acc1, acc5)
    print(res_str)
    with open('acc1_acc5.txt', 'a') as f:
        f.write(model + res_str + '\n')
    
    acc1, acc5 = round(acc1.item(),4), round(acc5.item(), 4)

    return acc1, acc5

