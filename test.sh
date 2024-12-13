CUDA_VISIBLE_DEVICES=0 python quantize.py --type resnet18 --quant_method minmax --param_bits 4 --fwd_bits 32 --bn_bits 32 --ngpu 1
