import torch
import torch.nn

trainable_parameters_num = lambda model : sum(p.numel() for p in model.parameters() if p.requires_grad)

def out_shape(model, input_shape=(1,1,256,256)):
    x = torch.randn(input_shape)
    out = model(x)
    
    return out.shape