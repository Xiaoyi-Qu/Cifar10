# Import necessary packages
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# Edit the code from https://github.com/facebookresearch/LLM-QAT/blob/main/models/utils_quant.py

# Implement a symmetric quantization method (per-tensor quantization)
class SymQuantizer(torch.autograd.Function):
    """
    uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits):
        """
        :ctx: save the input and clip_val in ctx for backward pass
        :input: tensor to be quantized
        :clip_val: clip the tensor before quantization
        :quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        # compute the maximum of the input 
        max_input = torch.max(torch.abs(input)).expand_as(input)

        # quantization process (use range (-128, +127) to understand the numerator)
        scale = max_input / (2 ** (num_bits - 1) - 1)
        output = torch.round(input.div(scale + 1e-6)) * scale

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :ctx: saved non-clipped full-precision tensor and clip_val
        :grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

# Implement a quantizer
class QuantizeLinear(nn.Linear):
    def __init__(
        self,
        *kargs,
        bias=True,
        w_bits=8,
        a_bits=8,
    ):
        super(QuantizeLinear, self).__init__(*kargs, bias=True)
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.act_quantizer = SymQuantizer

    def forward(self, input_):
        # quantize weight only
        assert len(self.weight.size()) == 2
        real_weights = self.weight

        weight_clip_val = torch.tensor([-200.0, 200.0])
        weight = SymQuantizer.apply(
            real_weights, weight_clip_val, self.w_bits
        )

        # act_clip_val = torch.tensor([-200.0, 200.0])
        # input_ = self.act_quantizer.apply(
        #     input_, act_clip_val, self.a_bits
        # )
             
        out = nn.functional.linear(input_, weight)
        if self.bias is not None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

# Implement a naive multilayer perceptron with fake quantization layer
# hidden_size1 = 3*32*32
# intermediate_size1 = 64
# intermediate_size1 = 32
# hidden_size2 = 10
class QMLP(nn.Module):
    def __init__(self, hidden_size1, intermediate_size1, intermediate_size2, hidden_size2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear1 = QuantizeLinear(
            hidden_size1,
            intermediate_size1,
            bias=True,
            w_bits=8,
            a_bits=8,
        )

        self.linear2 = QuantizeLinear(
            intermediate_size1,
            intermediate_size2,
            bias=True,
            w_bits=8,
            a_bits=8,
        )

        self.linear3 = QuantizeLinear(
            intermediate_size2,
            hidden_size2,
            bias=True,
            w_bits=8,
            a_bits=8,
        )

    def forward(self, x):
        x1 = self.linear1(self.flatten(x))
        x2 = self.linear2(nn.functional.relu(x1))
        x3 = self.linear3(nn.functional.relu(x2))
        
        return x3
    
# model = QMLP(3*32*32, 64, 32, 10) # Instantiate the model
# torch.manual_seed(0)
# input = torch.rand(1,3,32,32)*1000
# output = model.forward(input)
# print(output)