import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import torch.nn as nn

def plot_matrix(tensor, ax, title, vmin=0, vmax=1, cmap=None):
    """
    Plot a heatmap of tensors using seaborn
    """
    # print (tensor)
    sns.heatmap(tensor.numpy(), ax=ax, vmin=vmin, vmax=vmax, cmap=cmap, annot=True, fmt=".3f", cbar=False)
    ax.set_title(title)
    ax.set_yticklabels([])
    ax.set_xticklabels([])


def plot_quantization_errors(original_tensor, quantized_tensor, dequantized_tensor, dtype = torch.int8, n_bits = 8):
    """
    A method that plots 4 matrices, the original tensor, the quantized tensor
    the de-quantized tensor and the error tensor.
    """
    # Get a figure of 4 plots
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    # Plot the first matrix
    plot_matrix(original_tensor, axes[0], 'Original Tensor', cmap=ListedColormap(['white']))

    # Get the quantization range and plot the quantized tensor
    q_min, q_max = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    plot_matrix(quantized_tensor, axes[1], f'{n_bits}-bit Linear Quantized Tensor', vmin=q_min, vmax=q_max, cmap='coolwarm')

    # Plot the de-quantized tensors
    plot_matrix(dequantized_tensor, axes[2], 'Dequantized Tensor', cmap='coolwarm')

    # Get the quantization errors
    q_error_tensor = abs(original_tensor - dequantized_tensor)
    plot_matrix(q_error_tensor, axes[3], 'Quantization Error Tensor', cmap=ListedColormap(['white']))

    fig.tight_layout()
    plt.show()


def Calc_MSE(original_tensor, dequantized_tensor):
    return (dequantized_tensor-original_tensor).square().mean()


def linear_q_with_s_and_z(tensor,s,z,dtype = torch.int8):
    scaled_and_shifted_tensor = tensor/s + z
    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_max = torch.iinfo(dtype).max
    q_min = torch.iinfo(dtype).min

    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
    return q_tensor


def linear_dq_with_s_and_z(tensor,s,z,dtype = torch.int8):
    dq_tensor = s*(tensor.float()-z)
    return dq_tensor


def get_q_scale_and_zero_point(tensor,dtype = torch.int8):
    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max
    r_min = tensor.min().item()
    r_max = tensor.max().item()
    scale = (r_max-r_min)/(q_max-q_min)
    zero_point = int(round(q_min-r_min/scale))  
    if (zero_point < q_min): 
        zero_point = q_min
    if (zero_point > q_max):
        zero_point = q_max
    return scale, zero_point


def linear_quantization(tensor):
    scale,zero_point = get_q_scale_and_zero_point(tensor)

    quantized_tensor = linear_q_with_s_and_z(tensor,scale,zero_point)
    return quantized_tensor,scale,zero_point


def linear_dequantization(tensor,scale,zero_point,dtype = torch.int8):
    dq_tensor = linear_q_with_s_and_z(tensor,scale,zero_point)
    return dq_tensor


def get_q_scale_symmetric(tensor, dtype=torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max
    return r_max/q_max


def linear_quantization_symm(tensor):
    scale = get_q_scale_symmetric(tensor)
    zero_point = 0
    q_tensor = linear_q_with_s_and_z(tensor,scale,zero_point)
    return q_tensor,scale


def linear_dequantization_symm(tensor,s,dtype = torch.int8):
    dq_tensor = linear_dq_with_s_and_z(tensor,s,0)
    return dq_tensor


def w8_a16_forward(input, weight, scale, bias = None):
    weight = weight.to(dtype = input.dtype)
    if (bias == None): 
        return torch.nn.functional.linear(input,weight)*scale
    return torch.nn.functional.linear(input,weight)*scale + bias


class W8A16LinearLayer(nn.Module):
    def __init__(self,input_features,output_features, bias = True, dtype = torch.float32):
        super().__init__()

        self.register_buffer('int8_weights', torch.randint(-128,127, (output_features,input_features)).to(torch.int8))
        self.register_buffer('scales', torch.randn((output_features),dtype= dtype))
        if bias: 
            self.register_buffer('bias', torch.randn((1,output_features), dtype = dtype))
        else:
            self.bias = None

    def quantize(self,weights):
        r_max = weights.clone().abs().max(dim=-1).values
        q_max = torch.iinfo(torch.int8).max
        scales = r_max/q_max
        scales = scales.to(weights.dtype)
        self.scales = scales
        self.int8_weights = torch.round(weights/self.scales.unsqueeze(1)).to(torch.int8)

    def forward(self, input):
        return w8_a16_forward(input, self.int8_weights, self.scales, self.bias)
        

def replace_linear_with_target(module,target_class, excluded_module):
    for name,child in module.named_children():
        if (isinstance(child, nn.Linear) and not any([x==name for x in excluded_module])):
            old_bias = child.bias
            new_module = target_class(child.in_features, 
                                      child.out_features, 
                                      old_bias is not None, 
                                      child.weight.dtype)
            setattr(module, name, new_module)
            if old_bias != None:
                getattr(module,name).bias = old_bias
        else:
            replace_linear_with_target(child,target_class,excluded_module)


def replace_linear_with_target_and_quantize(module,target_class, excluded_module):
    for name,child in module.named_children():
        if (isinstance(child, nn.Linear) and not any([x==name for x in excluded_module])):
            old_bias = child.bias
            old_wt = child.weight
            new_module = target_class(child.in_features, 
                                      child.out_features, 
                                      old_bias is not None, 
                                      child.weight.dtype)
            new_module.quantize(old_wt)
            setattr(module, name, new_module)

            # print("Before Wt: ", old_wt)
            # print("After Wt: ", getattr(module,name).int8_weights, '\n')
            
            if old_bias != None:
                getattr(module,name).bias = old_bias
            
        else:
            replace_linear_with_target_and_quantize(child,target_class,excluded_module)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(model, pil_img, results):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    scores, labels, boxes = results["scores"], results["labels"], results["boxes"]
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()