import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.cifar_resnet import resnet as resnet_cifar

def compute_spectral_norm(layer, input_shape):
    """
    Compute the spectral norm of a given layer using power iteration.
    
    For a Conv2d layer, a random input tensor is generated, normalized, and then 
    power iteration is performed by alternately applying the layer and its transposed
    operation (conv_transpose2d). For a Linear layer, a similar power iteration is used.
    
    Parameters:
        layer (nn.Module): The neural network layer (nn.Conv2d or nn.Linear).
        input_shape (tuple): The shape of the input tensor (e.g., (1, 3, 32, 32)).
        
    Returns:
        float: The estimated spectral norm of the layer.
    """
    if isinstance(layer, nn.Conv2d):
        # Generate a random input and normalize it.
        x = torch.randn(input_shape).to(next(layer.parameters()).device)
        x = x / torch.norm(x)
        
        # Power iteration loop.
        for _ in range(10):
            # Forward pass through the layer.
            y = layer(x)
            y_norm = torch.norm(y)
            y = y / y_norm
            
            # Backward pass using conv_transpose2d.
            x = F.conv_transpose2d(
                y, layer.weight, bias=None, stride=layer.stride,
                padding=layer.padding, output_padding=0,
                groups=layer.groups, dilation=layer.dilation
            )
            x_norm = torch.norm(x)
            x = x / x_norm
        
        spectral_norm = y_norm / x_norm
        return spectral_norm.item()
    
    elif isinstance(layer, nn.Linear):
        weight = layer.weight.data
        # Initialize random vector for power iteration.
        u = torch.randn(weight.size(1), 1).to(weight.device)  # shape (in_features, 1)
        u = u / torch.norm(u)
        for _ in range(10):
            v = weight @ u  # shape (out_features, 1)
            v = v / torch.norm(v)
            u = weight.t() @ v  # shape (in_features, 1)
            u = u / torch.norm(u)
        spectral_norm = torch.norm(weight @ u)
        return spectral_norm.item()
    
    else:
        # For layers that are not Conv2d or Linear, return a default value.
        return 1.0

def compute_batchnorm_lipschitz(layer):
    """
    Compute the Lipschitz constant of a BatchNorm2d layer.
    
    If the BatchNorm layer uses affine transformation (i.e., has learnable parameters), 
    the Lipschitz constant is given by the maximum absolute value of gamma divided by the 
    square root of running variance plus epsilon. If affine=False, the Lipschitz constant is 1.
    
    Parameters:
        layer (nn.BatchNorm2d): The BatchNorm2d layer.
        
    Returns:
        float: The Lipschitz constant of the BatchNorm layer.
    """
    if not layer.affine:
        return 1.0  # No scaling applied.
    gamma = layer.weight.data
    running_var = layer.running_var.data
    eps = layer.eps
    lipschitz_constants = gamma / torch.sqrt(running_var + eps)
    return lipschitz_constants.abs().max().item()

def compute_module_lipschitz(module, input_shape):
    """
    Recursively compute the Lipschitz constant of a module.
    
    This function handles layers such as Conv2d, Linear, BatchNorm2d, ReLU, pooling layers,
    nn.Sequential containers, and residual blocks (detected via conv1 and conv2 attributes).
    It multiplies the Lipschitz constants of submodules sequentially.
    
    Parameters:
        module (nn.Module): The neural network module.
        input_shape (tuple): The input shape to the module.
        
    Returns:
        float: The overall Lipschitz constant of the module.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        return compute_spectral_norm(module, input_shape)
    elif isinstance(module, nn.BatchNorm2d):
        return compute_batchnorm_lipschitz(module)
    elif isinstance(module, nn.ReLU):
        return 1.0  # ReLU is 1-Lipschitz.
    elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)):
        return 1.0  # Pooling layers are 1-Lipschitz.
    elif isinstance(module, nn.Sequential):
        lipschitz = 1.0
        x_shape = input_shape
        for submodule in module:
            sub_lip = compute_module_lipschitz(submodule, x_shape)
            lipschitz *= sub_lip
            x_shape = get_output_shape(submodule, x_shape)
        return lipschitz
    elif isinstance(module, nn.Module):
        # Handle residual blocks by checking for conv1 and conv2 attributes.
        if hasattr(module, 'conv1') and hasattr(module, 'conv2'):
            return compute_residual_block_lipschitz(module, input_shape)
        else:
            lipschitz = 1.0
            x_shape = input_shape
            for submodule in module.children():
                sub_lip = compute_module_lipschitz(submodule, x_shape)
                lipschitz *= sub_lip
                x_shape = get_output_shape(submodule, x_shape)
            return lipschitz
    else:
        return 1.0  # Default Lipschitz constant for unsupported modules.

def compute_residual_block_lipschitz(block, input_shape):
    """
    Compute the Lipschitz constant of a residual block.
    
    For a residual block with a main path (typically with conv1, bn1, relu, conv2, bn2)
    and a skip connection (either identity or a downsampling module), the overall Lipschitz
    constant is the sum of the Lipschitz constant of the main path and the skip connection.
    
    Parameters:
        block (nn.Module): The residual block.
        input_shape (tuple): The input shape to the block.
        
    Returns:
        float: The Lipschitz constant of the residual block.
    """
    # Compute Lipschitz constant of the main path.
    x_shape = input_shape
    lipschitz_main = 1.0
    for layer in [block.conv1, block.bn1, block.relu, block.conv2, block.bn2]:
        layer_lip = compute_module_lipschitz(layer, x_shape)
        lipschitz_main *= layer_lip
        x_shape = get_output_shape(layer, x_shape)
    
    # Compute Lipschitz constant of the skip connection.
    if hasattr(block, 'downsample') and block.downsample is not None:
        lipschitz_skip = compute_module_lipschitz(block.downsample, input_shape)
    else:
        lipschitz_skip = 1.0  # Identity mapping.
    
    # The output of the residual block is the sum of the main path and skip connection.
    lipschitz_block = lipschitz_main + lipschitz_skip  # Sum due to addition.
    # A final ReLU is applied after the addition.
    lipschitz_block *= 1.0  # ReLU is 1-Lipschitz.
    return lipschitz_block

def get_output_shape(layer, input_shape):
    """
    Compute the output shape of a layer given an input shape.
    
    This function performs a forward pass with a zero tensor (with no gradient)
    through the layer to obtain its output shape. It handles both single layers
    and sequential containers.
    
    Parameters:
        layer (nn.Module): The layer or module.
        input_shape (tuple): The input shape, e.g., (batch_size, channels, height, width).
        
    Returns:
        torch.Size: The output shape produced by the layer.
    """
    with torch.no_grad():
        device = (next(layer.parameters()).device 
                  if any(p.requires_grad for p in layer.parameters()) 
                  else torch.device('cpu'))
        x = torch.zeros(*input_shape).to(device)
        if isinstance(layer, nn.Sequential):
            for sublayer in layer:
                x = sublayer(x)
        else:
            if isinstance(layer, nn.Linear) and x.dim() > 2:
                x = x.view(x.size(0), -1)
            x = layer(x)
        return x.shape

def load_checkpoint_and_compute_pub(checkpoint_path, input_shape=(1, 3, 32, 32)):
    """
    Load a ResNet checkpoint and compute the Product Upper Bound (PUB) of its Lipschitz constant.
    
    This function instantiates a ResNet model (from cifar_resnet), loads the checkpoint, and 
    computes the overall Lipschitz constant of the network by recursively multiplying the Lipschitz
    constants of its submodules.
    
    Parameters:
        checkpoint_path (str): Path to the checkpoint file.
        input_shape (tuple): The shape of the input tensor (default is (1, 3, 32, 32) for CIFAR-10).
    
    Returns:
        float: The estimated Lipschitz constant (PUB) of the network.
    """
    # Instantiate the ResNet model.
    model = resnet_cifar(depth=110, num_classes=10)
    model.eval()  # Set model to evaluation mode.

    # Load checkpoint.
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    # Remove prefixes if necessary.
    new_state_dict = {k.partition(".")[-1]: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Compute the overall Lipschitz constant for the model.
    lipschitz_constant = compute_module_lipschitz(model, input_shape)
    return lipschitz_constant

if __name__ == "__main__":
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Compute the Product Upper Bound (PUB) of the Lipschitz constant for a ResNet model.")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the ResNet checkpoint file.")
    parser.add_argument("--input_size", type=int, default=32,
                        help="Input image size (default: 32 for CIFAR-10).")
    args = parser.parse_args()

    # Define input shape: batch size 1, 3 channels, image size provided.
    input_shape = (1, 3, args.input_size, args.input_size)
    
    # Compute the PUB.
    pub = load_checkpoint_and_compute_pub(args.checkpoint_path, input_shape)
    pub_formatted = "{:.2e}".format(pub)
    print(f"Estimated Lipschitz constant (PUB): {pub_formatted}")
