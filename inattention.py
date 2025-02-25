import torch
import torch.nn as nn
import torch.nn.functional as F


class InattentionDropout(nn.Module):
    """
    InattentionDropout: A novel approach that works in opposition to attention mechanisms.
    Instead of focusing on important features, it strategically "forgets" less important ones
    through a stacked dropout mechanism.
    
    Args:
        channels (int): Number of input channels
        dropout_prob (float, optional): Probability of dropping a feature. Default: 0.2
        num_stacks (int, optional): Number of stacked inattention mechanisms. Default: 3
        
    Shape:
        - Input: (batch_size, channels, height, width)
        - Output: Same as input
        
    Examples:
        >>> # With 64 input channels
        >>> m = InattentionDropout(64)
        >>> input = torch.randn(20, 64, 32, 32)
        >>> output = m(input)  # During training
        >>> output = m(input, training=False)  # During evaluation
        
    Notes:
        - During training, the layer applies stacked dropout patterns weighted by learnable parameters
        - During evaluation, the layer acts as an identity function
        - The stacked nature allows learning different dropout patterns at different semantic levels
    """
    
    def __init__(self, channels, dropout_prob=0.2, num_stacks=3):
        super(InattentionDropout, self).__init__()
        self.dropout_prob = dropout_prob
        self.num_stacks = num_stacks
        
        # Learnable weights for each stack
        self.alpha = nn.Parameter(torch.ones(num_stacks) / num_stacks)
        
        # Create stack of inattention mechanisms
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.Sigmoid()
            ) 
            for _ in range(num_stacks)
        ])
    
    def forward(self, x, training=True):
        """
        Forward pass of the InattentionDropout layer.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, channels, height, width)
            training (bool, optional): Whether in training mode. Default: True
            
        Returns:
            Tensor: Output tensor of the same shape as input
        """
        if not training or self.dropout_prob == 0:
            return x
            
        # Normalize alphas to sum to 1
        alpha_norm = F.softmax(self.alpha, dim=0)
        
        # Initialize combined mask
        batch_size, channels, height, width = x.shape
        combined_mask = torch.ones_like(x)
        
        for k in range(self.num_stacks):
            # Generate attention-based retention probabilities
            retention_prob = self.attention_layers[k](x)
            
            # Create binary dropout mask
            mask = torch.bernoulli(retention_prob * (1 - self.dropout_prob) + self.dropout_prob)
            
            # Weight the mask by alpha and combine
            combined_mask = combined_mask * (alpha_norm[k] * mask + (1 - alpha_norm[k]))
        
        # Apply the mask and scale by inverse dropout probability
        return x * combined_mask / (1 - self.dropout_prob)
    
    def extra_repr(self) -> str:
        """Returns a string with extra information about the layer."""
        return f'channels={self.attention_layers[0][0].in_channels}, ' \
               f'dropout_prob={self.dropout_prob}, ' \
               f'num_stacks={self.num_stacks}'


class StackedInattentionDropout(nn.Module):
    """
    A higher-level implementation of InattentionDropout that allows for easy creation
    of multiple stacked inattention layers with different configurations.
    
    Args:
        channels_list (list): List of channel sizes for each inattention layer
        dropout_prob (float or list, optional): Dropout probability or list of probabilities for each layer
        num_stacks (int or list, optional): Number of stacks or list of stack numbers for each layer
        
    Example:
        >>> # Create a stacked inattention with different configurations
        >>> m = StackedInattentionDropout(
        ...     channels_list=[64, 128, 256],
        ...     dropout_prob=[0.2, 0.3, 0.4],
        ...     num_stacks=[3, 4, 5]
        ... )
    """
    
    def __init__(self, channels_list, dropout_prob=0.2, num_stacks=3):
        super(StackedInattentionDropout, self).__init__()
        
        # Convert scalar values to lists if necessary
        if not isinstance(dropout_prob, (list, tuple)):
            dropout_prob = [dropout_prob] * len(channels_list)
        if not isinstance(num_stacks, (list, tuple)):
            num_stacks = [num_stacks] * len(channels_list)
            
        assert len(channels_list) == len(dropout_prob) == len(num_stacks), \
            "channels_list, dropout_prob, and num_stacks must have the same length"
            
        # Create the stack of inattention layers
        self.layers = nn.ModuleList([
            InattentionDropout(channels, dp, ns)
            for channels, dp, ns in zip(channels_list, dropout_prob, num_stacks)
        ])
        
    def forward(self, x, training=True):
        """
        Forward pass through all inattention layers.
        
        Args:
            x (Tensor): Input tensor
            training (bool, optional): Whether in training mode. Default: True
            
        Returns:
            Tensor: Output tensor after applying all inattention layers
        """
        for layer in self.layers:
            x = layer(x, training)
        return x
