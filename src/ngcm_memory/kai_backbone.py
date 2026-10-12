"""
Copyright (c) 2025 MetaEarth-Lab
Licensed under the BSD 3-Clause License. Redistribution and use in source and binary forms, with or without modification, are permitted. See the LICENSE file in the project root for the full license text.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def trunc_normal_(tensor, mean=0., std=1.):
    """
    Sample from a truncated normal distribution in-place on the provided tensor.

    Args:
        tensor (torch.Tensor): Tensor whose values will be replaced by samples.
        mean (float): Mean of the (non-truncated) normal distribution.
        std (float): Standard deviation of the (non-truncated) normal distribution.

    Returns:
        torch.Tensor: Reference to the input tensor containing truncated samples.

    Raises:
        RuntimeError: If random sampling fails for the given tensor device.
    """
    with torch.no_grad():
        return tensor.normal_(mean, std).fmod_(2).mul_(std).add_(mean)

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable 2D convolution consisting of a depthwise and pointwise stage.

    Attributes:
        depthwise (nn.Conv2d): Channel-wise spatial convolution.
        pointwise (nn.Conv2d): 1x1 convolution that mixes channel information.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        """
        Initialize depthwise and pointwise convolutional layers.

        Args:
            in_channels (int): Number of incoming feature channels.
            out_channels (int): Number of output feature channels.
            kernel_size (int): Spatial kernel size for the depthwise convolution.
            stride (int): Stride applied by the depthwise convolution.
            padding (int): Padding applied by the depthwise convolution.

        Returns:
            None: Layers are stored as module attributes.

        Raises:
            ValueError: If `in_channels` is not positive.
        """
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        """
        Apply depthwise then pointwise convolutions.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Tensor with shape (N, C_out, H, W).

        Raises:
            RuntimeError: Propagated if convolution operations fail.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class PointwiseGroupConv(nn.Module):
    """
    Grouped pointwise convolution used for efficient channel mixing.

    Attributes:
        conv (nn.Conv2d): Underlying grouped 1x1 convolution.
    """

    def __init__(self, in_channels, out_channels, groups=6):
        """
        Configure the grouped 1x1 convolution.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            groups (int): Number of groups used in the convolution.

        Returns:
            None: Initializes internal Conv2d module.

        Raises:
            AssertionError: If channel counts are not divisible by `groups`.
        """
        super(PointwiseGroupConv, self).__init__()
        assert in_channels % groups == 0 and out_channels % groups == 0, "Channels must be divisible by groups"
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, bias=False)
    
    def forward(self, x):
        """
        Apply grouped 1x1 convolution.

        Args:
            x (torch.Tensor): Input tensor shaped (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor shaped (N, C_out, H, W).

        Raises:
            RuntimeError: If the convolution cannot be applied.
        """
        return self.conv(x)

class GeoCyclicPadding(nn.Module):
    """
    Apply cyclical padding along longitude and mirrored padding along latitude.

    Attributes:
        padx (int): Amount of padding to apply along the longitudinal axis.
        pady (int): Amount of padding to apply along the latitudinal axis.
    """

    def __init__(self, pad_width):
        """
        Store padding widths for geocyclic padding.

        Args:
            pad_width (Union[int, Tuple[int, int]]): Padding along latitude and longitude.

        Returns:
            None: Padding settings are stored on the instance.

        Raises:
            ValueError: If `pad_width` is neither int nor tuple.
        """
        super(GeoCyclicPadding, self).__init__()
        if isinstance(pad_width, int):
            self.padx = self.pady = pad_width
        elif isinstance(pad_width, tuple):
            self.pady, self.padx = pad_width
        else:
            raise ValueError("pad_width must be an int or a tuple of two ints.")
        
    def forward(self, x):
        """
        Apply geocyclic padding to the tensor.

        Args:
            x (torch.Tensor): Input tensor with shape (N, C, H, W).

        Returns:
            torch.Tensor: Tensor padded in longitude and latitude.

        Raises:
            RuntimeError: If concatenation operations fail due to dimension mismatch.
        """
        m = x.shape[-1] // 2
        if self.pady > 0:
            top = torch.cat([x[..., :self.pady, m:], x[..., :self.pady, :m]], dim=-1)
            bot = torch.cat([x[..., -self.pady:, m:], x[..., -self.pady:, :m]], dim=-1)
            tb = torch.cat([torch.flip(top, [-2]), x, torch.flip(bot, [-2])], dim=-2)
        else:
            tb = x
        out = torch.cat([tb[..., -self.padx:], tb, tb[..., :self.padx]], dim=-1) if self.padx > 0 else tb
        return out

class InceptionDWConv2d(nn.Module):
    """
    Inception-style depthwise convolution with multiple directional branches.

    Attributes:
        remaining_channels (int): Channels bypassing the depthwise branches.
        split_indexes (List[int]): Channel splits for identity and branch paths.
        dwconv_hw (nn.Conv2d): Square depthwise convolution branch.
        dwconv_w (nn.Conv2d): Horizontal band depthwise convolution branch.
        dwconv_h (nn.Conv2d): Vertical band depthwise convolution branch.
        pad_hw (GeoCyclicPadding): Padding for square branch.
        pad_w (GeoCyclicPadding): Padding for horizontal branch.
        pad_h (GeoCyclicPadding): Padding for vertical branch.
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        """
        Configure channel splits and padding-aware convolutions.

        Args:
            in_channels (int): Number of incoming channels.
            square_kernel_size (int): Kernel size for the square branch.
            band_kernel_size (int): Kernel size for band branches.
            branch_ratio (float): Fraction of channels allocated to each branch.

        Returns:
            None: Initializes convolutional submodules.

        Raises:
            ValueError: If `branch_ratio` allocates more channels than available.
        """
        super(InceptionDWConv2d, self).__init__()
        # Calculate number of channels per branch
        gc = max(1, int(in_channels * branch_ratio))
        total_branch_channels = 3 * gc
        self.remaining_channels = in_channels - total_branch_channels
        if self.remaining_channels < 0:
            raise ValueError(f"Invalid branch ratio for {in_channels} channels. Adjust branch_ratio.")
        
        # The split sizes: [remaining_channels, gc, gc, gc] (sums to in_channels)
        self.split_indexes = [self.remaining_channels, gc, gc, gc]
        if sum(self.split_indexes) != in_channels:
            self.split_indexes[-1] += in_channels - sum(self.split_indexes)
        
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=0, groups=gc)
        self.dwconv_w  = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=0, groups=gc)
        self.dwconv_h  = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=0, groups=gc)
        
        self.pad_hw = GeoCyclicPadding(pad_width=square_kernel_size // 2)
        self.pad_w  = GeoCyclicPadding(pad_width=(0, band_kernel_size // 2))
        self.pad_h  = GeoCyclicPadding(pad_width=(band_kernel_size // 2, 0))
    
    def forward(self, x):
        """
        Apply the directional depthwise convolutions and concatenate results.

        Args:
            x (torch.Tensor): Input tensor shaped (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor with preserved channel count.

        Raises:
            RuntimeError: If tensor splitting fails due to inconsistent shapes.
        """
        x_splits = torch.split(x, self.split_indexes, dim=1)
        x_id, x_hw, x_w, x_h = x_splits
        
        x_hw = self.dwconv_hw(self.pad_hw(x_hw))
        x_w  = self.dwconv_w(self.pad_w(x_w))
        x_h  = self.dwconv_h(self.pad_h(x_h))
        
        return torch.cat([x_id, x_hw, x_w, x_h], dim=1)

class LayerNorm(nn.Module):
    """
    Layer normalization supporting both channel-last and channel-first layouts.

    Attributes:
        weight (nn.Parameter): Learnable scaling factors.
        bias (nn.Parameter): Learnable bias offsets.
        eps (float): Numerical stability constant.
        data_format (str): Expected layout of incoming tensors.
        normalized_shape (Tuple[int]): Shape used for normalization statistics.
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        """
        Initialize layer normalization parameters.

        Args:
            normalized_shape (int): Size of the channel dimension to normalize.
            eps (float): Numerical stability constant.
            data_format (str): Either "channels_last" or "channels_first".

        Returns:
            None: Parameters are registered as module attributes.

        Raises:
            ValueError: If an unsupported data_format is supplied.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps    = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        """
        Normalize the input tensor across the feature dimension.

        Args:
            x (torch.Tensor): Input tensor in either channel-last or channel-first layout.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as `x`.

        Raises:
            NotImplementedError: If `data_format` is neither supported option.
        """
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # Permute to [N, H, W, C], normalize, then permute back
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)

class Block(nn.Module):
    """
    Building block combining Inception-style depthwise convs and grouped pointwise convs.

    Attributes:
        dwconv (InceptionDWConv2d): Depthwise convolutional unit.
        norm (LayerNorm): Layer normalization in channels-first format.
        pwconv1 (PointwiseGroupConv): First pointwise convolution expanding channels.
        act (nn.Module): Non-linear activation function.
        pwconv2 (PointwiseGroupConv): Second pointwise convolution reducing channels.
        gamma (Optional[nn.Parameter]): Layer-scale parameter for residual branch.
        drop_path (nn.Module): Drop-path regularizer or identity mapping.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, pw_groups=1):
        """
        Initialize convolutional branches and residual scaling.

        Args:
            dim (int): Number of channels processed by the block.
            drop_path (float): Drop probability for stochastic depth.
            layer_scale_init_value (float): Initial value for residual scaling parameter.
            pw_groups (int): Group count for pointwise convolutions.

        Returns:
            None: Layers are stored as attributes.

        Raises:
            ValueError: If `dim` is not positive.
        """
        super().__init__()
        self.dwconv = InceptionDWConv2d(dim, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125)
        self.norm   = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        
        self.pwconv1 = PointwiseGroupConv(dim, 4 * dim, groups=pw_groups)
        self.act     = nn.GELU()
        self.pwconv2 = PointwiseGroupConv(4 * dim, dim, groups=pw_groups)
        
        self.gamma   = nn.Parameter(layer_scale_init_value * torch.ones(dim)) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity() if drop_path == 0 else nn.Dropout(drop_path)
    
    def forward(self, x):
        """
        Apply depthwise separable transformations with residual connection.

        Args:
            x (torch.Tensor): Input tensor of shape (N, dim, H, W).

        Returns:
            torch.Tensor: Output tensor with the same shape as `x`.

        Raises:
            ValueError: If the output shape differs from the input shape.
        """
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        
        if x.shape != input.shape:
            raise ValueError(f"Shape mismatch in Block! input: {input.shape}, x: {x.shape}")
        x = input + self.drop_path(x)
        return x

class InceptionNeXt(nn.Module):
    """
    Scale-invariant Inception-style backbone for geophysical forecasting.

    Attributes:
        downsample_layers (nn.ModuleList): Stacked stem and stage transition layers.
        stages (nn.ModuleList): Lists of residual blocks per resolution stage.
        head (nn.Sequential): Final convolutional projection into output channels.
    """

    def __init__(self, in_chans=67, num_classes=67, depths=[3, 3, 15, 3], 
                 dims=[48, 96, 192, 288], drop_path_rate=0.):
        """
        Build the multi-stage architecture with shared resolution.

        Args:
            in_chans (int): Number of input channels.
            num_classes (int): Number of output channels/classes.
            depths (Sequence[int]): Block counts per stage.
            dims (Sequence[int]): Channel widths per stage.
            drop_path_rate (float): Drop-path rate applied to all blocks.

        Returns:
            None: Modules are composed and stored.

        Raises:
            ValueError: If lengths of `depths` or `dims` are invalid.
        """
        super().__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        
        # Stem layer: Convert input channels (67) to dims[0] (48)
        stem = nn.Sequential(
            DepthwiseSeparableConv(in_chans, dims[0], kernel_size=3, stride=1, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        
        # Downsample layers for subsequent stages (keep spatial dims with stride=1)
        for i in range(1, 4):
            downsample = nn.Sequential(
                LayerNorm(dims[i-1], eps=1e-6, data_format="channels_first"),
                DepthwiseSeparableConv(dims[i-1], dims[i], kernel_size=3, stride=1, padding=1)
            )
            self.downsample_layers.append(downsample)
        
        # Build stages with Blocks for each resolution stage.
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=drop_path_rate) for _ in range(depths[i])]
            )
            self.stages.append(stage)
        
        # Head: Map final features (dims[-1] channels) to num_classes channels.
        self.head = nn.Sequential(
            nn.Conv2d(dims[-1], dims[-1], kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[-1], num_classes, kernel_size=1)
        )
    
    def forward_features(self, x):
        """
        Apply stem and stage stacks to extract feature maps.

        Args:
            x (torch.Tensor): Input tensor with shape (N, in_chans, H, W).

        Returns:
            torch.Tensor: Feature tensor after the final stage.

        Raises:
            RuntimeError: If intermediate modules fail.
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x
    
    def forward(self, x):
        """
        Compute full forward pass including feature extraction and head projection.

        Args:
            x (torch.Tensor): Input tensor shaped (N, in_chans, H, W).

        Returns:
            torch.Tensor: Prediction tensor shaped (N, num_classes, H, W).

        Raises:
            RuntimeError: Propagated from feature or head modules.
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x

class KAI(nn.Module):
    """
    High-level wrapper around the InceptionNeXt core model.

    Attributes:
        in_chans (int): Number of expected input channels.
        out_chans (int): Number of predicted channels.
        core_model (InceptionNeXt): Backbone network producing forecasts.
    """

    def __init__(self, img_size=(72, 144), in_chans=67, out_chans=67, depths=[3, 3, 15, 3], dims=[48, 96, 192, 288], drop_path_rate=0.):
        """
        Instantiate the KAI forecasting model.

        Args:
            img_size (Tuple[int, int]): Spatial resolution of the input fields.
            in_chans (int): Number of input feature channels.
            out_chans (int): Number of output feature channels.
            depths (Sequence[int]): Block count per stage in the core model.
            dims (Sequence[int]): Channel widths per stage.
            drop_path_rate (float): Drop-path probability for the backbone.

        Returns:
            None: Initializes the composed InceptionNeXt backbone.

        Raises:
            ValueError: If provided configuration arguments are invalid.
        """
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        # Using the modified InceptionNeXt as the core model
        self.core_model = InceptionNeXt(in_chans=self.in_chans, num_classes=self.out_chans, depths=depths, dims=dims, drop_path_rate=drop_path_rate)

    def forward(self, x):
        """
        Generate predictions using the core InceptionNeXt backbone.

        Args:
            x (torch.Tensor): Input tensor with shape (N, in_chans, H, W).

        Returns:
            torch.Tensor: Output tensor with shape (N, out_chans, H, W).

        Raises:
            RuntimeError: If the underlying backbone forward pass fails.
        """
        x = self.core_model(x)
        return x


if __name__ == "__main__":
    # Example of creating a model and emphasizing specific channels
    model = KAI()  # Update this line if you need to pass any specific parameters
    sample = torch.randn(1, 67, 72, 144)  # Example input tensor
    result = model(sample)
    print(result.shape)
    print(torch.norm(result))
