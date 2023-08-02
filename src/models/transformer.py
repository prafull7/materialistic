import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

activations = {}


def print_min_max(x):
    print(x.min(), x.max(), x.shape)

def display_embedding(embedding):
    print_min_max(embedding)
    plt.figure()
    plt.imshow(embedding)    

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output
    return hook

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        x = x.transpose(self.dim0, self.dim1)
        return x
    
class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features))

    def forward(self, x, global_token=None):
        if global_token is None:
            global_token = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index :])
        
        features = torch.cat((x[:, self.start_index :], global_token), -1)
        return self.project(features)
    
def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "project":
        readout_oper = [
            ProjectReadout(vit_features, start_index) for out_feat in features
        ]
    else:
        assert (
            False
        ), "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"

    return readout_oper

class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()
        
    def forward(self, x):
        # size is the square root of the number of tokens
        size = int(x.shape[1] ** 0.5)
        spatial_transform =  nn.Sequential(Transpose(1, 2), 
                                            nn.Unflatten(2, torch.Size([size , size])))
        x = spatial_transform(x)
        return x
    
def local_pixel_coord(x, y, s):
    patch_idx_x = torch.div(x, s, rounding_mode="trunc")
    patch_idx_y = torch.div(y, s, rounding_mode="trunc")

    patch_center_x = patch_idx_x * s + (s - 1.)/2.
    patch_center_y = patch_idx_y * s + (s - 1.)/2.

    out_x = (x - patch_center_x) / ((s - 1) / 2 + 1e-4)
    out_y = (y - patch_center_y) / ((s - 1) / 2 + 1e-4)
    return out_x, out_y

class ReferenceEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ReferenceEmbedding, self).__init__()
        self.reference_feature_extractor = nn.Sequential(nn.Linear(in_channels + 2, in_channels * 2),
                                                         nn.ReLU(),
                                                         nn.Linear(in_channels * 2, out_channels))
        
        self.stride = stride
    
    def forward(self, embeddings, reference_locations):
        # embeddings: (B, C, H, W)
        # reference_locations: (B, 2)
        
        # print("Embedding shape:", embeddings.shape, "\tStride:", self.stride)
        reference_locations_patch = torch.div(reference_locations, self.stride, rounding_mode="trunc")
        reference_embeddings = embeddings[range(embeddings.shape[0]), :, reference_locations_patch[:, 0].type(torch.long), reference_locations_patch[:, 1].type(torch.long)]
        
        local_coordinate = local_pixel_coord(reference_locations[:, 0].type(torch.long), reference_locations[:, 1].type(torch.long), self.stride)
        local_coordinate = torch.stack(local_coordinate, dim=1).cuda()
        # print("Reference location patch:", reference_locations_patch, "\tLocal coordinate:", local_coordinate)
        
        # concatenate the local coordinate with the reference embedding
        # print(reference_locations.shape, reference_embeddings.shape)
        reference_embeddings = torch.cat([reference_embeddings, local_coordinate], dim=1)
        reference_embeddings = self.reference_feature_extractor(reference_embeddings)
        return reference_embeddings


# cross attention CrossAttentionWithReferenceEmbeddingr with MLPs as the attention mechanism and 
# a reference embedding layer
class CrossAttentionWithReferenceEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=16, stride=1):
        super(CrossAttentionWithReferenceEmbedding, self).__init__()
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head_channels = out_channels // num_heads
        assert self.head_channels * num_heads == out_channels, "out_channels must be divisible by num_heads"

        self.Unflatten = Unflatten()
        self.LayerNorm = nn.LayerNorm(in_channels)
        self.get_reference_embedding = ReferenceEmbedding(in_channels, in_channels, stride=stride)
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        self.out = nn.Linear(out_channels, out_channels)
        self.out_norm = nn.LayerNorm(out_channels)
        
    def forward(self, embeddings, reference_locations, reference_embeddings=None):
        # embeddings: (B, N, C)
        # reference_locations: (B, 2)

        B, N, C = embeddings.shape
        embeddings = self.LayerNorm(embeddings)
        if reference_embeddings is None:
            reference_embeddings = self.get_reference_embedding(self.Unflatten(embeddings), reference_locations) # (B, C)
        
        q = self.query(reference_embeddings).reshape(B, 1, self.num_heads, self.head_channels).permute(0, 2, 1, 3) # (B, num_heads, N, head_channels)
        k = self.key(embeddings).reshape(B, N, self.num_heads, self.head_channels).permute(0, 2, 1, 3) # (B, num_heads, N, head_channels)
        v = self.value(embeddings).reshape(B, N, self.num_heads, self.head_channels).permute(0, 2, 1, 3) # (B, num_heads, N, head_channels)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1) # (B, num_heads, #Q=1, #K=HW) 
        # sigmoid
        attn = torch.sigmoid(attn)
        
        x = (attn.transpose(-1, -2) * v).permute(0, 2, 1, 3).reshape(B, N, self.out_channels)
        x = self.out(x)
        x = self.out_norm(x) # (B, N, C)
        return x, attn, q, k, v, reference_embeddings
    
class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()
        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=self.groups,
        )

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)

        return self.skip_add.add(out, x)
    
class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        align_corners=True,
        upsample=True
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.align_corners = align_corners
        out_features = features

        self.groups = 1

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation)
        self.upsample = upsample
        self.layer_norm_1 = nn.LayerNorm(out_features)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        if self.upsample:
            output = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
            )

        output = self.out_conv(output)
        return output

def init_weights(self, init_type='xavier_uniform', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=1.0)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    
    self.apply(init_func)        
    # propagate to children
    for n, m in self.named_children():
        # if hasattr(m, 'init_weights'):
        # print("Initializating ", n)
        m.apply(init_weights)

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        # Load DINO model        
        self.dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8', pretrained=True)
        self.dino_model.eval()

        # Add hooks to get activations
        hooks = [2, 5, 8, 11]
        self.dino_model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
        self.dino_model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
        self.dino_model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
        self.dino_model.blocks[hooks[3]].register_forward_hook(get_activation("4"))
        self.activations = activations
        
        # Spatial processing
        features = [96, 192, 384, 768]
        vit_features = 768
        attn_out_feats = 256
        out_channels = 256
                
        self.spatial_transform_1 = nn.Sequential(ProjectReadout(vit_features, 1),
                                    Unflatten(),
                                    nn.Conv2d(
                                        in_channels=vit_features,
                                        out_channels=attn_out_feats,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                                    )
        
        
        self.spatial_transform_2 = nn.Sequential(ProjectReadout(vit_features, 1),
                                            Unflatten(),
                                            nn.Conv2d(in_channels=vit_features,
                                                    out_channels=attn_out_feats,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0),
                                            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                            )

        self.spatial_transform_3 = nn.Sequential(ProjectReadout(vit_features, 1),
                                            Unflatten(),
                                            nn.Conv2d(in_channels=vit_features,
                                                    out_channels=attn_out_feats,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0))

        self.spatial_transform_4 = nn.Sequential(ProjectReadout(vit_features, 1),
                                            Unflatten(),
                                            nn.Conv2d(in_channels=vit_features,
                                                    out_channels=attn_out_feats,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0))

        # Cross attention
        self.cross_attention_1 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=2)
        self.cross_attention_2 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=4)
        self.cross_attention_3 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=8)
        self.cross_attention_4 = CrossAttentionWithReferenceEmbedding(attn_out_feats, out_channels, stride=8)

        self.fusion_1 =  FeatureFusionBlock_custom(out_channels,
                                            nn.ReLU(),
                                            align_corners=True)

        self.fusion_2 =  FeatureFusionBlock_custom(out_channels,
                                            nn.ReLU(),
                                            align_corners=True)

        self.fusion_3 =  FeatureFusionBlock_custom(out_channels,
                                                nn.ReLU(),
                                                align_corners=True)

        self.fusion_4 =  FeatureFusionBlock_custom(out_channels,
                                                nn.ReLU(),
                                                align_corners=True, upsample=False)

        # Output Conv
        self.out_conv = nn.Sequential(nn.Linear(256, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 128),
                                        nn.ReLU(),
                                        nn.Linear(128, 1))
        
        self.spatial_transform_1.apply(init_weights)
        self.spatial_transform_2.apply(init_weights)
        self.spatial_transform_3.apply(init_weights)
        self.spatial_transform_4.apply(init_weights)

        self.cross_attention_1.apply(init_weights)
        self.cross_attention_2.apply(init_weights)
        self.cross_attention_3.apply(init_weights)
        self.cross_attention_4.apply(init_weights)

        self.fusion_1.apply(init_weights)
        self.fusion_2.apply(init_weights)
        self.fusion_3.apply(init_weights)
        self.fusion_4.apply(init_weights)

        self.out_conv.apply(init_weights)
        
        
    def forward(self, x, reference_locations):
        B, C, H, W = x.shape
        with torch.no_grad():
            x = self.dino_model(x)
        
        # Get activations
        layer_1 = self.activations["1"]
        layer_2 = self.activations["2"]
        layer_3 = self.activations["3"]
        layer_4 = self.activations["4"]
                
        # readout operation which combines the global CLS token with the local context tokens
        context_embeddings_1 = self.spatial_transform_1(layer_1)
        context_embeddings_2 = self.spatial_transform_2(layer_2)
        context_embeddings_3 = self.spatial_transform_3(layer_3)
        context_embeddings_4 = self.spatial_transform_4(layer_4)
        
        # Cross attention
        cross_attention_layer_1, attn_map_1, q1, k1, v1, _ = self.cross_attention_1(context_embeddings_1.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        cross_attention_layer_2, attn_map_2, q2, k2, v2, _ = self.cross_attention_2(context_embeddings_2.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        cross_attention_layer_3, attn_map_3, q3, k3, v3, _ = self.cross_attention_3(context_embeddings_3.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        cross_attention_layer_4, attn_map_4, q4, k4, v4, _ = self.cross_attention_4(context_embeddings_4.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        
        # Unflattening the spatial token maps at the different scales from different blocks
        cross_attention_layer_1_unflattened = cross_attention_layer_1.permute(0, 2, 1).reshape(B, -1, 256, 256)
        cross_attention_layer_2_unflattened = cross_attention_layer_2.permute(0, 2, 1).reshape(B, -1, 128, 128)
        cross_attention_layer_3_unflattened = cross_attention_layer_3.permute(0, 2, 1).reshape(B, -1, 64, 64)
        cross_attention_layer_4_unflattened = cross_attention_layer_4.permute(0, 2, 1).reshape(B, -1, 64, 64)
                      
        # Fusing the spatial token maps at the different scales from different blocks
        path4 = self.fusion_4(cross_attention_layer_4_unflattened)
        path3 = self.fusion_3(cross_attention_layer_3_unflattened, path4)
        path2 = self.fusion_2(cross_attention_layer_2_unflattened, path3)
        path1 = self.fusion_1(cross_attention_layer_1_unflattened, path2)
        
        prediction = self.out_conv(path1.permute(0, 2, 3, 1)).reshape(B, H, W)
        return torch.sigmoid(prediction), path1, path2, path3, path4, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, layer_1, layer_2, layer_3, layer_4
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def check_attentions(self, x, reference_locations):
        B, C, H, W = x.shape
        with torch.no_grad():
            x = self.dino_model(x)
        
        # print("activations", self.activations.keys())
        # Get activations
        layer_1 = self.activations["1"]
        layer_2 = self.activations["2"]
        layer_3 = self.activations["3"]
        layer_4 = self.activations["4"]
                
        # readout operation which combines the global CLS token with the local context tokens
        context_embeddings_1 = self.spatial_transform_1(layer_1)
        context_embeddings_2 = self.spatial_transform_2(layer_2)
        context_embeddings_3 = self.spatial_transform_3(layer_3)
        context_embeddings_4 = self.spatial_transform_4(layer_4)
        
        # print("context_embeddings_1", context_embeddings_1.shape)
        
        # Cross attention
        cross_attention_layer_1, attn_map_1, q1, k1, v1, ref_1 = self.cross_attention_1(context_embeddings_1.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        cross_attention_layer_2, attn_map_2, q2, k2, v2, ref_2 = self.cross_attention_2(context_embeddings_2.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        cross_attention_layer_3, attn_map_3, q3, k3, v3, ref_3 = self.cross_attention_3(context_embeddings_3.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        cross_attention_layer_4, attn_map_4, q4, k4, v4, ref_4 = self.cross_attention_4(context_embeddings_4.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations)
        
        # create a dictionary of the attention maps and reference embeddings
        result = {"layer_1": layer_1,
                  "layer_2": layer_2,
                  "layer_3": layer_3,
                  "layer_4": layer_4,
                  "cross_attention_layer_1": cross_attention_layer_1, 
                  "cross_attention_layer_2": cross_attention_layer_2, 
                  "cross_attention_layer_3": cross_attention_layer_3, 
                  "cross_attention_layer_4": cross_attention_layer_4, 
                  "ref_1": ref_1, 
                  "ref_2": ref_2, 
                  "ref_3": ref_3, 
                  "ref_4": ref_4}
        
        return result
    
    def cross_image_reference_output(self, image_1, image_2, reference_locations):
        # image_1 and image_2 are of shape (1, C, H, W)
        with torch.no_grad():
            result_image_1 = self.check_attentions(image_1, reference_locations)
            # print("result_image_1:", activations.keys())
            self.activations = activations
            
            layer_1_imgA = result_image_1["layer_1"]
            layer_2_imgA = result_image_1["layer_2"]
            layer_3_imgA = result_image_1["layer_3"]
            layer_4_imgA = result_image_1["layer_4"]
            
            cross_attention_layer_1 = result_image_1["cross_attention_layer_1"]
            cross_attention_layer_2 = result_image_1["cross_attention_layer_2"]
            cross_attention_layer_3 = result_image_1["cross_attention_layer_3"]
            cross_attention_layer_4 = result_image_1["cross_attention_layer_4"]
            
            ref_1 = result_image_1["ref_1"]
            ref_2 = result_image_1["ref_2"]
            ref_3 = result_image_1["ref_3"]
            ref_4 = result_image_1["ref_4"]
            
            B, C, H, W = image_2.shape
            with torch.no_grad():
                x = self.dino_model(image_2)
            
            # print(self.activations.keys())
            # Get activations
            layer_1 = self.activations["1"]
            layer_2 = self.activations["2"]
            layer_3 = self.activations["3"]
            layer_4 = self.activations["4"]

            # readout operation which combines the global CLS token with the local context tokens
            context_embeddings_1 = self.spatial_transform_1(layer_1)
            context_embeddings_2 = self.spatial_transform_2(layer_2)
            context_embeddings_3 = self.spatial_transform_3(layer_3)
            context_embeddings_4 = self.spatial_transform_4(layer_4)
            
            # print("context_embeddings_1.shape", context_embeddings_1.shape)
            
            # Cross attention
            cross_attention_layer_1, attn_map_1, q1, k1, v1, ref_1 = self.cross_attention_1(context_embeddings_1.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations, ref_1)
            cross_attention_layer_2, attn_map_2, q2, k2, v2, ref_2 = self.cross_attention_2(context_embeddings_2.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations, ref_2)
            cross_attention_layer_3, attn_map_3, q3, k3, v3, ref_3 = self.cross_attention_3(context_embeddings_3.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations, ref_3)
            cross_attention_layer_4, attn_map_4, q4, k4, v4, ref_4 = self.cross_attention_4(context_embeddings_4.permute(0, 2, 3, 1).reshape(B, -1, 256), reference_locations, ref_4)
            
            # Unflattening the spatial token maps at the different scales from different blocks
            cross_attention_layer_1_unflattened = cross_attention_layer_1.permute(0, 2, 1).reshape(B, -1, 256, 256)
            cross_attention_layer_2_unflattened = cross_attention_layer_2.permute(0, 2, 1).reshape(B, -1, 128, 128)
            cross_attention_layer_3_unflattened = cross_attention_layer_3.permute(0, 2, 1).reshape(B, -1, 64, 64)
            cross_attention_layer_4_unflattened = cross_attention_layer_4.permute(0, 2, 1).reshape(B, -1, 64, 64)
                        
            # Fusing the spatial token maps at the different scales from different blocks
            path4 = self.fusion_4(cross_attention_layer_4_unflattened)
            path3 = self.fusion_3(cross_attention_layer_3_unflattened, path4)
            path2 = self.fusion_2(cross_attention_layer_2_unflattened, path3)
            path1 = self.fusion_1(cross_attention_layer_1_unflattened, path2)
            
            prediction = self.out_conv(path1.permute(0, 2, 3, 1)).reshape(B, H, W)
            return torch.sigmoid(prediction), path1, path2, path3, path4, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, layer_1, layer_2, layer_3, layer_4
    
        
        