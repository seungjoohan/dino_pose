import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=16, dropout=0.1):
        super(LoRALayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scaling = alpha / r
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.zeros(in_features, r))
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # LoRAoutput
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling
    

class LoRAAttention(nn.Module):
    def __init__(self, original_attention, r=8, alpha=16, dropout=0.1):
        super(LoRAAttention, self).__init__()
        self.original_attention = original_attention
        self.rank = r
        self.alpha = alpha

        # Get dimensions from original attention
        self.in_dim = original_attention.attention.query.in_features
        self.out_dim = original_attention.attention.query.out_features

        # Create LoRA layer for the attention output (full hidden dimension)
        self.lora_output = LoRALayer(self.in_dim, self.in_dim, r, alpha, dropout)

        # Freeze original attention parameters
        for param in self.original_attention.parameters():
            param.requires_grad = False

    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        attention_outputs = self.original_attention(hidden_states, head_mask, output_attentions)
        attention_output = attention_outputs[0]
        
        # Apply LoRA transformation
        lora_output = self.lora_output(attention_output)
        modified_output = attention_output + lora_output
        
        # Return in the same format as original attention
        if output_attentions:
            return (modified_output,) + attention_outputs[1:]
        else:
            return (modified_output,)
    

class ConvLoRA(nn.Module):
    """ LoRA for Conv2d layers (specifically for FastViT ConvMlp) """
    def __init__(self, original_conv, r=8, alpha=16, dropout=0.1):
        super(ConvLoRA, self).__init__()
        self.original_conv = original_conv
        self.rank = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Get dimensions from original conv
        self.in_channels = original_conv.in_channels
        self.out_channels = original_conv.out_channels
        self.kernel_size = original_conv.kernel_size
        self.stride = original_conv.stride
        self.padding = original_conv.padding

        # LoRA parameters for conv (use 1x1 kernels for efficiency)
        self.lora_A = nn.Conv2d(
            self.in_channels, r, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.lora_B = nn.Conv2d(
            r, self.out_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.dropout = nn.Dropout2d(dropout)

        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze original conv parameters
        for param in self.original_conv.parameters():
            param.requires_grad = False
            
        # Ensure LoRA parameters are trainable
        for param in self.lora_A.parameters():
            param.requires_grad = True
        for param in self.lora_B.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Original conv output
        original_output = self.original_conv(x)
        
        # LoRA output
        lora_output = self.lora_A(x)
        lora_output = self.dropout(lora_output)
        lora_output = self.lora_B(lora_output) * self.scaling
        
        return original_output + lora_output


class FastViTLoRA:
    """ Utility class to apply LoRA to FastViT models """
    @staticmethod
    def apply_lora_to_model(model, target_layers=['mlp.fc1', 'mlp.fc2'], 
                           r=8, alpha=16, dropout=0.1):
        """Apply LoRA to specified layers in FastViT model"""
        lora_modules = []
        
        for stage_idx, stage in enumerate(model.stages):
            for block_idx, block in enumerate(stage.blocks):
                for target_layer in target_layers:
                    if hasattr(block, 'mlp') and target_layer.startswith('mlp.'):
                        layer_name = target_layer.split('.')[1]  # 'fc1' or 'fc2'
                        if hasattr(block.mlp, layer_name):
                            original_layer = getattr(block.mlp, layer_name)
                            if isinstance(original_layer, nn.Conv2d):
                                # Replace with LoRA version
                                lora_layer = ConvLoRA(
                                    original_layer, r=r, alpha=alpha, dropout=dropout
                                )
                                setattr(block.mlp, layer_name, lora_layer)
                                lora_modules.append(
                                    f"stages.{stage_idx}.blocks.{block_idx}.mlp.{layer_name}"
                                )
        
        return lora_modules
    
    