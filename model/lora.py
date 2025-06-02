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
    
    