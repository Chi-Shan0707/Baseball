import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    A per-frame feature extractor using a pretrained ResNet-18.
    
    Args:
        freeze_backbone (bool): If True, freeze the weights of the ResNet backbone.
    """
    def __init__(self, freeze_backbone=True):
        super(ResNetBackbone, self).__init__()
        # Load pretrained ResNet18
        # Updated to use new 'weights' parameter instead of deprecated 'pretrained'
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final fully connected layer (fc)
        # ResNet18 structure: ... -> avgpool -> fc
        # We want the output of avgpool, which is (Batch, 512, 1, 1).
        # We can just use the layers up to avgpool.
        # Alternatively, replace fc with Identity, but we need to flatten the 1x1.
        
        # Let's decompose it to make sure we get the embedding directly.
        # list(resnet.children())[:-1] gives everything up to avgpool.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.output_dim = 512

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, T, C, H, W)
            
        Returns:
            torch.Tensor: Output embeddings of shape (Batch, T, D)
        """
        b, t, c, h, w = x.shape
        
        # Reshape to (Batch * T, C, H, W) for the 2D CNN
        x = x.view(b * t, c, h, w)
        
        # Pass through backbone
        features = self.backbone(x) # (B*T, 512, 1, 1)
        
        # Flatten
        features = features.view(b * t, -1) # (B*T, 512)
        
        # Reshape back to (Batch, T, D)
        features = features.view(b, t, -1)
        
        return features

class TemporalPoolHead(nn.Module):
    """
    A temporal classification head that averages features over time.
    """
    def __init__(self, input_dim, num_classes=2):
        super(TemporalPoolHead, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Sequence of features (Batch, T, Input_Dim)
            
        Returns:
            torch.Tensor: Logits (Batch, Num_Classes)
        """
        # Average pooling over the time dimension (dim=1)
        # x shape: (B, T, D) -> (B, D)
        x_mean = x.mean(dim=1)
        
        # Classification
        logits = self.fc(x_mean)
        return logits

class LSTMHead(nn.Module):
    """
    A temporal classification head using an LSTM.
    """
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, bidirectional=False):
        super(LSTMHead, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            batch_first=True, 
            bidirectional=bidirectional
        )
        
        # If bidirectional, the output dimension is hidden_dim * 2
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_out_dim, num_classes)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Sequence of features (Batch, T, Input_Dim)
        
        Returns:
            torch.Tensor: Logits (Batch, Num_Classes)
        """
        # LSTM forward
        # self.lstm returns: output, (h_n, c_n)
        # output shape: (Batch, T, D_out)
        lstm_out, _ = self.lstm(x)
        
        # We can take the last time step's output, or pool them.
        # Taking the last step is standard for "many-to-one"
        
        # If bidirectional, we might want to concatenate forward and backward final states,
        # but lstm_out[:, -1, :] contains the last accumulated output.
        # Caution: For bidirectional, 'last' step of backward layer is at index 0 in time,
        # but PyTorch's packed output handles it.
        # A simple robust way for beginner code: take the last output of the sequence.
        final_feature = lstm_out[:, -1, :]
        
        logits = self.fc(final_feature)
        return logits

class FullModel(nn.Module):
    """
    The full model composing the backbone and the head.
    """
    def __init__(self, arch='pool', num_classes=2, freeze_backbone=True):
        super(FullModel, self).__init__()
        
        self.backbone = ResNetBackbone(freeze_backbone=freeze_backbone)
        feature_dim = self.backbone.output_dim
        
        if arch == 'pool':
            self.head = TemporalPoolHead(feature_dim, num_classes)
        elif arch == 'lstm':
            self.head = LSTMHead(feature_dim, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown architecture: {arch}")
            
    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits
