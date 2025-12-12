import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DirectionalLoss(nn.Module):
    def __init__(self, lambda_reg=0.5):
        super(DirectionalLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_reg = lambda_reg

    def forward(self, pred, target):
        # 1. Standard MSE (Magnitude accuracy)
        loss_mse = self.mse(pred, target)
        
        # 2. Directional Penalty (Sign accuracy)
        # If signs match, product is > 0. If signs mismatch, product is < 0.
        # We want to penalize negative products.
        sign_mismatch = torch.sign(pred) != torch.sign(target)
        loss_dir = torch.mean(sign_mismatch.float())
        
        # Combine: Minimize Error + Minimize Direction Mismatch
        return loss_mse + (self.lambda_reg * loss_dir)

class TrimodalNetwork(nn.Module):
    """
    A Trimodal Neural Network for Financial Forecasting.
    
    Modalities:
    1. Numerical: LSTM processing of time-series data (Price, Volume, Tech Indicators).
    2. Visual: ResNet-18 CNN processing of chart images (GAF/OHLC).
    3. Text: (Optional/Placeholder) Dense layer for text embeddings.
    
    Fusion Strategy:
    Uses a Gated Multimodal Unit (GMU) to learn importance weights (alpha, beta, gamma)
    for each modality before fusing them into a final regression head.
    """
    def __init__(self, num_input_dim, text_embed_dim=0, hidden_dim=128):
        super(TrimodalNetwork, self).__init__()
        
        # --- Branch 1: Numerical (LSTM for Time Series) ---
        # Input shape: (Batch, Seq_Len, Features)
        self.num_lstm = nn.LSTM(input_size=num_input_dim, hidden_size=hidden_dim, 
                                num_layers=2, batch_first=True, dropout=0.2)
        # We take the last hidden state of the LSTM
        self.num_fc = nn.Linear(hidden_dim, hidden_dim)
        self.num_bn = nn.BatchNorm1d(hidden_dim) # Batch Norm for stability

        # --- Branch 2: Visual (CNN - ResNet18) ---
        # Input shape: (Batch, 3, 224, 224)
        # We use a lightweight ResNet18
        self.cnn_backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.cnn_backbone.fc = nn.Identity() # Remove default 1000-class classification head
        
        # Project the 512 dim output of ResNet down to hidden_dim
        self.img_fc = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.img_bn = nn.BatchNorm1d(hidden_dim)

        # --- Branch 3: Text (Placeholder / Optional) ---
        # Handles cases where text might be missing by using a learnable parameter
        self.use_text = text_embed_dim > 0
        if self.use_text:
            self.text_fc = nn.Sequential(
                nn.Linear(text_embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            # If no text data is provided, learn a static "bias" vector representing "no news"
            self.text_dummy = nn.Parameter(torch.randn(1, hidden_dim)) 

        # --- Gated Fusion Mechanism ---
        # A learnable gate that sees ALL hidden states and decides weights
        # Input: Concat(h_num, h_text, h_img) -> Output: 3 weights (alpha, beta, gamma)
        self.gate_fc = nn.Linear(hidden_dim * 3, 3) 
        
        # --- Final Regressor ---
        # Input: Weighted sum of all modalities
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1) # Output: Predicted Log Return
        )

    def forward(self, x_num, x_img, x_text=None):
        """
        Forward pass.
        Returns: 
          - prediction: The regression output
          - gates: The attention weights (alpha, beta, gamma) for visualization
        """
        # 1. Numerical Path
        # LSTM returns output, (h_n, c_n). We want h_n[-1] (last layer's last hidden state)
        _, (h_n, _) = self.num_lstm(x_num)
        h_num = self.num_fc(h_n[-1])
        h_num = self.num_bn(h_num)
        
        # 2. Visual Path
        features_img = self.cnn_backbone(x_img)
        h_img = self.img_fc(features_img)
        h_img = self.img_bn(h_img)
        
        # 3. Text Path
        if self.use_text and x_text is not None:
            h_text = self.text_fc(x_text)
        else:
            # Expand the dummy parameter to match batch size
            h_text = self.text_dummy.expand(h_num.size(0), -1)

        # 4. Calculate Gates (The "Proportionality" Logic)
        concat_h = torch.cat([h_num, h_text, h_img], dim=1)
        gates = F.softmax(self.gate_fc(concat_h), dim=1) 
        
        # Split gates
        g_num = gates[:, 0:1]  # Weight for Numerical
        g_text = gates[:, 1:2] # Weight for Text
        g_img = gates[:, 2:3]  # Weight for Image
        
        # 5. Fuse Features
        # Weighted sum: Alpha*Num + Beta*Text + Gamma*Img
        fused_vector = (g_num * h_num) + (g_text * h_text) + (g_img * h_img)
        
        # 6. Prediction
        prediction = self.regressor(fused_vector)
        
        return prediction, gates