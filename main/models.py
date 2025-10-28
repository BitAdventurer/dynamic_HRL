import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TemporalEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, embed_dim, batch_first=True)
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.fc1(x)
        z = self.norm(z)
        z = self.dropout(z)
        z = self.act(z)
        z_seq = z.unsqueeze(1)
        _, hidden = self.gru(z_seq)
        return hidden.squeeze(0)

class ROIEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim, embed_dim, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, embed_dim, batch_first=True)
        self.act = nn.ReLU()

    def forward(self, x):
        z = self.fc1(x)
        z = self.norm(z)
        z = self.dropout(z)
        z = self.act(z)
        z_seq = z.unsqueeze(1)
        _, hidden = self.gru(z_seq)
        return hidden.squeeze(0)

class MacroQNet(nn.Module):
    """
    Macro-level Q-Network for hierarchical reinforcement learning.
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, num_macro_actions, dropout=0.5):
        super().__init__()
        self.temporal_emb = TemporalEmbedding(input_dim, hidden_dim, embed_dim, dropout)
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.qhead = nn.Linear(embed_dim, num_macro_actions)
        self.act = nn.ReLU()
        self.num_actions = num_macro_actions
        self._initialize_weights()

    def forward(self, x):
        emb = self.temporal_emb(x)
        emb = self.temporal_norm(emb)
        emb = self.act(emb)
        return self.qhead(emb)
    
    def select_action(self, state, epsilon=0.1):
        """
        MacroQNet: Select action according to epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration probability (epsilon-greedy)
            
        Returns:
            Selected action index
        """
        import torch
        import random
        
        # MacroQNet: Random exploration (with epsilon probability)
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # MacroQNet: Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1).item()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.qhead.weight)
        nn.init.constant_(self.qhead.bias, 0)

class MicroQNet(nn.Module):
    """
    Micro-level Q-Network for hierarchical reinforcement learning.
    """
    def __init__(self, input_dim, hidden_dim, embed_dim, num_shift_actions, dropout=0.2):
        super().__init__()
        self.roi_emb = ROIEmbedding(input_dim, hidden_dim, embed_dim, dropout)
        self.roi_norm = nn.LayerNorm(embed_dim)
        self.qhead = nn.Linear(embed_dim, num_shift_actions)
        self.act = nn.ReLU()
        self.num_actions = num_shift_actions
        self._initialize_weights()

    def forward(self, x):
        emb = self.roi_emb(x)
        emb = self.roi_norm(emb)
        emb = self.act(emb)
        return self.qhead(emb)
        
    def select_action(self, state, epsilon=0.1):
        """
        MicroQNet: Select action according to epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration probability (epsilon-greedy)
            
        Returns:
            Selected action index
        """
        import torch
        import random
        
        # MicroQNet: Random exploration (with epsilon probability)
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # MicroQNet: Greedy action selection
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values, dim=1).item()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.qhead.weight)
        nn.init.constant_(self.qhead.bias, 0)

class CRNNClassifier(nn.Module):
    """
    Convolutional Recurrent Neural Network (CRNN) or CBGRU for sequence classification.
    Supports both unidirectional (CRNN) and bidirectional (CBGRU) GRU.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, num_classes, out_channels, kernel_size, pool_size, bidirectional=False):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0 or out_channels <= 0:
            raise ValueError("Dimensions and layer counts must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size should be odd for symmetric padding")
        self.bidirectional = bidirectional
        self.conv1 = nn.Conv1d(1, out_channels, kernel_size, padding=kernel_size//2)
        self.conv_norm = nn.LayerNorm(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.activation = nn.ReLU()
        self.gru_input_dim = out_channels * (input_dim // pool_size)
        self.gru = nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(gru_output_dim)
        self.fc1 = nn.Linear(gru_output_dim, hidden_dim)
        self.fc1_norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_in',
                    nonlinearity='leaky_relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

    def forward(self, x):
        batch_size, seq_len, num_features = x.size()
        x = x.view(-1, 1, num_features)
        x = self.conv1(x)  # (B*seq, C, L)
        # LayerNorm over channel dimension (C)
        x = self.conv_norm(x.transpose(1,2)).transpose(1,2)
        x = self.activation(x)
        x = self.pool(x)
        x = x.reshape(batch_size, seq_len, -1)
        out, _ = self.gru(x)
        last_output = out[:, -1, :]
        last_output = self.layer_norm(last_output)
        x = self.fc1(last_output)
        x = self.activation(x)
        x = self.fc1_norm(x)
        logits = self.fc2(x)
        return logits

class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer to inject sequence order information.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SpatioTemporalTransformer(nn.Module):
    """
    Spatio-Temporal Transformer for sequence classification.
    Processes both spatial (ROI features) and temporal (time series) information.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, dim_feedforward, 
                 dropout, num_classes):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("Dimensions and layer counts must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Positional encoding for temporal information
        self.positional_encoding = PositionalEncoding(hidden_dim, max_len=5000, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling options
        self.use_cls_token = True
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Classification head
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize linear layers
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.constant_(self.input_projection.bias, 0)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size, seq_len, _ = x.size()
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (B, S, hidden_dim)
        x = self.input_norm(x)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, hidden_dim)
            x = torch.cat([cls_tokens, x], dim=1)  # (B, S+1, hidden_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Apply Transformer encoder
        x = self.transformer_encoder(x)  # (B, S+1, hidden_dim) or (B, S, hidden_dim)
        
        # Global pooling: use CLS token or mean pooling
        if self.use_cls_token:
            # Use CLS token output
            pooled = x[:, 0, :]  # (B, hidden_dim)
        else:
            # Mean pooling over sequence
            pooled = torch.mean(x, dim=1)  # (B, hidden_dim)
        
        pooled = self.layer_norm(pooled)
        
        # Classification head
        out = self.fc1(pooled)
        out = self.activation(out)
        out = self.fc1_norm(out)
        out = self.dropout(out)
        logits = self.fc2(out)
        
        return logits
