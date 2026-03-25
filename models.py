"""
模型定义模块 - 负责所有神经网络模型的定义
包含专门的HI预测头和Sigmoid激活
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Union, Dict, Any

# 尝试导入torchvision，用于预训练模型
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("警告: torchvision未安装，无法使用预训练模型")


class ResBlock1D(nn.Module):
    """1D残差块 - 用于1D CNN分支的深度特征提取"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        super(ResBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class PredictionHead(nn.Module):
    """通用预测头 - 可配置输出层"""
    
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.3, 
                 use_batch_norm=True, activation='relu', output_activation=None):
        """
        参数:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            dropout_rate: Dropout率
            use_batch_norm: 是否使用批归一化
            activation: 激活函数类型
            output_activation: 输出层激活函数（如'sigmoid'用于HI）
        """
        super(PredictionHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        if activation == 'relu':
            act_layer = nn.ReLU
        elif activation == 'gelu':
            act_layer = nn.GELU
        elif activation == 'leaky_relu':
            act_layer = nn.LeakyReLU
        else:
            act_layer = nn.ReLU
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(act_layer(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        if output_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif output_activation == 'tanh':
            layers.append(nn.Tanh())
        
        self.head = nn.Sequential(*layers)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.head(x)


class PositionalEncoding(nn.Module):
    """Transformer位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoder1D(nn.Module):
    """Transformer编码器 - 用于替代LSTM处理1D信号"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=256, dropout=0.1, max_len=1000):
        """
        参数:
            input_dim: 输入特征维度
            d_model: 模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            dim_feedforward: 前馈网络维度
            dropout: Dropout率
            max_len: 最大序列长度
        """
        super(TransformerEncoder1D, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_dim = d_model
    
    def forward(self, x):
        """
        输入: x [batch_size, seq_len, input_dim]
        输出: [batch_size, d_model] (取平均池化)
        """
        x = self.input_proj(x)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        return x


class RULPredictor(nn.Module):
    """RUL预测模型 (单模态MLP) - 同时输出RUL和HI"""
    
    def __init__(self, input_features, hidden_sizes=[64, 32, 16], dropout=0.3):
        super(RULPredictor, self).__init__()
        
        # 共享特征提取层
        layers = []
        prev_size = input_features
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.shared_features = nn.Sequential(*layers)
        
        # RUL预测头（无激活函数）
        self.rul_head = nn.Linear(prev_size, 1)
        
        # HI预测头（带Sigmoid激活，输出范围[0,1]）
        self.hi_head = nn.Sequential(
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        features = self.shared_features(x)
        pred_rul = self.rul_head(features)
        pred_hi = self.hi_head(features)
        return pred_rul, pred_hi
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters())
    
    def move_to_device(self, device):
        self.to(device)
        return self


class MultiModalRULPredictor(nn.Module):
    """增强版多模态RUL预测模型 - 支持专门的HI预测头"""
    
    def __init__(self,
                 cwt_image_shape=(1, 64, 64),
                 signal_length=1024,
                 cnn_channels=[16, 32, 64],
                 lstm_hidden_size=64,
                 lstm_num_layers=2,
                 fusion_method='late',
                 dropout_rate=0.3,
                 use_batch_norm=True,
                 cnn_architecture='simple',
                 signal_processor='lstm',
                 pretrained_model_name='resnet18',
                 prediction_head_dims=[128, 64, 32],
                 transformer_config=None):
        
        super(MultiModalRULPredictor, self).__init__()
        
        self.cwt_image_shape = cwt_image_shape
        self.signal_length = signal_length
        self.cnn_channels = cnn_channels
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.fusion_method = fusion_method
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.cnn_architecture = cnn_architecture
        self.signal_processor = signal_processor
        
        if transformer_config is None:
            self.transformer_config = {
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 256
            }
        else:
            self.transformer_config = transformer_config
        
        # CWT图像分支
        self.cwt_branch = self._build_cwt_branch(pretrained_model_name)
        
        # 振动信号分支
        self.vibration_branch, self.sequence_processor = self._build_vibration_branch()
        
        # 计算特征维度
        cwt_feature_dim = self._get_cwt_feature_dim()
        signal_feature_dim = self._get_signal_feature_dim()
        fused_feature_dim = cwt_feature_dim + signal_feature_dim
        
        # RUL预测头（无激活函数）
        self.rul_head = PredictionHead(
            input_dim=fused_feature_dim,
            hidden_dims=prediction_head_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            output_activation=None
        )
        
        # HI预测头（带Sigmoid激活，确保输出在[0,1]范围）
        self.hi_head = PredictionHead(
            input_dim=fused_feature_dim,
            hidden_dims=prediction_head_dims,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            output_activation='sigmoid'
        )
        
        self._initialize_weights()
    
    def _build_cwt_branch(self, pretrained_model_name):
        if self.cnn_architecture == 'pretrained' and TORCHVISION_AVAILABLE:
            return self._build_pretrained_cnn(pretrained_model_name)
        elif self.cnn_architecture == 'residual':
            return self._build_residual_cnn()
        else:
            return self._build_simple_cnn()
    
    def _build_simple_cnn(self):
        layers = []
        in_channels = self.cwt_image_shape[0]
        
        for i, out_channels in enumerate(self.cnn_channels):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1)
            )
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            
            if self.dropout_rate > 0:
                layers.append(nn.Dropout2d(self.dropout_rate))
            
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        
        return nn.Sequential(*layers)
    
    def _build_residual_cnn(self):
        class ResidualCNN(nn.Module):
            def __init__(self, in_channels, channels, dropout_rate, use_batch_norm):
                super(ResidualCNN, self).__init__()
                layers = []
                current_channels = in_channels
                
                for i, out_channels in enumerate(channels):
                    downsample = None
                    stride = 1
                    
                    if current_channels != out_channels:
                        downsample = nn.Sequential(
                            nn.Conv2d(current_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
                        )
                    
                    layers.append(
                        ResBlock2D(current_channels, out_channels, stride, downsample, dropout_rate)
                    )
                    
                    current_channels = out_channels
                
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                layers.append(nn.Flatten())
                
                self.model = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.model(x)
        
        return ResidualCNN(
            in_channels=self.cwt_image_shape[0],
            channels=self.cnn_channels,
            dropout_rate=self.dropout_rate,
            use_batch_norm=self.use_batch_norm
        )
    
    def _build_pretrained_cnn(self, model_name):
        if not TORCHVISION_AVAILABLE:
            print("警告: torchvision未安装，回退到简单CNN")
            return self._build_simple_cnn()
        
        if model_name == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
        elif model_name == 'resnet34':
            base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            feature_dim = 512
        elif model_name == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            feature_dim = 1280
        else:
            print(f"未知模型 {model_name}，使用ResNet18")
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            feature_dim = 512
        
        # 修改第一层以适应单通道输入
        if hasattr(base_model, 'conv1'):
            old_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                self.cwt_image_shape[0], old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
        elif hasattr(base_model, 'features'):
            old_conv = base_model.features[0][0]
            base_model.features[0][0] = nn.Conv2d(
                self.cwt_image_shape[0], old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
        
        # 移除最后的分类层
        if hasattr(base_model, 'fc'):
            base_model.fc = nn.Identity()
        elif hasattr(base_model, 'classifier'):
            base_model.classifier = nn.Identity()
        
        class PretrainedCNN(nn.Module):
            def __init__(self, backbone, feature_dim, dropout_rate):
                super(PretrainedCNN, self).__init__()
                self.backbone = backbone
                self.dropout = nn.Dropout(dropout_rate)
                self.feature_dim = feature_dim
            
            def forward(self, x):
                x = self.backbone(x)
                x = self.dropout(x)
                return x
        
        return PretrainedCNN(base_model, feature_dim, self.dropout_rate)
    
    def _build_vibration_branch(self):
        cnn_layers = []
        
        cnn_layers.append(nn.Conv1d(1, 32, kernel_size=7, padding=3, stride=2))
        if self.use_batch_norm:
            cnn_layers.append(nn.BatchNorm1d(32))
        cnn_layers.append(nn.ReLU(inplace=True))
        cnn_layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
        
        cnn_layers.append(nn.Conv1d(32, 64, kernel_size=5, padding=2, stride=1))
        if self.use_batch_norm:
            cnn_layers.append(nn.BatchNorm1d(64))
        cnn_layers.append(nn.ReLU(inplace=True))
        cnn_layers.append(nn.MaxPool1d(kernel_size=3, stride=2))
        
        vibration_branch = nn.Sequential(*cnn_layers)
        
        test_input = torch.randn(1, 1, self.signal_length)
        with torch.no_grad():
            cnn_output = vibration_branch(test_input)
            cnn_output_dim = cnn_output.shape[1]
            seq_len = cnn_output.shape[2]
        
        if self.signal_processor == 'transformer':
            sequence_processor = TransformerEncoder1D(
                input_dim=cnn_output_dim,
                d_model=self.transformer_config['d_model'],
                nhead=self.transformer_config['nhead'],
                num_layers=self.transformer_config['num_layers'],
                dim_feedforward=self.transformer_config['dim_feedforward'],
                dropout=self.dropout_rate,
                max_len=seq_len
            )
            processor_output_dim = self.transformer_config['d_model']
            
        else:
            self.lstm = nn.LSTM(
                input_size=cnn_output_dim,
                hidden_size=self.lstm_hidden_size,
                num_layers=self.lstm_num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=self.dropout_rate if self.lstm_num_layers > 1 else 0
            )
            sequence_processor = self.lstm
            processor_output_dim = self.lstm_hidden_size * 2
        
        self.processor_output_dim = processor_output_dim
        return vibration_branch, sequence_processor
    
    def _get_cwt_feature_dim(self):
        test_input = torch.randn(1, *self.cwt_image_shape)
        with torch.no_grad():
            features = self.cwt_branch(test_input)
        return features.shape[1]
    
    def _get_signal_feature_dim(self):
        return self.processor_output_dim
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, cwt_images, vibration_signals):
        """
        前向传播，返回RUL和HI预测值
        
        返回:
            tuple: (pred_rul, pred_hi)
                - pred_rul: 预测的剩余使用寿命（无范围限制）
                - pred_hi: 预测的健康因子（经Sigmoid，范围[0,1]）
        """
        cwt_features = self.cwt_branch(cwt_images)
        
        conv_features = self.vibration_branch(vibration_signals)
        
        if self.signal_processor == 'transformer':
            conv_features = conv_features.permute(0, 2, 1)
            signal_features = self.sequence_processor(conv_features)
        else:
            conv_features = conv_features.permute(0, 2, 1)
            lstm_out, (hidden, cell) = self.sequence_processor(conv_features)
            signal_features = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        fused_features = torch.cat([cwt_features, signal_features], dim=1)
        
        # 分别预测RUL和HI
        pred_rul = self.rul_head(fused_features)
        pred_hi = self.hi_head(fused_features)
        
        return pred_rul, pred_hi
    
    def get_parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def move_to_device(self, device):
        self.to(device)
        return self


class ResBlock2D(nn.Module):
    """2D残差块 - 用于图像分支"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dropout_rate=0.2):
        super(ResBlock2D, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.dropout = nn.Dropout2d(dropout_rate)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out
