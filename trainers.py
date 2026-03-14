"""
模型训练模块 - 负责模型训练、验证和早停逻辑
支持健康因子(HI)损失计算和样本加权
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sklearn.metrics as metrics
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union

# 尝试导入实验跟踪库
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from utils import DEVICE as global_device
from models import RULPredictor, MultiModalRULPredictor
from evaluation import calculate_comprehensive_metrics, calculate_phm_score


class NegativeR2Loss(nn.Module):
    """负R²损失 - 最大化R²"""
    
    def __init__(self):
        super(NegativeR2Loss, self).__init__()
    
    def forward(self, y_pred, y_true):
        y_true_mean = torch.mean(y_true)
        ss_tot = torch.sum((y_true - y_true_mean) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        return -r2


class WeightedMSELoss(nn.Module):
    """加权MSE损失 - 根据样本重要性加权"""
    
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
    
    def forward(self, pred, target, weights=None):
        """
        计算加权MSE损失
        
        参数:
            pred: 预测值
            target: 目标值
            weights: 样本权重，形状与target相同
        """
        squared_error = (pred - target) ** 2
        if weights is not None:
            # 确保weights与squared_error形状一致
            if weights.dim() < squared_error.dim():
                weights = weights.view(-1, 1)
            return torch.mean(weights * squared_error)
        else:
            return torch.mean(squared_error)


class BaseTrainer:
    """基础训练器 - 包含通用训练逻辑"""
    
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None,
                 experiment_tracker=None, config=None):
        
        self.device = device if device is not None else global_device
        self.model = model.move_to_device(self.device)
        self.optimizer = optimizer
        self.criterion = criterion  # 用于RUL的损失函数
        self.scheduler = scheduler
        self.experiment_tracker = experiment_tracker
        self.config = config or {}
        
        # 损失权重配置
        self.rul_loss_weight = self.config.get('rul_loss_weight', 1.0)
        self.hi_loss_weight = self.config.get('hi_loss_weight', 1.0)  # 增大HI权重
        
        # 创建损失函数
        self.mse_loss = nn.MSELoss()
        self.weighted_mse_loss = WeightedMSELoss()
        
        # 是否使用样本加权（退化后期权重更大）
        self.use_sample_weighting = self.config.get('use_sample_weighting', True)
        self.weighting_alpha = self.config.get('weighting_alpha', 2.0)  # 权重指数
        
        self.history = {
            'train_total_loss': [],
            'val_total_loss': [],
            'train_rul_loss': [],
            'val_rul_loss': [],
            'train_hi_loss': [],
            'val_hi_loss': [],
            'train_r2': [],
            'val_r2': [],
            'train_hi_r2': [],
            'val_hi_r2': [],
            'train_phm_score': [],
            'val_phm_score': [],
            'learning_rates': []
        }
        
        self.model_type = type(model).__name__
        print(f"训练器初始化: 使用设备 {self.device}, 模型类型: {self.model_type}")
        print(f"损失权重: RUL={self.rul_loss_weight}, HI={self.hi_loss_weight}")
        print(f"使用样本加权: {self.use_sample_weighting}, 权重指数: {self.weighting_alpha}")
        
        self._init_experiment_tracking()
    
    def _init_experiment_tracking(self):
        if self.experiment_tracker == 'mlflow' and MLFLOW_AVAILABLE:
            mlflow.start_run()
            for key, value in self.config.items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
                elif isinstance(value, (list, dict)):
                    mlflow.log_param(key, str(value))
        
        elif self.experiment_tracker == 'wandb' and WANDB_AVAILABLE:
            wandb.init(config=self.config)
    
    def _log_metrics(self, metrics, step=None):
        if self.experiment_tracker == 'mlflow' and MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        
        elif self.experiment_tracker == 'wandb' and WANDB_AVAILABLE:
            wandb.log(metrics, step=step)
    
    def _compute_sample_weights(self, hi_labels):
        """
        根据HI值计算样本权重
        退化越严重（HI越小），权重越大
        权重 = (1 - HI) ^ alpha
        """
        if not self.use_sample_weighting:
            return None
        
        weights = (1 - hi_labels) ** self.weighting_alpha
        # 避免除零
        weights = weights / (torch.sum(weights) + 1e-8)
        return weights
    
    def train_epoch(self, train_loader):
        raise NotImplementedError
    
    def validate(self, val_loader):
        raise NotImplementedError
    
    def train(self, train_loader, val_loader, num_epochs=100, 
              patience=15, checkpoint_path='best_model.pth'):
        
        best_val_hi_r2 = -float('inf')  # 使用HI-R²作为早停指标
        patience_counter = 0
        
        print(f"开始训练，共{num_epochs}个epoch")
        print(f"训练设备: {self.device}")
        print(f"模型参数量: {self.model.get_parameter_count():,}")
        print("-" * 80)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            try:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                train_loss, train_metrics = self.train_epoch(train_loader)
                val_loss, val_metrics = self.validate(val_loader)
                
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # 保存训练历史
                self.history['train_total_loss'].append(train_loss)
                self.history['val_total_loss'].append(val_loss)
                self.history['train_rul_loss'].append(train_metrics.get('rul_loss', 0))
                self.history['val_rul_loss'].append(val_metrics.get('rul_loss', 0))
                self.history['train_hi_loss'].append(train_metrics.get('hi_loss', 0))
                self.history['val_hi_loss'].append(val_metrics.get('hi_loss', 0))
                self.history['train_r2'].append(train_metrics.get('r2', 0))
                self.history['val_r2'].append(val_metrics.get('r2', 0))
                self.history['train_hi_r2'].append(train_metrics.get('hi_r2', 0))
                self.history['val_hi_r2'].append(val_metrics.get('hi_r2', 0))
                self.history['train_phm_score'].append(train_metrics.get('phm_score', 0))
                self.history['val_phm_score'].append(val_metrics.get('phm_score', 0))
                
                # 打印进度
                print(f"Train Loss: {train_loss:.6f} (RUL: {train_metrics.get('rul_loss', 0):.6f}, HI: {train_metrics.get('hi_loss', 0):.6f})")
                print(f"         R²: {train_metrics.get('r2', 0):.4f}, HI-R²: {train_metrics.get('hi_r2', 0):.4f}")
                print(f"         PHM Score: {train_metrics.get('phm_score', 0):.4f}")
                print(f"Val Loss:   {val_loss:.6f} (RUL: {val_metrics.get('rul_loss', 0):.6f}, HI: {val_metrics.get('hi_loss', 0):.6f})")
                print(f"         R²: {val_metrics.get('r2', 0):.4f}, HI-R²: {val_metrics.get('hi_r2', 0):.4f}")
                print(f"         PHM Score: {val_metrics.get('phm_score', 0):.4f}")
                
                log_metrics = {
                    'train_total_loss': train_loss,
                    'val_total_loss': val_loss,
                    'train_rul_loss': train_metrics.get('rul_loss', 0),
                    'val_rul_loss': val_metrics.get('rul_loss', 0),
                    'train_hi_loss': train_metrics.get('hi_loss', 0),
                    'val_hi_loss': val_metrics.get('hi_loss', 0),
                    'train_r2': train_metrics.get('r2', 0),
                    'val_r2': val_metrics.get('r2', 0),
                    'train_hi_r2': train_metrics.get('hi_r2', 0),
                    'val_hi_r2': val_metrics.get('hi_r2', 0),
                    'train_phm_score': train_metrics.get('phm_score', 0),
                    'val_phm_score': val_metrics.get('phm_score', 0),
                    'learning_rate': current_lr
                }
                self._log_metrics(log_metrics, step=epoch)
                
                # 使用HI-R²作为早停指标
                current_val_score = val_metrics.get('hi_r2', 0)
                if current_val_score > best_val_hi_r2:
                    best_val_hi_r2 = current_val_score
                    patience_counter = 0
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                        'val_loss': val_loss,
                        'val_r2': val_metrics.get('r2', 0),
                        'val_hi_r2': val_metrics.get('hi_r2', 0),
                        'train_loss': train_loss,
                        'train_r2': train_metrics.get('r2', 0),
                        'model_type': type(self.model).__name__,
                    }, checkpoint_path)
                    print(f"✓ 保存最佳模型: {checkpoint_path}")
                    
                    if self.experiment_tracker == 'mlflow' and MLFLOW_AVAILABLE:
                        mlflow.log_artifact(checkpoint_path)
                    elif self.experiment_tracker == 'wandb' and WANDB_AVAILABLE:
                        wandb.save(checkpoint_path)
                        
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"早停触发，停止训练")
                        break
                        
            except Exception as e:
                print(f"Epoch {epoch+1} 训练出错: {e}")
                traceback.print_exc()
                break
        
        print(f"\n训练完成! 最佳验证HI-R²: {best_val_hi_r2:.4f}")
        
        if self.experiment_tracker == 'mlflow' and MLFLOW_AVAILABLE:
            mlflow.end_run()
        elif self.experiment_tracker == 'wandb' and WANDB_AVAILABLE:
            wandb.finish()
        
        return self.history, best_val_hi_r2


class RULTrainer(BaseTrainer):
    """RUL模型训练器 (单模态) - 支持RUL和HI联合训练"""
    
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None,
                 experiment_tracker=None, config=None):
        super().__init__(model, optimizer, criterion, scheduler, device,
                        experiment_tracker, config)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_rul_loss = 0
        total_hi_loss = 0
        
        all_rul_preds = []
        all_rul_labels = []
        all_hi_preds = []
        all_hi_labels = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            rul_labels, hi_labels = labels[0].to(self.device), labels[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 模型返回 (pred_rul, pred_hi)
            pred_rul, pred_hi = self.model(inputs)
            
            # 计算样本权重
            weights = self._compute_sample_weights(hi_labels)
            
            # RUL损失
            rul_loss = self.criterion(pred_rul, rul_labels)
            
            # HI损失 - 使用加权MSE
            hi_loss = self.weighted_mse_loss(pred_hi, hi_labels, weights)
            
            # 组合损失
            loss = self.rul_loss_weight * rul_loss + self.hi_loss_weight * hi_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_rul_loss += rul_loss.item()
            total_hi_loss += hi_loss.item()
            
            all_rul_preds.extend(pred_rul.detach().cpu().numpy().flatten())
            all_rul_labels.extend(rul_labels.detach().cpu().numpy().flatten())
            all_hi_preds.extend(pred_hi.detach().cpu().numpy().flatten())
            all_hi_labels.extend(hi_labels.detach().cpu().numpy().flatten())
            
            if self.device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        avg_rul_loss = total_rul_loss / len(train_loader)
        avg_hi_loss = total_hi_loss / len(train_loader)
        
        # 计算RUL指标
        rul_metrics = calculate_comprehensive_metrics(
            np.array(all_rul_labels), 
            np.array(all_rul_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        # 计算HI指标
        hi_metrics = calculate_comprehensive_metrics(
            np.array(all_hi_labels), 
            np.array(all_hi_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        # 计算PHM Score
        phm_score = calculate_phm_score(
            np.array(all_rul_labels),
            np.array(all_rul_preds)
        )
        
        metrics_dict = {
            'rul_loss': avg_rul_loss,
            'hi_loss': avg_hi_loss,
            'r2': rul_metrics['r2'],
            'rmse': rul_metrics['rmse'],
            'mae': rul_metrics['mae'],
            'mape': rul_metrics['mape'],
            'hi_r2': hi_metrics['r2'],
            'hi_rmse': hi_metrics['rmse'],
            'hi_mae': hi_metrics['mae'],
            'hi_mape': hi_metrics['mape'],
            'phm_score': phm_score
        }
        
        return avg_loss, metrics_dict
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_rul_loss = 0
        total_hi_loss = 0
        
        all_rul_preds = []
        all_rul_labels = []
        all_hi_preds = []
        all_hi_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(self.device)
                rul_labels, hi_labels = labels[0].to(self.device), labels[1].to(self.device)
                
                pred_rul, pred_hi = self.model(inputs)
                
                rul_loss = self.criterion(pred_rul, rul_labels)
                hi_loss = self.weighted_mse_loss(pred_hi, hi_labels)
                loss = self.rul_loss_weight * rul_loss + self.hi_loss_weight * hi_loss
                
                total_loss += loss.item()
                total_rul_loss += rul_loss.item()
                total_hi_loss += hi_loss.item()
                
                all_rul_preds.extend(pred_rul.cpu().numpy().flatten())
                all_rul_labels.extend(rul_labels.cpu().numpy().flatten())
                all_hi_preds.extend(pred_hi.cpu().numpy().flatten())
                all_hi_labels.extend(hi_labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        avg_rul_loss = total_rul_loss / len(val_loader)
        avg_hi_loss = total_hi_loss / len(val_loader)
        
        rul_metrics = calculate_comprehensive_metrics(
            np.array(all_rul_labels), 
            np.array(all_rul_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        hi_metrics = calculate_comprehensive_metrics(
            np.array(all_hi_labels), 
            np.array(all_hi_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        phm_score = calculate_phm_score(
            np.array(all_rul_labels),
            np.array(all_rul_preds)
        )
        
        metrics_dict = {
            'rul_loss': avg_rul_loss,
            'hi_loss': avg_hi_loss,
            'r2': rul_metrics['r2'],
            'rmse': rul_metrics['rmse'],
            'mae': rul_metrics['mae'],
            'mape': rul_metrics['mape'],
            'hi_r2': hi_metrics['r2'],
            'hi_rmse': hi_metrics['rmse'],
            'hi_mae': hi_metrics['mae'],
            'hi_mape': hi_metrics['mape'],
            'phm_score': phm_score
        }
        
        return avg_loss, metrics_dict


class MultiModalTrainer(BaseTrainer):
    """多模态模型训练器 - 支持健康因子(HI)训练"""
    
    def __init__(self, model, optimizer, criterion, scheduler=None, device=None,
                 experiment_tracker=None, config=None):
        super().__init__(model, optimizer, criterion, scheduler, device,
                        experiment_tracker, config)
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_rul_loss = 0
        total_hi_loss = 0
        
        all_rul_preds = []
        all_rul_labels = []
        all_hi_preds = []
        all_hi_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            cwt_images = inputs[0].to(self.device)
            vibration_signals = inputs[1].to(self.device)
            
            rul_labels, hi_labels = labels[0].to(self.device), labels[1].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 模型返回 (pred_rul, pred_hi)
            pred_rul, pred_hi = self.model(cwt_images, vibration_signals)
            
            # 计算样本权重
            weights = self._compute_sample_weights(hi_labels)
            
            # RUL损失
            rul_loss = self.criterion(pred_rul, rul_labels)
            
            # HI损失 - 使用加权MSE
            hi_loss = self.weighted_mse_loss(pred_hi, hi_labels, weights)
            
            # 组合损失
            loss = self.rul_loss_weight * rul_loss + self.hi_loss_weight * hi_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_rul_loss += rul_loss.item()
            total_hi_loss += hi_loss.item()
            
            all_rul_preds.extend(pred_rul.detach().cpu().numpy().flatten())
            all_rul_labels.extend(rul_labels.detach().cpu().numpy().flatten())
            all_hi_preds.extend(pred_hi.detach().cpu().numpy().flatten())
            all_hi_labels.extend(hi_labels.detach().cpu().numpy().flatten())
            
            if self.device.type == 'cuda' and batch_idx % 50 == 0:
                torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(train_loader)
        avg_rul_loss = total_rul_loss / len(train_loader)
        avg_hi_loss = total_hi_loss / len(train_loader)
        
        # 计算RUL指标
        rul_metrics = calculate_comprehensive_metrics(
            np.array(all_rul_labels), 
            np.array(all_rul_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        # 计算HI指标
        hi_metrics = calculate_comprehensive_metrics(
            np.array(all_hi_labels), 
            np.array(all_hi_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        # 计算PHM Score
        phm_score = calculate_phm_score(
            np.array(all_rul_labels),
            np.array(all_rul_preds)
        )
        
        metrics_dict = {
            'rul_loss': avg_rul_loss,
            'hi_loss': avg_hi_loss,
            'r2': rul_metrics['r2'],
            'rmse': rul_metrics['rmse'],
            'mae': rul_metrics['mae'],
            'mape': rul_metrics['mape'],
            'hi_r2': hi_metrics['r2'],
            'hi_rmse': hi_metrics['rmse'],
            'hi_mae': hi_metrics['mae'],
            'hi_mape': hi_metrics['mape'],
            'phm_score': phm_score
        }
        
        return avg_loss, metrics_dict
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_rul_loss = 0
        total_hi_loss = 0
        
        all_rul_preds = []
        all_rul_labels = []
        all_hi_preds = []
        all_hi_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                cwt_images = inputs[0].to(self.device)
                vibration_signals = inputs[1].to(self.device)
                
                rul_labels, hi_labels = labels[0].to(self.device), labels[1].to(self.device)
                
                pred_rul, pred_hi = self.model(cwt_images, vibration_signals)
                
                rul_loss = self.criterion(pred_rul, rul_labels)
                hi_loss = self.weighted_mse_loss(pred_hi, hi_labels)
                loss = self.rul_loss_weight * rul_loss + self.hi_loss_weight * hi_loss
                
                total_loss += loss.item()
                total_rul_loss += rul_loss.item()
                total_hi_loss += hi_loss.item()
                
                all_rul_preds.extend(pred_rul.cpu().numpy().flatten())
                all_rul_labels.extend(rul_labels.cpu().numpy().flatten())
                all_hi_preds.extend(pred_hi.cpu().numpy().flatten())
                all_hi_labels.extend(hi_labels.cpu().numpy().flatten())
        
        avg_loss = total_loss / len(val_loader)
        avg_rul_loss = total_rul_loss / len(val_loader)
        avg_hi_loss = total_hi_loss / len(val_loader)
        
        rul_metrics = calculate_comprehensive_metrics(
            np.array(all_rul_labels), 
            np.array(all_rul_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        hi_metrics = calculate_comprehensive_metrics(
            np.array(all_hi_labels), 
            np.array(all_hi_preds),
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        phm_score = calculate_phm_score(
            np.array(all_rul_labels),
            np.array(all_rul_preds)
        )
        
        metrics_dict = {
            'rul_loss': avg_rul_loss,
            'hi_loss': avg_hi_loss,
            'r2': rul_metrics['r2'],
            'rmse': rul_metrics['rmse'],
            'mae': rul_metrics['mae'],
            'mape': rul_metrics['mape'],
            'hi_r2': hi_metrics['r2'],
            'hi_rmse': hi_metrics['rmse'],
            'hi_mae': hi_metrics['mae'],
            'hi_mape': hi_metrics['mape'],
            'phm_score': phm_score
        }
        
        return avg_loss, metrics_dict