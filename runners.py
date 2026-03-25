"""
流程控制模块 - 负责协调整个流程，作为高层控制器
包含增强版模型工厂和训练器，支持健康因子(HI)联合训练
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import traceback
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import glob
import sklearn.metrics as metrics

# 直接导入（不使用相对导入）
from config import BatchConfig
from utils import DEVICE, selected_font, SCIPY_ZOOM_AVAILABLE
from data_loader import XJTUDataLoader
from processors import RULDataProcessor, RULDataset, MultiModalDataset, MultiModalDataProcessor
from models import (
    RULPredictor, 
    MultiModalRULPredictor,
    PredictionHead,
    ResBlock1D,
    TransformerEncoder1D
)
from trainers import (
    RULTrainer, 
    MultiModalTrainer, 
    NegativeR2Loss,
    BaseTrainer,
    WeightedMSELoss
)
from evaluators import RULEvaluator, ModelComparisonVisualizer, VisualizationTool
from feature_extractors import CWTFeatureExtractor
from evaluation import calculate_comprehensive_metrics, print_metrics_summary


class ModelRunner:
    """通用模型运行器，封装模型训练和评估的通用逻辑"""
    
    def __init__(self, config, device=None):
        self.config = config
        self.device = device if device is not None else DEVICE
        self.logger = logging.getLogger("ModelRunner")
    
    def create_dataloaders(self, X_train, y_train, X_val, y_val, X_test, y_test,
                          batch_size, dataset_class, **dataset_kwargs):
        """创建数据加载器"""
        # 创建数据集
        train_dataset = dataset_class(X_train, y_train, **dataset_kwargs)
        val_dataset = dataset_class(X_val, y_val, **dataset_kwargs)
        test_dataset = dataset_class(X_test, y_test, **dataset_kwargs)
        
        # 获取配置参数
        pin_memory = self.config.get('pin_memory', True) and torch.cuda.is_available()
        num_workers = self.config.get('num_workers', 0)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model, train_loader, val_loader, checkpoint_path, 
                   patience=None, num_epochs=None, config=None):
        """
        训练模型 - 支持加权损失和学习率调度器
        """
        # 获取配置参数
        learning_rate = self.config.get('learning_rate', 0.001)
        epochs = num_epochs or self.config.get('epochs', 100)
        patience_val = patience or self.config.get('patience', 15)
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, 
                              weight_decay=self.config.get('weight_decay', 0))
        
        # 创建损失函数 - 使用MSE（具体的加权在Trainer中处理）
        criterion = nn.MSELoss()
        
        # 创建学习率调度器
        scheduler = None
        lr_scheduler_type = self.config.get('lr_scheduler', 'none')
        
        if lr_scheduler_type == 'plateau':
            params = self.config.get('lr_scheduler_params', {})
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=params.get('mode', 'min'),
                factor=params.get('factor', 0.5),
                patience=params.get('patience', 5),
                min_lr=params.get('min_lr', 1e-6)
            )
            self.logger.info(f"使用ReduceLROnPlateau调度器: factor={params.get('factor', 0.5)}")
        elif lr_scheduler_type == 'cosine':
            params = self.config.get('lr_scheduler_params', {})
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params.get('T_max', epochs),
                eta_min=params.get('min_lr', 1e-6)
            )
            self.logger.info(f"使用CosineAnnealingLR调度器: T_max={params.get('T_max', epochs)}")
        
        # 实验跟踪配置
        experiment_tracker = None
        if self.config.get('enable_experiment_tracking', False):
            experiment_tracker = self.config.get('experiment_tracker', 'mlflow')
            self.logger.info(f"启用实验跟踪: {experiment_tracker}")
        
        # 准备训练配置
        train_config = {
            'tolerance_threshold': self.config.get('tolerance_threshold', 0.1),
            'learning_rate': learning_rate,
            'batch_size': self.config.get('batch_size', 32),
            'epochs': epochs,
            'lr_scheduler': lr_scheduler_type,
            'model_architecture': self.config.get('cnn_architecture', 'simple'),
            'signal_processor': self.config.get('signal_processor', 'lstm'),
            # 新增HI相关配置
            'rul_loss_weight': self.config.get('rul_loss_weight', 1.0),
            'hi_loss_weight': self.config.get('hi_loss_weight', 1.0),
            'use_sample_weighting': self.config.get('use_sample_weighting', True),
            'weighting_alpha': self.config.get('weighting_alpha', 2.0)
        }
        
        # 根据模型类型创建合适的训练器
        if isinstance(model, MultiModalRULPredictor):
            trainer = MultiModalTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=self.device,
                experiment_tracker=experiment_tracker,
                config=train_config
            )
        else:
            trainer = RULTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                device=self.device,
                experiment_tracker=experiment_tracker,
                config=train_config
            )
        
        # 执行训练
        history, best_val_score = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            patience=patience_val,
            checkpoint_path=checkpoint_path
        )
        
        return history, best_val_score
    
    # 在 ModelRunner 类的 evaluate_model 方法中
    def evaluate_model(self, model, test_loader, data_processor, device):
        """评估模型 - 使用XJTU-SY标准指标"""
        model.eval()
        all_rul_preds = []
        all_rul_labels = []
        
        model = model.to(device)
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, labels = batch
                else:
                    self.logger.warning(f"批次数据格式异常: {type(batch)}")
                    continue
                
                if isinstance(model, MultiModalRULPredictor):
                    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
                        cwt_images = inputs[0].to(device, non_blocking=True)
                        vibration_signals = inputs[1].to(device, non_blocking=True)
                        pred_rul, _ = model(cwt_images, vibration_signals)
                        
                        if isinstance(labels, (list, tuple)) and len(labels) == 2:
                            rul_labels = labels[0].cpu().numpy().flatten()
                        else:
                            self.logger.error(f"错误: 标签格式不正确: {type(labels)}")
                            continue
                    else:
                        self.logger.error(f"错误: 多模态模型输入格式不正确")
                        continue
                else:
                    if isinstance(inputs, (list, tuple)):
                        inputs = inputs[0].to(device, non_blocking=True)
                    else:
                        inputs = inputs.to(device, non_blocking=True)
                    
                    pred_rul, _ = model(inputs)
                    
                    if isinstance(labels, (list, tuple)) and len(labels) == 2:
                        rul_labels = labels[0].cpu().numpy().flatten()
                    else:
                        self.logger.error(f"错误: 标签格式不正确: {type(labels)}")
                        continue
                
                all_rul_preds.extend(pred_rul.cpu().numpy().flatten())
                all_rul_labels.extend(rul_labels)
        
        all_rul_preds = np.array(all_rul_preds)
        all_rul_labels = np.array(all_rul_labels)
        
        if all_rul_preds.size == 0 or all_rul_labels.size == 0:
            self.logger.error("Prediction or label array is empty after evaluation.")
            return {}, [], []
        
        # 【修改】反归一化RUL - 使用 data_processor 的方法
        all_rul_preds_original = all_rul_preds.copy()
        all_rul_labels_original = all_rul_labels.copy()
        
        if data_processor is not None:
            try:
                # 反归一化预测值和真实值
                all_rul_preds_original = data_processor.inverse_transform_labels(all_rul_preds)
                all_rul_labels_original = data_processor.inverse_transform_labels(all_rul_labels)
                
                self.logger.info(f"反归一化后统计 (RUL):")
                self.logger.info(f"  - 真实值范围: [{all_rul_labels_original.min():.2f}, {all_rul_labels_original.max():.2f}]")
                self.logger.info(f"  - 预测值范围: [{all_rul_preds_original.min():.2f}, {all_rul_preds_original.max():.2f}]")
                self.logger.info(f"  - 归一化预测值范围: [{all_rul_preds.min():.4f}, {all_rul_preds.max():.4f}]")
                self.logger.info(f"  - 归一化真实值范围: [{all_rul_labels.min():.4f}, {all_rul_labels.max():.4f}]")
                
                if np.any(all_rul_preds_original < 0):
                    negative_preds = np.sum(all_rul_preds_original < 0)
                    self.logger.warning(f"⚠️ 发现 {negative_preds} 个负的RUL预测值，将裁剪为0")
                    all_rul_preds_original = np.maximum(all_rul_preds_original, 0)
                    
            except Exception as e:
                self.logger.error(f"反归一化标签时出错: {e}")
                self.logger.warning("将在归一化尺度上计算指标")
                all_rul_preds_original = all_rul_preds
                all_rul_labels_original = all_rul_labels
        
        # 计算XJTU-SY标准指标（在原始尺度上计算）
        eval_metrics = calculate_comprehensive_metrics(
            all_rul_labels_original, 
            all_rul_preds_original,
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        # 【新增】同时计算归一化尺度上的指标（用于监控训练过程）
        norm_metrics = calculate_comprehensive_metrics(
            all_rul_labels, 
            all_rul_preds,
            tolerance=self.config.get('tolerance_threshold', 0.1)
        )
        
        # 记录评估结果
        self.logger.info("=" * 60)
        self.logger.info("XJTU-SY标准评估结果 (原始尺度):")
        self.logger.info("-" * 40)
        self.logger.info(f"  R²: {eval_metrics['r2']:.4f}")
        self.logger.info(f"  MSE: {eval_metrics['mse']:.6f}")
        self.logger.info(f"  RMSE: {eval_metrics['rmse']:.4f}")
        self.logger.info(f"  MAE: {eval_metrics['mae']:.4f}")
        self.logger.info("-" * 40)
        self.logger.info("归一化尺度指标 (监控用):")
        self.logger.info(f"  R²: {norm_metrics['r2']:.4f}")
        self.logger.info(f"  MSE: {norm_metrics['mse']:.6f}")
        self.logger.info(f"  RMSE: {norm_metrics['rmse']:.4f}")
        self.logger.info(f"  MAE: {norm_metrics['mae']:.4f}")
        self.logger.info("=" * 60)
        
        # 打印前10个样本的详细结果（原始尺度）
        self.logger.info("\n前10个样本的详细结果 (原始尺度):")
        self.logger.info("样本 | 真实RUL | 预测RUL | 绝对误差 | 归一化真实 | 归一化预测")
        self.logger.info("-" * 75)
        for i in range(min(10, len(all_rul_labels_original))):
            abs_error = abs(all_rul_preds_original[i] - all_rul_labels_original[i])
            self.logger.info(f"{i:4d} | {all_rul_labels_original[i]:7.2f} | {all_rul_preds_original[i]:7.2f} | "
                            f"{abs_error:7.2f} | {all_rul_labels[i]:6.4f} | {all_rul_preds[i]:6.4f}")
        
        return eval_metrics, all_rul_preds_original, all_rul_labels_original
    
    def _calculate_metrics(self, preds, labels):
        """计算评估指标（向后兼容）"""
        eval_metrics = {}
        
        eval_metrics['mse'] = metrics.mean_squared_error(labels, preds)
        eval_metrics['rmse'] = np.sqrt(eval_metrics['mse'])
        eval_metrics['mae'] = metrics.mean_absolute_error(labels, preds)
        
        try:
            eval_metrics['r2'] = metrics.r2_score(labels, preds)
        except Exception as e:
            print(f"计算R²时出错: {e}")
            eval_metrics['r2'] = 0.0
        
        return eval_metrics


class ModelFactory:
    """模型工厂，用于创建不同类型的模型"""
    
    @staticmethod
    def create_model(model_type, config, input_dim=None, **kwargs):
        """创建模型实例"""
        if model_type == 'mlp':
            if input_dim is None:
                raise ValueError("MLP模型需要指定input_dim参数")
            return RULPredictor(
                input_features=input_dim,
                hidden_sizes=config['hidden_sizes'],
                dropout=config['dropout']
            )
        
        elif model_type == 'multimodal':
            cwt_image_shape = kwargs.get('cwt_image_shape', (1, 64, 64))
            signal_length = kwargs.get('signal_length', 1024)
            
            # 获取增强配置
            cnn_architecture = config.get('cnn_architecture', 'simple')
            signal_processor = config.get('signal_processor', 'lstm')
            pretrained_model_name = config.get('pretrained_model_name', 'resnet18')
            prediction_head_dims = config.get('prediction_head_dims', [128, 64, 32])
            transformer_config = config.get('transformer_config', None)
            
            print(f"创建多模态模型: CNN架构={cnn_architecture}, 信号处理器={signal_processor}")
            
            return MultiModalRULPredictor(
                cwt_image_shape=cwt_image_shape,
                signal_length=signal_length,
                cnn_channels=kwargs.get('cnn_channels', [16, 32, 64]),
                lstm_hidden_size=kwargs.get('lstm_hidden_size', 64),
                lstm_num_layers=kwargs.get('lstm_num_layers', 2),
                fusion_method=kwargs.get('fusion_method', 'late'),
                dropout_rate=config['dropout'],
                cnn_architecture=cnn_architecture,
                signal_processor=signal_processor,
                pretrained_model_name=pretrained_model_name,
                prediction_head_dims=prediction_head_dims,
                transformer_config=transformer_config
            )
        
        elif model_type == 'linear':
            linear_model_type = config.get('linear_model_type', 'ridge')
            if linear_model_type == 'linear':
                return LinearRegression()
            elif linear_model_type == 'ridge':
                return Ridge(alpha=config.get('ridge_alpha', 1.0))
            elif linear_model_type == 'lasso':
                return Lasso(alpha=config.get('lasso_alpha', 0.1))
            else:
                return LinearRegression()
        
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")


class DataProcessorFactory:
    """数据处理器工厂"""
    
    @staticmethod
    def create_processor(processor_type, config, **kwargs):
        """创建数据处理器实例"""
        if processor_type == 'standard':
            return RULDataProcessor(
                window_size=config['window_size'],
                overlap_ratio=config['overlap_ratio'],
                sampling_rate=config['sampling_rate']
            )
        
        elif processor_type == 'multimodal':
            return MultiModalDataProcessor(
                window_size=config['window_size'],
                overlap_ratio=config['overlap_ratio'],
                sampling_rate=config['sampling_rate'],
                cwt_image_shape=kwargs.get('cwt_image_shape', (1, 64, 64)),
                config=config
            )
        
        else:
            raise ValueError(f"不支持的处理器类型: {processor_type}")


class EnhancedBatchRULProcessor:
    """增强版批量RUL处理器（支持模型对比和鲁棒性验证）"""
    
    def __init__(self, config: BatchConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger("BatchProcessor")
        self.results_summary = []
        
        # 获取工况选择配置
        self.use_only_35khz = self.config.get('use_only_35khz', True)
        
        # 创建输出目录
        self.output_root = Path(config['output_root'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型运行器
        self.model_runner = ModelRunner(config)
        
        # 初始化可视化工具
        self.rul_evaluator = RULEvaluator(config)
        self.model_comparison_visualizer = ModelComparisonVisualizer(config)
        
        self.logger.info(f"批量处理器初始化: 使用设备 {DEVICE}")
        self.logger.info(f"只使用35kHz工况: {self.use_only_35khz}")
    
    def find_bearing_folders(self) -> List[str]:
        """
        查找所有轴承文件夹，支持二级目录结构
        兼容两种结构：
        1. 旧结构: data_root/Bearing1_1/
        2. 新结构: data_root/工况目录/Bearing1_1/
        """
        data_root = Path(self.config['data_root'])
        
        # 调试信息：打印根目录
        self.logger.info("=" * 60)
        self.logger.info(f"开始扫描轴承文件夹")
        self.logger.info(f"数据根目录: {data_root}")
        self.logger.info(f"目录是否存在: {data_root.exists()}")
        
        if not data_root.exists():
            self.logger.error(f"❌ 数据根目录不存在: {data_root}")
            return []
        
        bearing_folders = []
        pattern = re.compile(self.config['pattern'])  # 默认为 r'.*Bearing.*_.*'
        
        # 步骤1: 列出根目录下的所有内容
        try:
            root_contents = list(data_root.iterdir())
            root_dirs = [d for d in root_contents if d.is_dir()]
            self.logger.info(f"根目录下的文件夹: {[d.name for d in root_dirs]}")
        except Exception as e:
            self.logger.error(f"无法读取根目录: {e}")
            return []
        
        # 步骤2: 递归扫描策略
        # 首先检查根目录下是否直接有轴承文件夹（兼容旧结构）
        self.logger.info("正在扫描根目录下的轴承文件夹...")
        found_in_root = False
        
        for item in root_dirs:
            if pattern.match(item.name):
                bearing_folders.append(str(item))
                self.logger.info(f"✅ 在根目录找到轴承文件夹: {item.name}")
                found_in_root = True
        
        # 如果根目录下没有轴承文件夹，则扫描二级目录（工况目录）
        if not found_in_root:
            self.logger.info("根目录下未直接找到轴承文件夹，开始扫描工况子目录...")
            
            # 遍历每个一级子目录（工况目录）
            for condition_dir in root_dirs:
                self.logger.info(f"检查工况目录: {condition_dir.name}")
                
                try:
                    # 获取工况目录下的所有内容
                    condition_contents = list(condition_dir.iterdir())
                    condition_subdirs = [d for d in condition_contents if d.is_dir()]
                    
                    if not condition_subdirs:
                        self.logger.debug(f"  工况目录 {condition_dir.name} 下没有子文件夹")
                        continue
                    
                    # 在工况目录下查找轴承文件夹
                    for sub_dir in condition_subdirs:
                        if pattern.match(sub_dir.name):
                            # 保存相对路径（相对于data_root）
                            rel_path = os.path.relpath(sub_dir, data_root)
                            bearing_folders.append(rel_path)
                            self.logger.info(f"  ✅ 找到轴承文件夹: {rel_path}")
                            
                except Exception as e:
                    self.logger.warning(f"  读取工况目录 {condition_dir.name} 时出错: {e}")
                    continue
        
        # 步骤3: 汇总结果
        self.logger.info("-" * 40)
        if bearing_folders:
            self.logger.info(f"🎯 扫描完成，共找到 {len(bearing_folders)} 个轴承文件夹:")
            for i, folder in enumerate(bearing_folders, 1):
                self.logger.info(f"  {i}. {folder}")
        else:
            self.logger.warning("⚠️ 未找到任何轴承文件夹！")
            self.logger.warning("可能的原因:")
            self.logger.warning("  1. 数据路径配置错误")
            self.logger.warning("  2. 文件夹命名不符合模式: " + self.config['pattern'])
            self.logger.warning("  3. 数据文件不在预期位置")
            self.logger.warning("\n请检查:")
            self.logger.warning(f"  - 数据根目录: {data_root}")
            self.logger.warning(f"  - 文件夹命名是否包含 'Bearing'")
            self.logger.warning(f"  - 是否有多级目录结构")
        
        self.logger.info("=" * 60)
        
        # 按目录深度排序（可选）
        bearing_folders.sort(key=lambda x: len(x.split(os.sep)))
        
        return bearing_folders
    
    def add_gaussian_noise(self, X, noise_level=0.05):
        """向特征添加高斯白噪声"""
        if noise_level <= 0:
            return X.copy()
        
        feature_std = np.std(X, axis=0)
        feature_std = np.where(feature_std < 1e-10, 1.0, feature_std)
        
        noise = np.random.normal(0, noise_level * feature_std, X.shape)
        
        return X + noise
    
    def process_single_bearing(self, bearing_folder: str) -> Optional[Dict]:
        """处理单个轴承文件夹"""
        self.logger.info(f"开始处理轴承: {bearing_folder}")
        
        bearing_name = bearing_folder.replace(os.sep, "_").replace("/", "_")
        bearing_output_dir = self.output_root / bearing_name
        bearing_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否跳过已处理的结果
        if self.config['skip_existing']:
            result_file = bearing_output_dir / f"results_{bearing_name}.pkl"
            if result_file.exists():
                self.logger.info(f"跳过已处理的轴承: {bearing_folder}")
                try:
                    results = joblib.load(result_file)
                    return self._extract_summary_info(bearing_folder, results['models_results'])
                except:
                    pass
        
        try:
            # 1. 加载数据（现在返回signal, rul, hi）
            bearing_path = os.path.join(self.config['data_root'], bearing_folder)
            
            if not os.path.exists(bearing_path):
                self.logger.error(f"轴承路径不存在: {bearing_path}")
                return None
            
            data_loader = XJTUDataLoader(vibration_column=self.config['vibration_column'])
            signal, rul, hi = data_loader.load_bearing_data(bearing_path)
            
            if signal is None or rul is None or hi is None:
                self.logger.error(f"数据加载失败: {bearing_folder}")
                return None
            
            self.logger.info(f"数据加载成功: 信号长度={len(signal):,}, RUL标签数={len(rul)}")
            self.logger.info(f"HI范围: [{hi.min():.3f}, {hi.max():.3f}]")
            
            # 2. 创建数据处理器
            processor = DataProcessorFactory.create_processor('standard', self.config)
            
            # 3. 提取特征（现在需要hi数组）
            features, rul_labels, hi_labels = processor.create_dataset(signal, rul, hi)
            
            if len(features) == 0:
                self.logger.error(f"特征提取失败: {bearing_folder}")
                return None
            
            self.logger.info(f"特征提取成功: {len(features)} 个样本, {features.shape[1]} 个特征")
            
            # 4. 数据预处理
            valid_mask = ~np.isnan(features).any(axis=1) & ~np.isnan(rul_labels) & ~np.isnan(hi_labels)
            features = features[valid_mask]
            rul_labels = rul_labels[valid_mask]
            hi_labels = hi_labels[valid_mask]
            
            if len(features) < 100:
                self.logger.warning(f"样本数量较少: {len(features)} 个样本")
            
            features_scaled = processor.preprocess_features(features, fit=True)
            rul_labels_scaled = processor.preprocess_labels(rul_labels, fit=True)
            # HI不需要归一化
            
            # 5. 划分数据集
            # 使用train_test_split多次来实现
            X_train, X_temp, y_rul_train, y_rul_temp, y_hi_train, y_hi_temp = train_test_split(
                features_scaled, rul_labels_scaled, hi_labels,
                test_size=self.config['test_size'], 
                random_state=self.config['random_seed']
            )
            
            # 从temp中划分验证集和测试集
            X_val, X_test, y_rul_val, y_rul_test, y_hi_val, y_hi_test = train_test_split(
                X_temp, y_rul_temp, y_hi_temp,
                test_size=self.config['val_split'],
                random_state=self.config['random_seed']
            )
            
            self.logger.info(f"数据集划分: 训练集={X_train.shape[0]}, 验证集={X_val.shape[0]}, 测试集={X_test.shape[0]}")
            
            # 6. 训练和评估多个模型
            models_results, model_predictions = self._train_and_evaluate_all_models(
                X_train, y_rul_train, y_hi_train,
                X_val, y_rul_val, y_hi_val,
                X_test, y_rul_test, y_hi_test,
                processor, bearing_output_dir, bearing_name
            )
            
            if not models_results:
                self.logger.error(f"所有模型训练失败: {bearing_folder}")
                return None
            
            # 7. 创建单轴承可视化
            if self.config.get('save_per_bearing_comparisons', True):
                self._create_per_bearing_visualizations(
                    model_predictions, processor, y_rul_test, y_hi_test,
                    bearing_name, bearing_output_dir
                )
            
            # 8. 保存结果
            if self.config['save_scalers']:
                scaler_dir = bearing_output_dir / "scalers"
                processor.save_scalers(str(scaler_dir))
            
            # 准备结果数据
            results = {
                'bearing_folder': bearing_folder,
                'bearing_name': bearing_name,
                'models_results': models_results,
                'model_predictions': model_predictions,
                'data_info': {
                    'total_samples': len(features),
                    'train_samples': X_train.shape[0],
                    'val_samples': X_val.shape[0],
                    'test_samples': X_test.shape[0],
                    'signal_length': len(signal),
                    'rul_range': [float(rul_labels.min()), float(rul_labels.max())],
                    'hi_range': [float(hi_labels.min()), float(hi_labels.max())]
                },
                'processing_info': {
                    'noise_levels': self.config['noise_levels'],
                    'models_tested': list(models_results.keys()),
                    'device': str(DEVICE)
                }
            }
            
            # 保存结果文件
            result_file = bearing_output_dir / f"results_{bearing_name}.pkl"
            joblib.dump(results, result_file)
            
            # 保存为JSON便于阅读
            json_file = bearing_output_dir / f"results_{bearing_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                results_for_json = self._prepare_results_for_json(results)
                json.dump(results_for_json, f, indent=2, ensure_ascii=False)
            
            # 9. 可视化
            if self.config['save_plots']:
                VisualizationTool.visualize_model_comparison(
                    models_results, bearing_output_dir, bearing_name
                )
            
            self.logger.info(f"轴承 {bearing_folder} 处理完成")
            
            # 提取汇总信息
            summary_info = self._extract_summary_info(bearing_folder, models_results, X_test.shape[0])
            return summary_info, results
            
        except Exception as e:
            self.logger.error(f"处理轴承 {bearing_folder} 时出错: {e}")
            traceback.print_exc()
            return None
    
    def _train_and_evaluate_all_models(self, X_train, y_rul_train, y_hi_train,
                                      X_val, y_rul_val, y_hi_val,
                                      X_test, y_rul_test, y_hi_test,
                                      processor, bearing_output_dir, bearing_name):
        """训练和评估所有模型"""
        models_results = {}
        model_predictions = {}
        models_to_compare = self.config['models_to_compare']
        noise_levels = self.config['noise_levels'] if self.config['robustness_test'] else [0.0]
        
        for model_type in models_to_compare:
            try:
                if model_type == 'mlp':
                    model_results, predictions = self._train_mlp_model(
                        X_train, y_rul_train, y_hi_train,
                        X_val, y_rul_val, y_hi_val,
                        X_test, y_rul_test, y_hi_test,
                        processor, bearing_output_dir, bearing_name, noise_levels
                    )
                    models_results['mlp'] = model_results
                    model_predictions['mlp'] = predictions
                    
                elif model_type == 'linear':
                    model_results, predictions = self._train_linear_model(
                        X_train, y_rul_train, y_hi_train,
                        X_test, y_rul_test, y_hi_test,
                        processor, bearing_output_dir, bearing_name, noise_levels
                    )
                    models_results['linear'] = model_results
                    model_predictions['linear'] = predictions
                    
            except Exception as e:
                self.logger.error(f"{model_type}模型训练失败: {e}")
                traceback.print_exc()
        
        return models_results, model_predictions
    
    def _train_mlp_model(self, X_train, y_rul_train, y_hi_train,
                        X_val, y_rul_val, y_hi_val,
                        X_test, y_rul_test, y_hi_test,
                        processor, bearing_output_dir, bearing_name, noise_levels):
        """训练MLP模型"""
        self.logger.info(f"训练MLP模型...")
        
        # 创建MLP模型
        input_dim = X_train.shape[1]
        model = ModelFactory.create_model('mlp', self.config, input_dim=input_dim)
        
        # 创建数据加载器
        batch_size = min(self.config['batch_size'], len(X_train))
        
        # 创建数据集
        train_dataset = RULDataset(X_train, y_rul_train, y_hi_train)
        val_dataset = RULDataset(X_val, y_rul_val, y_hi_val)
        test_dataset = RULDataset(X_test, y_rul_test, y_hi_test)
        
        # 获取配置参数
        pin_memory = self.config.get('pin_memory', True) and torch.cuda.is_available()
        num_workers = self.config.get('num_workers', 0)
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # 训练模型
        model_save_path = bearing_output_dir / f"best_mlp_model_{bearing_name}.pth"
        history, best_val_score = self.model_runner.train_model(
            model, train_loader, val_loader, str(model_save_path)
        )
        
        # 加载最佳模型
        if model_save_path.exists():
            checkpoint = torch.load(model_save_path, map_location=self.model_runner.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.move_to_device(self.model_runner.device)
        
        # 保存训练历史
        history_file = bearing_output_dir / f"mlp_training_history_{bearing_name}.pkl"
        joblib.dump(history, history_file)
        
        # 评估MLP在不同噪声水平下的性能
        mlp_results = {}
        predictions_dict = {}
        
        for noise_level in noise_levels:
            if noise_level == 0.0:
                X_test_current = X_test
                test_desc = "clean"
            else:
                X_test_current = self.add_gaussian_noise(X_test, noise_level)
                test_desc = f"noisy_{int(noise_level*100)}pct"
            
            # 创建测试数据加载器
            test_dataset = RULDataset(X_test_current, y_rul_test, y_hi_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # 【修复】评估模型 - 现在只返回3个值
            metrics_dict, rul_preds, rul_labels = self.model_runner.evaluate_model(
                model, test_loader, processor, self.model_runner.device
            )
            
            # 存储预测结果
            if test_desc == "clean":
                predictions_dict['clean'] = {
                    'rul_predictions': rul_preds,
                    'rul_true_values': rul_labels
                }
            
            mlp_results[test_desc] = metrics_dict
        
        result = {
            'results': mlp_results,
            'training_info': {
                'best_val_score': best_val_score,
                'parameters': model.get_parameter_count(),
                'epochs_trained': len(history['train_total_loss']),
                'device': str(self.model_runner.device)
            }
        }
        
        self.logger.info(f"MLP训练完成，最佳验证RUL-R²: {best_val_score:.4f}")
        return result, predictions_dict
    
    def _train_linear_model(self, X_train, y_rul_train, y_hi_train,
                          X_test, y_rul_test, y_hi_test,
                          processor, bearing_output_dir, bearing_name, noise_levels):
        """训练线性回归模型 - 简化版，不包含HI"""
        self.logger.info(f"训练线性回归模型...")
        
        # 线性模型只预测RUL，不预测HI
        linear_model_type = self.config.get('linear_model_type', 'ridge')
        
        try:
            if linear_model_type == 'linear':
                model = LinearRegression()
            elif linear_model_type == 'ridge':
                model = Ridge(alpha=self.config.get('ridge_alpha', 1.0))
            elif linear_model_type == 'lasso':
                model = Lasso(alpha=self.config.get('lasso_alpha', 0.1))
            else:
                model = LinearRegression()
            
            # 训练模型
            model.fit(X_train, y_rul_train)
            
            # 保存模型
            model_path = bearing_output_dir / f"linear_model_{bearing_name}.joblib"
            joblib.dump(model, model_path)
            
            # 评估线性模型在不同噪声水平下的性能
            linear_results = {}
            predictions_dict = {}
            
            for noise_level in noise_levels:
                if noise_level == 0.0:
                    X_test_current = X_test
                    test_desc = "clean"
                else:
                    X_test_current = self.add_gaussian_noise(X_test, noise_level)
                    test_desc = f"noisy_{int(noise_level*100)}pct"
                
                # 预测
                y_pred_norm = model.predict(X_test_current)
                
                # 反归一化
                y_pred = processor.inverse_transform_labels(y_pred_norm)
                y_true = processor.inverse_transform_labels(y_rul_test)
                
                # 存储预测结果
                if test_desc == "clean":
                    predictions_dict['clean'] = {
                        'rul_predictions': y_pred,
                        'rul_true_values': y_true
                    }
                
                # 计算指标
                metrics_dict = self.model_runner._calculate_metrics(y_pred, y_true)
                linear_results[test_desc] = metrics_dict
            
            result = {
                'results': linear_results,
                'training_info': {
                    'model_type': type(model).__name__,
                    'coefficients': len(model.coef_) if hasattr(model, 'coef_') else 0,
                    'intercept': float(model.intercept_) if hasattr(model, 'intercept_') else 0.0
                }
            }
            
            self.logger.info(f"线性回归训练完成，R²: {linear_results['clean']['r2']:.4f}")
            return result, predictions_dict
            
        except Exception as e:
            self.logger.error(f"线性模型训练失败: {e}")
            
            # 尝试替代方案：手动实现Ridge回归
            try:
                from scipy import linalg
                
                alpha = self.config.get('ridge_alpha', 1.0)
                n_samples, n_features = X_train.shape
                
                X_train_bias = np.c_[np.ones(n_samples), X_train]
                reg_matrix = alpha * np.eye(n_features + 1)
                reg_matrix[0, 0] = 0
                
                coef = linalg.solve(
                    X_train_bias.T @ X_train_bias + reg_matrix,
                    X_train_bias.T @ y_rul_train
                )
                
                class ManualRidgeModel:
                    def __init__(self, coef):
                        self.coef_ = coef[1:]
                        self.intercept_ = coef[0]
                    
                    def predict(self, X):
                        return X @ self.coef_ + self.intercept_
                
                model = ManualRidgeModel(coef)
                
                model_path = bearing_output_dir / f"manual_ridge_{bearing_name}.joblib"
                joblib.dump(model, model_path)
                
                linear_results = {}
                predictions_dict = {}
                
                for noise_level in noise_levels:
                    if noise_level == 0.0:
                        X_test_current = X_test
                        test_desc = "clean"
                    else:
                        X_test_current = self.add_gaussian_noise(X_test, noise_level)
                        test_desc = f"noisy_{int(noise_level*100)}pct"
                    
                    y_pred_norm = model.predict(X_test_current)
                    y_pred = processor.inverse_transform_labels(y_pred_norm)
                    y_true = processor.inverse_transform_labels(y_rul_test)
                    
                    if test_desc == "clean":
                        predictions_dict['clean'] = {
                            'rul_predictions': y_pred,
                            'rul_true_values': y_true
                        }
                    
                    metrics_dict = self.model_runner._calculate_metrics(y_pred, y_true)
                    linear_results[test_desc] = metrics_dict
                
                result = {
                    'results': linear_results,
                    'training_info': {
                        'model_type': 'ManualRidge',
                        'coefficients': len(model.coef_),
                        'intercept': float(model.intercept_)
                    }
                }
                
                self.logger.info(f"手动Ridge回归训练完成")
                return result, predictions_dict
                
            except Exception as e2:
                self.logger.error(f"所有线性回归方法都失败: {e2}")
                return {}, {}
    
    def _create_per_bearing_visualizations(self, model_predictions, processor, y_rul_test_scaled, y_hi_test,
                                        bearing_name, bearing_output_dir):
        """创建单轴承可视化，包括RUL和HI曲线"""
        try:
            # 创建可视化目录
            viz_dir = bearing_output_dir / "per_bearing_visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 反归一化测试集RUL标签
            y_rul_test_true = processor.inverse_transform_labels(y_rul_test_scaled)
            
            # 准备各模型预测结果
            models_rul_pred_dict = {}
            
            for model_name, pred_dict in model_predictions.items():
                if 'clean' in pred_dict:
                    if 'rul_predictions' in pred_dict['clean']:
                        models_rul_pred_dict[model_name] = pred_dict['clean']['rul_predictions']
            
            # 创建RUL趋势对比图
            if models_rul_pred_dict:
                rul_trend_path = self.rul_evaluator.create_rul_trend_comparison(
                    models_predictions=models_rul_pred_dict,
                    true_rul=y_rul_test_true,
                    bearing_name=bearing_name,
                    output_dir=str(viz_dir)
                )
                if rul_trend_path:
                    self.logger.info(f"RUL趋势对比图已保存: {rul_trend_path}")
            
            # 【修改】HI曲线图 - 使用配置决定是否生成
            if self.config.get('enable_hi_visualization', True):
                # 由于不再有HI预测，可以选择生成真实HI曲线
                hi_curve_path = self.rul_evaluator.create_health_indicator_curves(
                    true_his=y_hi_test,
                    pred_his=y_hi_test,  # 使用真实值作为参考
                    bearing_name=bearing_name,
                    output_dir=str(viz_dir),
                    title_suffix="(真实HI)"
                )
                if hi_curve_path:
                    self.logger.info(f"真实HI曲线图已保存: {hi_curve_path}")
            
            # 为每个模型创建残差分析图
            for model_name, pred_dict in model_predictions.items():
                if 'clean' in pred_dict and 'rul_predictions' in pred_dict['clean']:
                    residual_path = self.rul_evaluator.create_residual_analysis(
                        model_name=model_name,
                        predictions=pred_dict['clean']['rul_predictions'],
                        true_rul=y_rul_test_true,
                        bearing_name=bearing_name,
                        output_dir=str(viz_dir)
                    )
                    if residual_path:
                        self.logger.info(f"{model_name}残差分析图已保存: {residual_path}")
                        
        except Exception as e:
            self.logger.error(f"创建单轴承可视化失败: {e}")
            traceback.print_exc()
 
    def _extract_summary_info(self, bearing_folder, models_results, test_samples=0):
        """从模型结果中提取汇总信息（只保留4个核心指标）"""
        summary = {
            'bearing': bearing_folder,
            'test_samples': test_samples,
        }
        
        for model_name, model_info in models_results.items():
            for test_desc, metrics_dict in model_info['results'].items():
                prefix = f"{model_name}_{test_desc}_"
                # 只保留4个核心指标
                for metric_name in ['r2', 'mse', 'rmse', 'mae']:
                    if metric_name in metrics_dict:
                        summary[f"{prefix}{metric_name}"] = metrics_dict[metric_name]
        
        # 添加模型特定信息
        if 'mlp' in models_results:
            summary['mlp_parameters'] = models_results['mlp']['training_info']['parameters']
            summary['mlp_epochs'] = models_results['mlp']['training_info']['epochs_trained']
            summary['mlp_device'] = models_results['mlp']['training_info'].get('device', 'cpu')
        
        if 'linear' in models_results:
            summary['linear_coefficients'] = models_results['linear']['training_info']['coefficients']
        
        return summary
    
    def _prepare_results_for_json(self, obj):
        """准备结果数据用于JSON序列化"""
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                processed_value = self._prepare_results_for_json(value)
                if processed_value is not None:
                    result[key] = processed_value
            return result
        elif isinstance(obj, list):
            result = []
            for item in obj:
                processed_item = self._prepare_results_for_json(item)
                if processed_item is not None:
                    result.append(processed_item)
            return result
        elif isinstance(obj, (np.ndarray, np.generic)):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj.item()
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif obj is None:
            return None
        else:
            try:
                return str(obj)
            except:
                return None
    
    def process_all_bearings(self):
        """处理所有轴承文件夹"""
        bearing_folders = self.find_bearing_folders()
        
        if not bearing_folders:
            self.logger.error("未找到任何轴承文件夹")
            return
        
        self.logger.info(f"开始批量处理 {len(bearing_folders)} 个轴承...")
        
        all_results = []
        
        for i, bearing_folder in enumerate(bearing_folders, 1):
            self.logger.info(f"处理进度: {i}/{len(bearing_folders)} - {bearing_folder}")
            
            try:
                result = self.process_single_bearing(bearing_folder)
                
                if result:
                    summary_info, detailed_result = result
                    self.results_summary.append(summary_info)
                    all_results.append(detailed_result)
                else:
                    self.logger.warning(f"轴承 {bearing_folder} 处理失败，跳过")
                    
            except Exception as e:
                self.logger.error(f"处理轴承 {bearing_folder} 时发生未捕获的异常: {e}")
                traceback.print_exc()
        
        # 汇总结果
        if self.results_summary:
            self._summarize_results(all_results)
    
    def _summarize_results(self, all_results):
        """汇总所有轴承的处理结果"""
        if not self.results_summary:
            self.logger.warning("没有可汇总的结果")
            return
        
        # 创建DataFrame
        df_summary = pd.DataFrame(self.results_summary)
        
        # 保存为CSV
        summary_csv_path = self.output_root / "enhanced_batch_results_summary.csv"
        df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        
        # 保存为Excel
        summary_excel_path = self.output_root / "enhanced_batch_results_summary.xlsx"
        df_summary.to_excel(summary_excel_path, index=False)
        
        # 创建跨轴承可视化
        if self.config.get('save_cross_bearing_summaries', True):
            self._create_cross_bearing_visualizations(all_results)
        
        # 打印汇总信息
        self.logger.info("\n" + "="*80)
        self.logger.info("增强版批量处理汇总结果")
        self.logger.info("="*80)
        
        # 打印关键指标
        key_metrics = [col for col in df_summary.columns if any(x in col for x in ['r2_clean', 'hi_r2'])]
        for metric in key_metrics:
            if metric in df_summary.columns:
                self.logger.info(f"{metric}: 平均={df_summary[metric].mean():.4f}, 标准差={df_summary[metric].std():.4f}")
        
        self.logger.info(f"\n详细结果:")
        for idx, row in df_summary.iterrows():
            model_perf = []
            for metric in key_metrics:
                if metric in row:
                    model_name = metric.split('_')[0] if '_' in metric else metric
                    model_perf.append(f"{model_name}: {row[metric]:.4f}")
            self.logger.info(f"{idx+1}. 轴承: {row['bearing']}, {', '.join(model_perf)}")
        
        self.logger.info(f"\n汇总文件已保存:")
        self.logger.info(f"  CSV: {summary_csv_path}")
        self.logger.info(f"  Excel: {summary_excel_path}")
    
    def _create_cross_bearing_visualizations(self, all_results):
        """创建跨轴承可视化"""
        try:
            viz_dir = self.output_root / "cross_bearing_visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            # 提取跨轴承指标数据
            cross_bearing_metrics = {}
            
            for result in all_results:
                bearing_name = result.get('bearing_name', 'unknown')
                models_results = result.get('models_results', {})
                
                for model_name, model_info in models_results.items():
                    if model_name not in cross_bearing_metrics:
                        cross_bearing_metrics[model_name] = {}
                    
                    # 提取clean测试集的指标
                    if 'clean' in model_info['results']:
                        metrics_dict = model_info['results']['clean']
                        for metric_name, metric_value in metrics_dict.items():
                            if metric_name not in cross_bearing_metrics[model_name]:
                                cross_bearing_metrics[model_name][metric_name] = []
                            cross_bearing_metrics[model_name][metric_name].append(metric_value)
            
            # 创建雷达图
            if cross_bearing_metrics:
                # 计算平均指标
                avg_metrics = {}
                for model_name, metrics_dict in cross_bearing_metrics.items():
                    avg_metrics[model_name] = {}
                    for metric_name, values in metrics_dict.items():
                        if values:
                            avg_metrics[model_name][metric_name] = np.mean(values)
                
                radar_path = viz_dir / "model_performance_radar.png"
                self.model_comparison_visualizer.create_radar_chart(
                    model_metrics=avg_metrics,
                    title="模型性能雷达图（跨轴承平均）",
                    output_path=str(radar_path)
                )
                
                # 创建指标对比图
                bar_chart_path = viz_dir / "model_performance_bars.png"
                self.model_comparison_visualizer.create_metrics_bar_chart(
                    model_metrics=avg_metrics,
                    title="模型性能指标对比（跨轴承平均）",
                    output_path=str(bar_chart_path)
                )
                
                # 创建箱线图
                boxplot_path = viz_dir / "cross_bearing_boxplots.png"
                self.model_comparison_visualizer.create_cross_bearing_boxplots(
                    all_results=all_results,
                    metrics=['r2', 'rmse', 'mae', 'hi_r2'],
                    title="跨轴承模型性能箱线图",
                    output_path=str(boxplot_path)
                )
                
                # 提取预测数据用于误差分布分析
                all_predictions = {}
                all_true_values = []
                model_names = []
                
                for result in all_results:
                    model_predictions = result.get('model_predictions', {})
                    for model_name, pred_dict in model_predictions.items():
                        if 'clean' in pred_dict:
                            if model_name not in all_predictions:
                                all_predictions[model_name] = []
                                model_names.append(model_name)
                            all_predictions[model_name].append(pred_dict['clean']['rul_predictions'])
                            all_true_values.append(pred_dict['clean']['rul_true_values'])
                
                # 创建误差分布对比图
                error_dist_path = viz_dir / "error_distribution_comparison.png"
                self.model_comparison_visualizer.create_error_distribution_comparison(
                    all_predictions=all_predictions,
                    all_labels=all_true_values,
                    model_names=model_names,
                    title="模型误差分布对比（跨轴承）",
                    output_path=str(error_dist_path)
                )
                
                self.logger.info(f"跨轴承可视化已保存到: {viz_dir}")
                
        except Exception as e:
            self.logger.error(f"创建跨轴承可视化失败: {e}")
            traceback.print_exc()


class EnhancedMultiModalBatchProcessor(EnhancedBatchRULProcessor):
    """增强版多模态批量处理器，继承自 EnhancedBatchRULProcessor"""
    
    def __init__(self, config: BatchConfig, logger: logging.Logger = None):
        super().__init__(config, logger)
        
        # 添加多模态配置
        self.multimodal_config = {
            'cwt_wavelet': 'cmor1.0-1.0',
            'cwt_scales': None,
            'cwt_image_shape': (1, 64, 64),
            'fusion_method': 'late',
            'cnn_channels': [16, 32, 64],
            'lstm_hidden_size': 64,
            'lstm_num_layers': 2,
        }
        
        # 添加 visualization_dpi 属性
        self.visualization_dpi = self.config.get('visualization_dpi', 300)
    
    def process_single_bearing(self, bearing_folder: str) -> Optional[Dict]:
        """多模态版本：处理单个轴承文件夹"""
        self.logger.info(f"开始多模态处理轴承: {bearing_folder}")
        
        bearing_name = bearing_folder.replace(os.sep, "_").replace("/", "_")
        bearing_output_dir = self.output_root / f"multimodal_{bearing_name}"
        bearing_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. 加载数据
            bearing_path = os.path.join(self.config['data_root'], bearing_folder)
            
            if not os.path.exists(bearing_path):
                self.logger.error(f"轴承路径不存在: {bearing_path}")
                return None
            
            data_loader = XJTUDataLoader(vibration_column=self.config['vibration_column'])
            signal, rul, hi = data_loader.load_bearing_data(bearing_path)
            
            if signal is None or rul is None or hi is None:
                self.logger.error(f"数据加载失败: {bearing_folder}")
                return None
            
            # 2. 创建多模态数据处理器
            processor = DataProcessorFactory.create_processor(
                'multimodal', 
                self.config,
                cwt_image_shape=self.multimodal_config['cwt_image_shape']
            )
            
            # 3. 提取多模态特征
            signals_list, cwt_images_list, rul_labels_list, hi_labels_list = processor.create_dataset(
                full_signal=signal, 
                rul_array=rul,
                hi_array=hi,
                bearing_output_dir=str(bearing_output_dir),
                bearing_name=bearing_name
            )
            
            if len(signals_list) == 0:
                self.logger.error(f"多模态特征提取失败: {bearing_folder}")
                return None
            
            # 4. 数据预处理和划分
            # 预处理RUL标签
            rul_labels_scaled = processor.preprocess_labels(rul_labels_list, fit=True)
            # HI不需要归一化
            
            # 转换为numpy数组
            signals_array = np.array(signals_list)
            cwt_images_array = np.array(cwt_images_list)
            hi_labels_array = np.array(hi_labels_list)
            
            # 划分数据集
            X_train_sig, X_temp_sig, X_train_cwt, X_temp_cwt, y_rul_train, y_rul_temp, y_hi_train, y_hi_temp = train_test_split(
                signals_array, cwt_images_array, rul_labels_scaled, hi_labels_array,
                test_size=self.config['test_size'],
                random_state=self.config['random_seed']
            )
            
            # 进一步划分验证集和测试集
            split_idx = int(len(X_temp_sig) * (1 - self.config['val_split']))
            X_val_sig, X_test_sig = X_temp_sig[:split_idx], X_temp_sig[split_idx:]
            X_val_cwt, X_test_cwt = X_temp_cwt[:split_idx], X_temp_cwt[split_idx:]
            y_rul_val, y_rul_test = y_rul_temp[:split_idx], y_rul_temp[split_idx:]
            y_hi_val, y_hi_test = y_hi_temp[:split_idx], y_hi_temp[split_idx:]
            
            self.logger.info(f"多模态数据集划分: "
                            f"训练集={len(X_train_sig)}, "
                            f"验证集={len(X_val_sig)}, "
                            f"测试集={len(X_test_sig)}")
            
            # 5. 训练多模态模型
            multimodal_results, multimodal_predictions = self._train_multimodal_model(
                X_train_sig, X_train_cwt, y_rul_train, y_hi_train,
                X_val_sig, X_val_cwt, y_rul_val, y_hi_val,
                X_test_sig, X_test_cwt, y_rul_test, y_hi_test,
                processor, bearing_output_dir, bearing_name
            )
            
            if not multimodal_results:
                self.logger.error(f"多模态模型训练失败: {bearing_folder}")
                return None
            
            # 6. 创建单轴承可视化
            if self.config.get('save_per_bearing_comparisons', True):
                # 准备预测结果
                model_predictions = {
                    'multimodal': multimodal_predictions
                }
                
                self._create_per_bearing_visualizations(
                    model_predictions, processor, y_rul_test, y_hi_test,
                    bearing_name, bearing_output_dir
                )
            
            # 7. 保存结果
            results = {
                'bearing_folder': bearing_folder,
                'bearing_name': bearing_name,
                'models_results': multimodal_results,
                'model_predictions': multimodal_predictions,
                'data_info': {
                    'total_samples': len(signals_list),
                    'train_samples': len(X_train_sig),
                    'val_samples': len(X_val_sig),
                    'test_samples': len(X_test_sig),
                    'signal_length': len(signal),
                    'rul_range': [float(min(rul_labels_list)), float(max(rul_labels_list))],
                    'hi_range': [float(min(hi_labels_list)), float(max(hi_labels_list))]
                },
                'multimodal_config': self.multimodal_config,
                'device': str(DEVICE)
            }
            
            # 保存结果文件
            result_file = bearing_output_dir / f"multimodal_results_{bearing_name}.pkl"
            joblib.dump(results, result_file)
            
            # 保存为JSON
            json_file = bearing_output_dir / f"multimodal_results_{bearing_name}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                results_for_json = self._prepare_results_for_json(results)
                json.dump(results_for_json, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"轴承 {bearing_folder} 多模态处理完成")
            
            return results
            
        except Exception as e:
            self.logger.error(f"处理轴承 {bearing_folder} 时出错: {e}")
            traceback.print_exc()
            return None
    
    def _train_multimodal_model(self, X_train_sig, X_train_cwt, y_rul_train, y_hi_train,
                            X_val_sig, X_val_cwt, y_rul_val, y_hi_val,
                            X_test_sig, X_test_cwt, y_rul_test, y_hi_test,
                            processor, bearing_output_dir, bearing_name):
        """训练多模态模型"""
        self.logger.info(f"训练多模态模型...")
        
        try:
            # 创建多模态模型
            input_signal_length = len(X_train_sig[0]) if len(X_train_sig) > 0 else 1024
            cwt_image_shape = self.multimodal_config['cwt_image_shape']
            
            model = ModelFactory.create_model(
                'multimodal',
                self.config,
                cwt_image_shape=cwt_image_shape,
                signal_length=input_signal_length,
                cnn_channels=self.multimodal_config['cnn_channels'],
                lstm_hidden_size=self.multimodal_config['lstm_hidden_size'],
                lstm_num_layers=self.multimodal_config['lstm_num_layers'],
                fusion_method=self.multimodal_config['fusion_method']
            )
            
            # 创建数据集
            train_dataset = MultiModalDataset(X_train_sig, X_train_cwt, y_rul_train, y_hi_train)
            val_dataset = MultiModalDataset(X_val_sig, X_val_cwt, y_rul_val, y_hi_val)
            test_dataset = MultiModalDataset(X_test_sig, X_test_cwt, y_rul_test, y_hi_test)
            
            # 创建数据加载器
            batch_size = min(self.config['batch_size'], len(X_train_sig))
            pin_memory = self.config.get('pin_memory', True) and torch.cuda.is_available()
            num_workers = self.config.get('num_workers', 0)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            test_loader = DataLoader(
                test_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
            
            # 训练模型
            model_save_path = bearing_output_dir / f"best_multimodal_model_{bearing_name}.pth"
            history, best_val_score = self.model_runner.train_model(
                model, train_loader, val_loader, str(model_save_path)
            )
            
            # 加载最佳模型
            if model_save_path.exists():
                checkpoint = torch.load(model_save_path, map_location=self.model_runner.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.move_to_device(self.model_runner.device)
            
            # 保存训练历史
            history_file = bearing_output_dir / f"multimodal_training_history_{bearing_name}.pkl"
            joblib.dump(history, history_file)
            
            # 【修复】评估模型 - 现在只返回3个值
            metrics_dict, rul_preds, rul_labels = self.model_runner.evaluate_model(
                model, test_loader, processor, self.model_runner.device
            )
            
            results = {
                'multimodal': {
                    'results': {'clean': metrics_dict},
                    'training_info': {
                        'best_val_score': best_val_score,
                        'parameters': model.get_parameter_count(),
                        'epochs_trained': len(history['train_total_loss']),
                        'fusion_method': self.multimodal_config['fusion_method'],
                        'device': str(self.model_runner.device)
                    }
                }
            }
            
            # 【修复】预测结果只包含RUL相关数据
            predictions = {
                'clean': {
                    'rul_predictions': rul_preds,
                    'rul_true_values': rul_labels
                }
            }
            
            self.logger.info(f"多模态模型训练完成，最佳验证RUL-R²: {best_val_score:.4f}")
            self.logger.info(f"模型参数量: {model.get_parameter_count():,}")
            
            # 使用全面评估指标打印结果
            print_metrics_summary(metrics_dict, "多模态模型测试集")
            
            # 可视化结果
            if self.config['save_plots']:
                fig = VisualizationTool.visualize_results(
                    rul_preds, rul_labels, metrics_dict, 
                    title=f"多模态模型预测结果 - {bearing_name}"
                )
                if fig:
                    plot_path = bearing_output_dir / f"multimodal_results_{bearing_name}.png"
                    fig.savefig(plot_path, dpi=self.visualization_dpi, bbox_inches='tight')
                    plt.close(fig)
            
            return results, predictions
            
        except Exception as e:
            self.logger.error(f"多模态模型训练失败: {e}")
            traceback.print_exc()
            return {}, {}
    
    def generate_single_bearing_model_comparison(self, bearing_name, results_base_dir=None):
        """
        生成单个轴承的模型性能对比图
        
        【修改】增强版：支持更多搜索模式，自动识别不同模型的结果
        """
        if results_base_dir is None:
            results_base_dir = self.config['output_root']
        
        results_base_dir = Path(results_base_dir)
        
        self.logger.info(f"开始生成轴承 {bearing_name} 的模型性能对比图")
        self.logger.info(f"扫描目录: {results_base_dir}")
        
        try:
            # 1. 明确查找所有相关文件夹
            model_folders_info = []
            
            # 【修改】定义更多搜索模式
            search_patterns = [
                # 标准命名
                (bearing_name, "mlp"),
                (f"multimodal_{bearing_name}", "multimodal"),
                (f"mlp_{bearing_name}", "mlp"),
                (f"linear_{bearing_name}", "linear"),
                (f"ridge_{bearing_name}", "ridge"),
                (f"lasso_{bearing_name}", "lasso"),
                
                # 处理35Hz12kN_Bearing1_1格式
                (f"multimodal_{bearing_name}", "multimodal"),
                (f"{bearing_name}", "mlp"),
                
                # 处理带有下划线的变体
                (bearing_name.replace('_', ''), "mlp_variant"),
                (f"multimodal_{bearing_name.replace('_', '')}", "multimodal_variant"),
            ]
            
            # 首先尝试精确匹配
            for folder_name, model_type in search_patterns:
                folder_path = results_base_dir / folder_name
                if folder_path.exists() and folder_path.is_dir():
                    self.logger.info(f"找到 {model_type} 模型文件夹: {folder_path}")
                    model_folders_info.append((model_type, str(folder_path)))
            
            # 【新增】如果精确匹配不够，进行模糊搜索
            if len(model_folders_info) < 2:
                self.logger.info("精确匹配不足2个模型，开始模糊搜索...")
                
                # 搜索所有包含轴承名的文件夹
                all_folders = list(results_base_dir.glob(f"*{bearing_name}*"))
                
                for folder_path in all_folders:
                    if folder_path.is_dir() and str(folder_path) not in [info[1] for info in model_folders_info]:
                        folder_name = folder_path.name
                        
                        # 推断模型类型
                        if "multimodal" in folder_name.lower():
                            model_type = "multimodal"
                        elif "linear" in folder_name.lower() or "ridge" in folder_name.lower() or "lasso" in folder_name.lower():
                            model_type = "linear"
                        elif "mlp" in folder_name.lower():
                            model_type = "mlp"
                        elif "cnn" in folder_name.lower():
                            model_type = "cnn"
                        elif "lstm" in folder_name.lower():
                            model_type = "lstm"
                        elif "transformer" in folder_name.lower():
                            model_type = "transformer"
                        else:
                            model_type = "unknown"
                        
                        self.logger.info(f"找到其他模型文件夹 ({model_type}): {folder_name}")
                        model_folders_info.append((model_type, str(folder_path)))
            
            if not model_folders_info:
                self.logger.error(f"未找到任何包含轴承名称 {bearing_name} 的文件夹")
                return None
            
            self.logger.info(f"总共找到 {len(model_folders_info)} 个模型文件夹")
            
            # 2. 为每个模型加载指标
            models_results = {}
            
            for model_type, folder_path in model_folders_info:
                try:
                    self.logger.info(f"处理模型: {model_type}, 文件夹: {folder_path}")
                    
                    metrics = self._load_metrics_smart(folder_path, bearing_name, model_type)
                    
                    if metrics and metrics.get('r2', 0) > 0:  # 【新增】只接受有效的R²
                        # 确保名称唯一
                        unique_name = model_type
                        counter = 1
                        while unique_name in models_results:
                            unique_name = f"{model_type}_{counter}"
                            counter += 1
                        
                        models_results[unique_name] = metrics
                        self.logger.info(f"成功加载 {unique_name}: R²={metrics.get('r2', 0):.4f}, "
                                    f"HI-R²={metrics.get('hi_r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.3f}")
                    else:
                        self.logger.warning(f"无法从 {folder_path} 加载有效指标")
                        
                except Exception as e:
                    self.logger.error(f"处理文件夹 {folder_path} 时出错: {e}")
                    continue
            
            # 3. 检查结果
            if len(models_results) < 2:
                self.logger.warning(f"只找到 {len(models_results)} 个有效模型，需要至少2个")
                if models_results:
                    self.logger.info(f"找到的模型: {list(models_results.keys())}")
                return None
            
            self.logger.info(f"成功加载 {len(models_results)} 个模型: {list(models_results.keys())}")
            
            # 4. 创建输出目录
            comparison_output_dir = results_base_dir / "single_bearing_comparisons" / bearing_name
            comparison_output_dir.mkdir(parents=True, exist_ok=True)
            
            # 5. 生成对比图
            chart_path = self._create_model_comparison_visualization(
                models_results=models_results,
                bearing_name=bearing_name,
                output_dir=str(comparison_output_dir)
            )
            
            if chart_path:
                self.logger.info(f"模型性能对比图已保存到: {chart_path}")
                
                # 6. 保存对比数据为JSON文件
                comparison_data = {
                    "bearing_name": bearing_name,
                    "models_results": models_results,
                    "generated_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chart_path": str(chart_path),
                    "total_models": len(models_results),
                    "search_patterns_used": [p[0] for p in search_patterns[:5]]  # 【新增】记录搜索模式
                }
                
                data_file = comparison_output_dir / f"model_comparison_data_{bearing_name}.json"
                with open(data_file, 'w', encoding='utf-8') as f:
                    json.dump(comparison_data, f, indent=2, ensure_ascii=False)
                
                self.logger.info(f"对比数据已保存到: {data_file}")
                
                return chart_path
            else:
                self.logger.error("生成对比图失败")
                return None
            
        except Exception as e:
            self.logger.error(f"生成轴承 {bearing_name} 的模型性能对比图时出错: {e}")
            traceback.print_exc()
            return None
    
    def _load_metrics_smart(self, folder_path, bearing_name, model_type):
        """
        智能加载指标 - 专门处理您的文件夹结构
        """
        try:
            folder_path_obj = Path(folder_path)
            
            # 查找所有可能的JSON文件
            json_files = []
            
            # 优先查找明确的评估文件
            priority_patterns = [
                f"results_{bearing_name}.json",
                f"{model_type}_results_{bearing_name}.json",
                f"evaluation_{bearing_name}.json",
                f"metrics_{bearing_name}.json",
                f"{bearing_name}_results.json",
                "results.json",
                "evaluation.json",
                "metrics.json",
            ]
            
            for pattern in priority_patterns:
                file_path = folder_path_obj / pattern
                if file_path.exists():
                    json_files.append(file_path)
            
            # 如果没有找到，查找所有JSON文件
            if not json_files:
                json_files = list(folder_path_obj.glob("*.json"))
            
            # 尝试加载每个JSON文件
            for json_file in json_files:
                try:
                    self.logger.debug(f"尝试加载JSON文件: {json_file}")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    metrics = self._extract_metrics_smart(data, model_type)
                    if metrics:
                        return metrics
                        
                except Exception as e:
                    self.logger.debug(f"加载JSON文件失败 {json_file}: {e}")
                    continue
            
            # 如果JSON失败，尝试PKL文件
            pkl_files = folder_path_obj.glob("*.pkl")
            for pkl_file in pkl_files:
                try:
                    self.logger.debug(f"尝试加载PKL文件: {pkl_file}")
                    data = joblib.load(pkl_file)
                    
                    metrics = self._extract_metrics_smart(data, model_type)
                    if metrics:
                        return metrics
                        
                except Exception as e:
                    self.logger.debug(f"加载PKL文件失败 {pkl_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"加载指标失败: {e}")
            return None
    
    def _extract_metrics_smart(self, data, model_type):
        """
        智能提取指标 - 处理不同数据结构
        """
        metrics = {}
        
        if not isinstance(data, dict):
            return metrics
        
        # 方法1: 直接查找指标
        if 'r2' in data or 'R2' in data:
            metrics['r2'] = data.get('r2', data.get('R2', 0))
            metrics['rmse'] = data.get('rmse', data.get('RMSE', 0))
            metrics['mae'] = data.get('mae', data.get('MAE', 0))
            metrics['mape'] = data.get('mape', data.get('MAPE', 0))
            metrics['hi_r2'] = data.get('hi_r2', 0)
            
            return metrics
        
        # 方法2: 查找models_results
        if 'models_results' in data:
            models_results = data['models_results']
            
            # 查找特定模型
            if model_type in models_results:
                model_data = models_results[model_type]
                return self._extract_metrics_from_model_data(model_data)
            
            # 查找'model'变体
            elif 'model' in models_results:
                model_data = models_results['model']
                return self._extract_metrics_from_model_data(model_data)
            
            # 使用第一个模型
            elif models_results:
                first_key = list(models_results.keys())[0]
                model_data = models_results[first_key]
                return self._extract_metrics_from_model_data(model_data)
        
        # 方法3: 查找evaluation_metrics
        if 'evaluation_metrics' in data:
            eval_metrics = data['evaluation_metrics']
            metrics['r2'] = eval_metrics.get('r2', 0)
            metrics['rmse'] = eval_metrics.get('rmse', 0)
            metrics['mae'] = eval_metrics.get('mae', 0)
            metrics['mape'] = eval_metrics.get('mape', 0)
            metrics['hi_r2'] = eval_metrics.get('hi_r2', 0)
            
            return metrics
        
        # 方法4: 在嵌套结构中查找
        def search_in_nested(obj, path=""):
            nonlocal metrics
            if isinstance(obj, dict):
                # 检查是否包含指标
                if any(key in obj for key in ['r2', 'R2', 'rmse', 'RMSE', 'mae', 'MAE', 'hi_r2', 'phm_score']):
                    metrics['r2'] = obj.get('r2', obj.get('R2', 0))
                    metrics['rmse'] = obj.get('rmse', obj.get('RMSE', 0))
                    metrics['mae'] = obj.get('mae', obj.get('MAE', 0))
                    metrics['mape'] = obj.get('mape', obj.get('MAPE', 0))
                    metrics['hi_r2'] = obj.get('hi_r2', 0)
                    
                    return True
                
                # 递归搜索
                for k, v in obj.items():
                    if search_in_nested(v, f"{path}.{k}"):
                        return True
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if search_in_nested(item, f"{path}[{i}]"):
                        return True
            return False
        
        if search_in_nested(data):
            return metrics
        
        return metrics
    
    def _extract_metrics_from_model_data(self, model_data):
        """从模型数据中提取指标"""
        metrics = {}
        
        if isinstance(model_data, dict):
            # 查找results -> clean
            if 'results' in model_data and 'clean' in model_data['results']:
                clean_results = model_data['results']['clean']
                metrics['r2'] = clean_results.get('r2', 0)
                metrics['rmse'] = clean_results.get('rmse', 0)
                metrics['mae'] = clean_results.get('mae', 0)
                metrics['mape'] = clean_results.get('mape', 0)
                metrics['hi_r2'] = clean_results.get('hi_r2', 0)
                
            # 直接包含指标
            elif 'r2' in model_data or 'R2' in model_data:
                metrics['r2'] = model_data.get('r2', model_data.get('R2', 0))
                metrics['rmse'] = model_data.get('rmse', model_data.get('RMSE', 0))
                metrics['mae'] = model_data.get('mae', model_data.get('MAE', 0))
                metrics['mape'] = model_data.get('mape', model_data.get('MAPE', 0))
                metrics['hi_r2'] = model_data.get('hi_r2', 0)
                
        
        return metrics
    
    def _create_model_comparison_visualization(self, models_results, bearing_name, output_dir):
        """
        创建模型性能对比可视化 - 添加指标说明
        """
        try:
            # 创建图形 - 稍微增加高度以容纳更多信息
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            model_names = list(models_results.keys())
            colors = plt.cm.Set2(range(len(model_names)))
            
            # 1. R²对比图
            ax1 = axes[0, 0]
            r2_values = [models_results[m].get('r2', 0) for m in model_names]
            bars1 = ax1.bar(range(len(model_names)), r2_values, color=colors, edgecolor='black', linewidth=1.5)
            
            ax1.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_ylabel(r'R $ ^2 $ 分数', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_title(r'R $ ^2 $ 分数对比\n(越大越好)', 
                        fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            
            ax1.set_xticks(range(len(model_names)))
            ax1.set_xticklabels([m.upper() for m in model_names], 
                            rotation=45, ha='right', 
                            fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax1.grid(True, alpha=0.3, linestyle='--')
            
            r2_max = max(r2_values) if r2_values else 0
            ax1.set_ylim(0, max(r2_max * 1.15, 0.1))
            
            for bar, value in zip(bars1, r2_values):
                height = bar.get_height()
                color = 'green' if value >= 0.8 else ('orange' if value >= 0.6 else 'red')
                ax1.annotate(f'{value:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, 
                            fontweight='bold', color=color)
            
            ax1.text(0.02, 0.98, '性能说明:\n≥0.8: 优秀\n0.6-0.8: 良好\n<0.6: 需改进',
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top',
                    fontproperties=fm.FontProperties(family=selected_font, size=9),
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            # 2. HI-R²对比图
            ax2 = axes[0, 1]
            hi_r2_values = [models_results[m].get('hi_r2', 0) for m in model_names]
            bars2 = ax2.bar(range(len(model_names)), hi_r2_values, color=colors, edgecolor='black', linewidth=1.5)
            
            ax2.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_ylabel('HI-R²分数', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_title('HI-R²分数对比\n(越大越好)', 
                        fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            
            ax2.set_xticks(range(len(model_names)))
            ax2.set_xticklabels([m.upper() for m in model_names], 
                            rotation=45, ha='right', 
                            fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax2.grid(True, alpha=0.3, linestyle='--')
            
            hi_r2_max = max(hi_r2_values) if hi_r2_values else 0
            ax2.set_ylim(0, max(hi_r2_max * 1.15, 0.1))
            
            for bar, value in zip(bars2, hi_r2_values):
                height = bar.get_height()
                color = 'green' if value >= 0.8 else ('orange' if value >= 0.6 else 'red')
                ax2.annotate(f'{value:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, 
                            fontweight='bold', color=color)
            
            # 3. RMSE对比图
            ax3 = axes[1, 0]
            rmse_values = [models_results[m].get('rmse', 0) for m in model_names]
            bars3 = ax3.bar(range(len(model_names)), rmse_values, color=colors, edgecolor='black', linewidth=1.5)
            
            ax3.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_ylabel('RMSE', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_title('RMSE对比\n(越小越好)', 
                        fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            
            ax3.set_xticks(range(len(model_names)))
            ax3.set_xticklabels([m.upper() for m in model_names], 
                            rotation=45, ha='right', 
                            fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax3.grid(True, alpha=0.3, linestyle='--')
            
            rmse_max = max(rmse_values) if rmse_values else 0
            ax3.set_ylim(0, rmse_max * 1.15)
            
            for bar, value in zip(bars3, rmse_values):
                height = bar.get_height()
                ax3.annotate(f'{value:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10,
                            fontweight='bold')
            
            # 4. MAE对比图（替代原来的PHM Score）
            ax4 = axes[1, 1]
            mae_values = [models_results[m].get('mae', 0) for m in model_names]
            bars4 = ax4.bar(range(len(model_names)), mae_values, color=colors, edgecolor='black', linewidth=1.5)

            ax4.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax4.set_ylabel('MAE', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax4.set_title('MAE对比\n(越小越好)', 
                        fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))

            ax4.set_xticks(range(len(model_names)))
            ax4.set_xticklabels([m.upper() for m in model_names], 
                            rotation=45, ha='right', 
                            fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax4.grid(True, alpha=0.3, linestyle='--')

            if mae_values:
                mae_max = max(mae_values)
                ax4.set_ylim(0, mae_max * 1.15)

            for bar, value in zip(bars4, mae_values):
                height = bar.get_height()
                ax4.annotate(f'{value:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10,
                            fontweight='bold')

            ax4.text(0.02, 0.98, 'MAE:\n越小越好',
                    transform=ax4.transAxes, fontsize=9,
                    verticalalignment='top',
                    fontproperties=fm.FontProperties(family=selected_font, size=9),
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            
            # 设置主标题
            models_count = len(models_results)
            best_model_r2 = max(models_results.items(), key=lambda x: x[1].get('r2', 0))[0]
            best_model_hi = max(models_results.items(), key=lambda x: x[1].get('hi_r2', 0))[0]
            best_r2 = models_results[best_model_r2].get('r2', 0)
            best_hi_r2 = models_results[best_model_hi].get('hi_r2', 0)
            
            plt.suptitle(f'{bearing_name} - 模型性能综合对比\n'
                        f'(共{models_count}个模型，最佳R²: {best_model_r2.upper()}={best_r2:.4f}, '
                        f'最佳HI-R²: {best_model_hi.upper()}={best_hi_r2:.4f})', 
                        fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'),
                        y=0.98)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            # 保存图像
            save_path = Path(output_dir) / f"model_comparison_{bearing_name}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            return save_path
            
        except Exception as e:
            self.logger.error(f"创建模型对比可视化时出错: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== 配置文件模板生成函数 ====================
def create_enhanced_multimodal_config_template():
    """创建增强版多模态配置文件模板"""
    template = """
# 增强版多模态批量RUL预测处理器配置文件
# 版本: 7.0 - 支持健康因子(HI)联合训练
# 评估指标: R², MSE, RMSE, MAE

# 数据路径配置
data_root: "E:/毕业设计/data"
output_root: "./batch_results"

# 文件匹配模式
pattern: ".*Bearing.*_.*"

# 工况选择
use_only_35khz: true

# 数据处理配置
vibration_column: "horizontal"
window_size: 4096
overlap_ratio: 0.75
sampling_rate: 25600

# ==================== RUL归一化配置 ====================
rul_normalization_mode: "global"
rul_global_max: 160

# ==================== 跨轴承评估配置 ====================
cross_bearing_eval: false              # 是否启用跨轴承评估
cross_bearing_mode: "leave_one_out"    # 评估模式: "leave_one_out" 或 "train_test_split"
cross_bearing_train_ratio: 0.8         # train_test_split模式下的训练集比例
cross_bearing_visualize: true          # 是否可视化跨轴承评估结果

# CWT图像配置
cwt_image_shape: [1, 128, 128]

# 训练配置
batch_size: 32
epochs: 100
patience: 15
min_delta: 0.001
learning_rate: 0.001
dropout: 0.3
weight_decay: 0.0001

# 验证平滑配置
validation_smoothing: true
validation_smoothing_alpha: 0.9
validation_frequency: 2

# GPU优化配置
use_gpu: true
pin_memory: true
num_workers: 4

# 数据划分配置
test_size: 0.3
val_split: 0.3
random_seed: 42

# 健康因子(HI)配置
rul_loss_weight: 1.0
hi_loss_weight: 1.0
use_sample_weighting: true
weighting_alpha: 2.0
enable_hi_visualization: true

# 模型架构配置
cnn_architecture: "simple"
pretrained_model_name: "resnet18"
signal_processor: "lstm"
transformer_config:
  d_model: 128
  nhead: 4
  num_layers: 2
  dim_feedforward: 256
prediction_head_dims: [128, 64, 32]

# 损失函数配置
loss_function: "mse"

# 学习率调度器配置
lr_scheduler: "plateau"
lr_scheduler_params:
  mode: "min"
  factor: 0.5
  patience: 5
  min_lr: 1e-6

# 评估指标配置
tolerance_threshold: 0.1

# 模型对比配置
compare_models: true
models_to_compare: ["mlp", "multimodal"]

# 鲁棒性验证配置
robustness_test: false
noise_levels: [0.0, 0.05]

# 可视化配置
save_cwt_images: true
cwt_visualization_points: "default"
cwt_image_dpi: 150
visualization_dpi: 300
save_per_bearing_comparisons: true
save_cross_bearing_summaries: true

# 输出配置
save_models: true
save_scalers: true
save_plots: true
skip_existing: false
"""
    
    os.makedirs("config_templates", exist_ok=True)
    template_path = "config_templates/enhanced_multimodal_config.yaml"
    
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"增强版多模态配置文件模板已创建: {template_path}")
    return template_path


def create_basic_config_template():
    """创建基础配置文件模板（向后兼容）"""
    template = """# 基础版批量RUL预测处理器配置文件
# 用于向后兼容

data_root: "E:/毕业设计/data"
output_root: "./batch_results"
pattern: ".*Bearing.*_.*"
vibration_column: "horizontal"
window_size: 1024
overlap_ratio: 0.75
sampling_rate: 25600
batch_size: 32
epochs: 100
patience: 15
learning_rate: 0.001
hidden_sizes: [64, 32, 16]
dropout: 0.3
test_size: 0.3
val_split: 0.5
random_seed: 42
use_gpu: true
pin_memory: true
num_workers: 4
save_models: true
save_scalers: true
save_plots: true
skip_existing: false
"""
    
    os.makedirs("config_templates", exist_ok=True)
    template_path = "config_templates/basic_config.yaml"
    
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"基础版配置文件模板已创建: {template_path}")
    return template_path



# ==================== 跨轴承评估器 ====================

class CrossBearingEvaluator:
    """
    跨轴承评估器 - 支持留一法和训练/测试划分
    用于验证模型的泛化能力，避免数据泄漏
    """
    
    def __init__(self, config: BatchConfig, logger: logging.Logger = None):
        self.config = config
        self.logger = logger or logging.getLogger("CrossBearingEvaluator")
        
        # 获取配置
        self.cross_bearing_mode = config.get('cross_bearing_mode', 'leave_one_out')
        self.cross_bearing_train_ratio = config.get('cross_bearing_train_ratio', 0.8)
        self.cross_bearing_visualize = config.get('cross_bearing_visualize', True)
        
        # 创建输出目录
        self.output_root = Path(config['output_root']) / "cross_bearing_evaluation"
        self.output_root.mkdir(parents=True, exist_ok=True)
        
        # 初始化模型运行器
        self.model_runner = ModelRunner(config)
        
        # 初始化可视化工具
        self.rul_evaluator = RULEvaluator(config)
        
        self.logger.info(f"跨轴承评估器初始化: 模式={self.cross_bearing_mode}")
    
    def group_bearings_by_condition(self, bearing_folders: List[str]) -> Dict[str, List[Dict]]:
        """
        按工况分组轴承
        
        参数:
            bearing_folders: 轴承文件夹路径列表
        
        返回:
            工况分组字典，格式: {condition: [{'name': folder, 'bearing_num': num}]}
        """
        conditions = {}
        
        for folder in bearing_folders:
            # 提取文件夹名（处理相对路径）
            folder_name = folder.replace(os.sep, '_').replace('/', '_')
            
            # 解析工况和轴承编号
            # XJTU-SY格式: "35Hz12kN_Bearing1_1" 或 "35Hz12kN/Bearing1_1"
            parts = folder_name.split('_')
            
            if len(parts) >= 2:
                # 工况通常是前2-3部分，如 "35Hz12kN"
                condition = '_'.join(parts[:-1])
                bearing_num = parts[-1] if len(parts) > 1 else folder_name
            else:
                condition = 'unknown'
                bearing_num = folder_name
            
            if condition not in conditions:
                conditions[condition] = []
            
            conditions[condition].append({
                'name': folder,
                'bearing_num': bearing_num,
                'folder_name': folder_name
            })
        
        # 按轴承编号排序
        for condition in conditions:
            conditions[condition].sort(key=lambda x: x['bearing_num'])
        
        return conditions
    
    def merge_bearings_data(self, bearing_list: List[Dict], fit_scaler: bool = True) -> Tuple:
        """
        合并多个轴承的数据
        
        参数:
            bearing_list: 轴承信息列表
            fit_scaler: 是否拟合归一化器（训练集为True，测试集为False）
        
        返回:
            (signals_list, cwt_images_list, rul_labels_list, hi_labels_list, processor)
        """
        all_signals = []
        all_rul_labels = []
        all_hi_labels = []
        
        # 创建数据处理器
        processor = MultiModalDataProcessor(
            window_size=self.config['window_size'],
            overlap_ratio=self.config['overlap_ratio'],
            sampling_rate=self.config['sampling_rate'],
            cwt_image_shape=(1, 128, 128),
            config=self.config
        )
        
        data_loader = XJTUDataLoader(vibration_column=self.config['vibration_column'])
        
        # 记录所有轴承的原始RUL最大值
        bearing_max_rul = []
        
        for bearing_info in bearing_list:
            bearing_path = os.path.join(self.config['data_root'], bearing_info['name'])
            
            self.logger.info(f"  加载轴承: {bearing_info['name']}")
            
            signal, rul, hi = data_loader.load_bearing_data(bearing_path)
            
            if signal is None or rul is None or hi is None:
                self.logger.warning(f"  警告: 轴承 {bearing_info['name']} 数据加载失败，跳过")
                continue
            
            all_signals.append(signal)
            all_rul_labels.append(rul)
            all_hi_labels.append(hi)
            bearing_max_rul.append(np.max(rul))
        
        if not all_signals:
            self.logger.error("没有成功加载任何轴承数据")
            return None, None, None, None, None
        
        # 合并信号
        combined_signal = np.concatenate(all_signals)
        combined_rul = np.concatenate(all_rul_labels)
        combined_hi = np.concatenate(all_hi_labels)
        
        # 记录合并后的RUL最大值（用于归一化）
        combined_max_rul = np.max(combined_rul)
        
        self.logger.info(f"  合并后信号长度: {len(combined_signal):,}")
        self.logger.info(f"  合并后RUL范围: [{combined_rul.min():.1f}, {combined_rul.max():.1f}]")
        self.logger.info(f"  各轴承RUL最大值: {bearing_max_rul}")
        
        # 特征提取
        signals_list, cwt_images_list, rul_labels_list, hi_labels_list = processor.create_dataset(
            full_signal=combined_signal,
            rul_array=combined_rul,
            hi_array=combined_hi,
            bearing_output_dir=None,
            bearing_name="merged"
        )
        
        if fit_scaler:
            # 训练集：拟合RUL归一化器
            # 保存训练集的实际最大值
            processor.current_bearing_max = combined_max_rul
            rul_labels_scaled = processor.preprocess_labels(rul_labels_list, fit=True)
            self.logger.info(f"  训练集RUL归一化: 使用最大值 {combined_max_rul:.1f}")
        else:
            # 测试集：使用训练集拟合的归一化器
            rul_labels_scaled = processor.preprocess_labels(rul_labels_list, fit=False)
        
        return signals_list, cwt_images_list, rul_labels_scaled, hi_labels_list, processor
    
    def train_model_on_bearings(self, train_bearings: List[Dict], 
                                test_bearing: Dict) -> Tuple[Dict, Dict]:
        """
        在训练轴承上训练模型，在测试轴承上评估
        
        参数:
            train_bearings: 训练轴承列表
            test_bearing: 测试轴承信息
        
        返回:
            (metrics_dict, predictions_dict)
        """
        self.logger.info(f"  训练轴承数: {len(train_bearings)}")
        self.logger.info(f"  测试轴承: {test_bearing['name']}")
        
        # 加载并合并训练数据
        train_signals, train_cwt, train_rul, train_hi, processor = self.merge_bearings_data(
            train_bearings, fit_scaler=True
        )
        
        if train_signals is None:
            return None, None
        
        # 加载测试数据
        test_processor = MultiModalDataProcessor(
            window_size=self.config['window_size'],
            overlap_ratio=self.config['overlap_ratio'],
            sampling_rate=self.config['sampling_rate'],
            cwt_image_shape=(1, 128, 128),
            config=self.config
        )
        
        # 【关键修复】测试集必须使用与训练集相同的归一化参数
        # 复制训练集拟合的归一化参数
        test_processor.global_rul_max = processor.global_rul_max
        test_processor.normalization_mode = processor.normalization_mode
        
        # 【关键修复】测试集使用训练集的最大值进行归一化，而不是自身最大值
        test_processor.current_bearing_max = processor.current_bearing_max
        
        data_loader = XJTUDataLoader(vibration_column=self.config['vibration_column'])
        test_bearing_path = os.path.join(self.config['data_root'], test_bearing['name'])
        test_signal, test_rul, test_hi = data_loader.load_bearing_data(test_bearing_path)
        
        if test_signal is None:
            self.logger.error(f"测试轴承数据加载失败: {test_bearing['name']}")
            return None, None
        
        # 特征提取
        test_signals, test_cwt, test_rul_raw, test_hi = test_processor.create_dataset(
            full_signal=test_signal,
            rul_array=test_rul,
            hi_array=test_hi,
            bearing_output_dir=None,
            bearing_name=test_bearing['name']
        )
        
        # 【关键修复】测试集RUL归一化 - 使用训练集的最大值（fit=False）
        test_rul_scaled = test_processor.preprocess_labels(test_rul_raw, fit=False)
        
        # 保存测试集的实际最大值（仅用于日志和可视化）
        test_max_rul = float(np.max(test_rul_raw)) if len(test_rul_raw) > 0 else 1.0
        
        self.logger.info(f"  训练集RUL最大值: {processor.current_bearing_max:.1f}")
        self.logger.info(f"  测试集RUL最大值: {test_max_rul:.1f}")
        self.logger.info(f"  归一化: 测试集使用训练集最大值 {processor.current_bearing_max:.1f}")
        
        # 划分训练集和验证集
        total_samples = len(train_signals)
        val_samples = int(total_samples * self.config['val_split'])
        train_samples = total_samples - val_samples
        
        indices = np.random.permutation(total_samples)
        train_indices = indices[:train_samples]
        val_indices = indices[train_samples:]
        
        X_train_sig = np.array([train_signals[i] for i in train_indices])
        X_train_cwt = np.array([train_cwt[i] for i in train_indices])
        y_rul_train = np.array([train_rul[i] for i in train_indices])
        y_hi_train = np.array([train_hi[i] for i in train_indices])
        
        X_val_sig = np.array([train_signals[i] for i in val_indices])
        X_val_cwt = np.array([train_cwt[i] for i in val_indices])
        y_rul_val = np.array([train_rul[i] for i in val_indices])
        y_hi_val = np.array([train_hi[i] for i in val_indices])
        
        X_test_sig = np.array(test_signals)
        X_test_cwt = np.array(test_cwt)
        y_rul_test = np.array(test_rul_scaled)
        y_hi_test = np.array(test_hi)
        
        self.logger.info(f"  训练集大小: {len(X_train_sig)}, 验证集大小: {len(X_val_sig)}, 测试集大小: {len(X_test_sig)}")
        
        # 创建多模态模型
        input_signal_length = len(X_train_sig[0]) if len(X_train_sig) > 0 else 1024
        cwt_image_shape = (1, 128, 128)
        
        model = MultiModalRULPredictor(
            cwt_image_shape=cwt_image_shape,
            signal_length=input_signal_length,
            cnn_channels=[16, 32, 64],
            lstm_hidden_size=64,
            lstm_num_layers=2,
            fusion_method='late',
            dropout_rate=self.config['dropout']
        )
        
        # 创建数据集
        train_dataset = MultiModalDataset(X_train_sig, X_train_cwt, y_rul_train, y_hi_train)
        val_dataset = MultiModalDataset(X_val_sig, X_val_cwt, y_rul_val, y_hi_val)
        test_dataset = MultiModalDataset(X_test_sig, X_test_cwt, y_rul_test, y_hi_test)
        
        # 创建数据加载器
        batch_size = min(self.config['batch_size'], len(X_train_sig))
        pin_memory = self.config.get('pin_memory', True) and torch.cuda.is_available()
        num_workers = self.config.get('num_workers', 0)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
        
        # 训练模型
        model_save_path = self.output_root / f"model_{test_bearing['name'].replace(os.sep, '_')}.pth"
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        criterion = nn.MSELoss()
        
        trainer = MultiModalTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=None,
            device=DEVICE,
            config=self.config
        )
        
        history, best_val_score = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=self.config['epochs'],
            patience=self.config['patience'],
            checkpoint_path=str(model_save_path)
        )
        
        # 加载最佳模型
        if model_save_path.exists():
            checkpoint = torch.load(model_save_path, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.move_to_device(DEVICE)
        
        # 【关键修复】评估模型时，传入 test_processor（它使用训练集的最大值进行反归一化）
        eval_metrics, rul_preds, rul_labels = self.model_runner.evaluate_model(
            model, test_loader, test_processor, DEVICE
        )
        
        # 打印评估结果
        self.logger.info(f"  测试集 R²: {eval_metrics.get('r2', 0):.4f}")
        self.logger.info(f"  测试集 RMSE: {eval_metrics.get('rmse', 0):.4f}")
        self.logger.info(f"  测试集 MAE: {eval_metrics.get('mae', 0):.4f}")
        
        return eval_metrics, {
            'predictions': rul_preds,
            'true_values': rul_labels,
            'history': history,
            'best_val_score': best_val_score,
            'train_max_rul': processor.current_bearing_max,
            'test_max_rul': test_max_rul
        }
    
    def evaluate_cross_bearing(self, bearing_folders: List[str]) -> List[Dict]:
        """
        执行跨轴承评估
        
        参数:
            bearing_folders: 轴承文件夹路径列表
        
        返回:
            评估结果列表
        """
        self.logger.info("=" * 80)
        self.logger.info("开始跨轴承评估")
        self.logger.info(f"模式: {self.cross_bearing_mode}")
        self.logger.info("=" * 80)
        
        # 按工况分组
        conditions = self.group_bearings_by_condition(bearing_folders)
        
        self.logger.info(f"找到 {len(conditions)} 个工况:")
        for condition, bearings in conditions.items():
            bearing_names = [b['name'] for b in bearings]
            self.logger.info(f"  {condition}: {len(bearings)}个轴承 - {bearing_names}")
        
        all_results = []
        
        for condition, bearings in conditions.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"工况: {condition}")
            self.logger.info(f"轴承列表: {[b['name'] for b in bearings]}")
            self.logger.info(f"{'='*60}")
            
            if self.cross_bearing_mode == 'leave_one_out':
                # 留一法：每个轴承作为测试集
                for test_idx, test_bearing in enumerate(bearings):
                    train_bearings = [b for i, b in enumerate(bearings) if i != test_idx]
                    
                    self.logger.info(f"\n测试轴承: {test_bearing['name']}")
                    self.logger.info(f"训练轴承: {[b['name'] for b in train_bearings]}")
                    
                    # 训练和评估
                    metrics, predictions = self.train_model_on_bearings(train_bearings, test_bearing)
                    
                    if metrics:
                        result = {
                            'condition': condition,
                            'test_bearing': test_bearing['name'],
                            'train_bearings': [b['name'] for b in train_bearings],
                            'metrics': metrics,
                            'predictions': predictions
                        }
                        all_results.append(result)
                        self.logger.info(f"  测试集 R²: {metrics.get('r2', 0):.4f}")
                    else:
                        self.logger.warning(f"  评估失败，跳过")
            
            elif self.cross_bearing_mode == 'train_test_split':
                # 训练/测试划分模式
                n_bearings = len(bearings)
                n_train = int(n_bearings * self.cross_bearing_train_ratio)
                
                # 随机划分
                indices = np.random.permutation(n_bearings)
                train_indices = indices[:n_train]
                test_indices = indices[n_train:]
                
                train_bearings = [bearings[i] for i in train_indices]
                test_bearings = [bearings[i] for i in test_indices]
                
                self.logger.info(f"训练轴承: {[b['name'] for b in train_bearings]}")
                self.logger.info(f"测试轴承: {[b['name'] for b in test_bearings]}")
                
                # 对每个测试轴承进行评估
                for test_bearing in test_bearings:
                    metrics, predictions = self.train_model_on_bearings(train_bearings, test_bearing)
                    
                    if metrics:
                        result = {
                            'condition': condition,
                            'test_bearing': test_bearing['name'],
                            'train_bearings': [b['name'] for b in train_bearings],
                            'metrics': metrics,
                            'predictions': predictions
                        }
                        all_results.append(result)
                        self.logger.info(f"  测试轴承 {test_bearing['name']} R²: {metrics.get('r2', 0):.4f}")
        
        # 汇总结果
        self._summarize_cross_bearing_results(all_results)
        
        # 可视化
        if self.cross_bearing_visualize:
            self._visualize_cross_bearing_results(all_results)
        
        return all_results
    
    def _summarize_cross_bearing_results(self, results: List[Dict]):
        """汇总跨轴承评估结果"""
        if not results:
            self.logger.warning("没有评估结果可汇总")
            return
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("跨轴承评估汇总结果")
        self.logger.info("=" * 80)
        
        # 创建DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'condition': result['condition'],
                'test_bearing': result['test_bearing'],
                'train_bearings_count': len(result['train_bearings']),
                'train_bearings': ', '.join(result['train_bearings']),
                'r2': result['metrics'].get('r2', 0),
                'rmse': result['metrics'].get('rmse', 0),
                'mae': result['metrics'].get('mae', 0),
                'mse': result['metrics'].get('mse', 0)
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # 保存汇总文件
        summary_csv_path = self.output_root / "cross_bearing_summary.csv"
        df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
        
        summary_excel_path = self.output_root / "cross_bearing_summary.xlsx"
        df_summary.to_excel(summary_excel_path, index=False)
        
        self.logger.info(f"汇总文件已保存:")
        self.logger.info(f"  CSV: {summary_csv_path}")
        self.logger.info(f"  Excel: {summary_excel_path}")
        
        # 打印统计
        self.logger.info(f"\n统计结果:")
        self.logger.info(f"  评估次数: {len(results)}")
        self.logger.info(f"  平均 R²: {df_summary['r2'].mean():.4f} ± {df_summary['r2'].std():.4f}")
        self.logger.info(f"  平均 RMSE: {df_summary['rmse'].mean():.4f} ± {df_summary['rmse'].std():.4f}")
        self.logger.info(f"  平均 MAE: {df_summary['mae'].mean():.4f} ± {df_summary['mae'].std():.4f}")
        
        # 按工况分组统计
        self.logger.info(f"\n按工况分组统计:")
        for condition in df_summary['condition'].unique():
            cond_df = df_summary[df_summary['condition'] == condition]
            self.logger.info(f"  {condition}: R²={cond_df['r2'].mean():.4f} ± {cond_df['r2'].std():.4f} (n={len(cond_df)})")
        
        # 打印详细结果
        self.logger.info(f"\n详细结果:")
        for idx, row in df_summary.iterrows():
            self.logger.info(f"  {idx+1}. {row['test_bearing']}: R²={row['r2']:.4f}, RMSE={row['rmse']:.4f}, MAE={row['mae']:.4f}")
    
    def _visualize_cross_bearing_results(self, results: List[Dict]):
        """可视化跨轴承评估结果"""
        try:
            viz_dir = self.output_root / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            if not results:
                return
            
            # 提取指标
            conditions = []
            r2_values = []
            rmse_values = []
            mae_values = []
            bearing_names = []
            
            for result in results:
                conditions.append(result['condition'])
                bearing_names.append(result['test_bearing'])
                r2_values.append(result['metrics'].get('r2', 0))
                rmse_values.append(result['metrics'].get('rmse', 0))
                mae_values.append(result['metrics'].get('mae', 0))
            
            # 创建柱状图
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # R²柱状图
            ax1 = axes[0]
            colors = plt.cm.Set3(range(len(r2_values)))
            bars1 = ax1.bar(range(len(r2_values)), r2_values, color=colors, edgecolor='black')
            ax1.set_xlabel('测试轴承', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_ylabel('R²', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_title('跨轴承评估 - R²分数', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax1.set_xticks(range(len(r2_values)))
            ax1.set_xticklabels(bearing_names, rotation=45, ha='right', fontsize=8)
            ax1.axhline(y=np.mean(r2_values), color='r', linestyle='--', label=f'平均={np.mean(r2_values):.3f}')
            ax1.legend(prop=fm.FontProperties(family=selected_font, size=9))
            ax1.set_ylim(0, 1.05)
            
            for bar, val in zip(bars1, r2_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            # RMSE柱状图
            ax2 = axes[1]
            bars2 = ax2.bar(range(len(rmse_values)), rmse_values, color=colors, edgecolor='black')
            ax2.set_xlabel('测试轴承', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_ylabel('RMSE', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_title('跨轴承评估 - RMSE', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax2.set_xticks(range(len(rmse_values)))
            ax2.set_xticklabels(bearing_names, rotation=45, ha='right', fontsize=8)
            ax2.axhline(y=np.mean(rmse_values), color='r', linestyle='--', label=f'平均={np.mean(rmse_values):.2f}')
            ax2.legend(prop=fm.FontProperties(family=selected_font, size=9))
            
            for bar, val in zip(bars2, rmse_values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            
            # MAE柱状图
            ax3 = axes[2]
            bars3 = ax3.bar(range(len(mae_values)), mae_values, color=colors, edgecolor='black')
            ax3.set_xlabel('测试轴承', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_ylabel('MAE', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_title('跨轴承评估 - MAE', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax3.set_xticks(range(len(mae_values)))
            ax3.set_xticklabels(bearing_names, rotation=45, ha='right', fontsize=8)
            ax3.axhline(y=np.mean(mae_values), color='r', linestyle='--', label=f'平均={np.mean(mae_values):.2f}')
            ax3.legend(prop=fm.FontProperties(family=selected_font, size=9))
            
            for bar, val in zip(bars3, mae_values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            
            plt.suptitle('跨轴承评估结果汇总', fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            plt.tight_layout()
            
            save_path = viz_dir / "cross_bearing_results.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            self.logger.info(f"跨轴承评估可视化已保存: {save_path}")
            
        except Exception as e:
            self.logger.error(f"可视化跨轴承结果失败: {e}")
            traceback.print_exc()

def create_quick_test_config():
    """创建快速测试配置文件"""
    template = """# 快速测试配置文件
# 版本: 快速测试用

data_root: "E:/毕业设计/data"
output_root: "./quick_test_results"
pattern: ".*Bearing.*_.*"
vibration_column: "horizontal"
window_size: 1024
overlap_ratio: 0.75
sampling_rate: 25600
use_only_35khz: true

# 简化训练配置
batch_size: 16
epochs: 30
patience: 10
learning_rate: 0.001
dropout: 0.3
weight_decay: 0.0001

# GPU优化配置
use_gpu: true
pin_memory: true
num_workers: 0

# 数据划分配置
test_size: 0.2
val_split: 0.3
random_seed: 42

# HI配置
rul_loss_weight: 1.0
hi_loss_weight: 1.0
use_sample_weighting: true
weighting_alpha: 2.0
enable_hi_visualization: true

# 简化模型架构
cnn_architecture: "simple"
signal_processor: "lstm"
prediction_head_dims: [64, 32, 16]

# 损失函数配置
loss_function: "mse"

# 学习率调度器
lr_scheduler: "plateau"
lr_scheduler_params:
  mode: "min"
  factor: 0.5
  patience: 5
  min_lr: 1e-6

# 实验跟踪
enable_experiment_tracking: false

# 评估指标
tolerance_threshold: 0.1

# 模型对比
compare_models: true
models_to_compare: ["mlp", "multimodal"]

# 可视化配置
save_cwt_images: true
cwt_visualization_points: "default"
cwt_image_dpi: 100
visualization_dpi: 150
save_per_bearing_comparisons: true
save_cross_bearing_summaries: false

# 输出配置
save_models: true
save_scalers: false
save_plots: true
skip_existing: true
"""
    
    os.makedirs("config_templates", exist_ok=True)
    template_path = "config_templates/quick_test_config.yaml"
    
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(template)
    
    print(f"快速测试配置文件已创建: {template_path}")
    return template_path
