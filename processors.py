"""
数据处理模块 - 负责数据预处理、数据集创建和特征处理流程
包含RULDataset和MultiModalDataset类，支持健康因子(HI)标签
"""

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os
from typing import Tuple, List, Optional, Dict, Any
import traceback

from feature_extractors import TimeFrequencyFeatureExtractor, CWTFeatureExtractor


class RULDataset(Dataset):
    """RUL数据集类 - 用于单模态MLP模型，返回特征、RUL标签和HI标签"""
    
    def __init__(self, features, rul_labels, hi_labels):
        """
        参数:
            features: 特征数组，形状为 [n_samples, n_features]
            rul_labels: RUL标签数组，形状为 [n_samples]
            hi_labels: HI标签数组，形状为 [n_samples]
        """
        self.features = torch.FloatTensor(features)
        self.rul_labels = torch.FloatTensor(rul_labels).view(-1, 1)
        self.hi_labels = torch.FloatTensor(hi_labels).view(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], (self.rul_labels[idx], self.hi_labels[idx])


class MultiModalDataset(Dataset):
    """多模态RUL数据集 - 支持健康因子(HI)标签"""
    
    def __init__(self, signals, cwt_images, rul_labels, hi_labels):
        """
        初始化多模态数据集
        
        参数:
            signals: 原始振动信号数组，形状为 [n_samples, signal_length]
            cwt_images: CWT图像数组，形状为 [n_samples, 1, height, width]
            rul_labels: RUL标签数组，形状为 [n_samples]
            hi_labels: HI标签数组，形状为 [n_samples]
        """
        assert len(signals) == len(cwt_images) == len(rul_labels) == len(hi_labels), \
            f"Lengths must match: signals ({len(signals)}), cwt_images ({len(cwt_images)}), rul_labels ({len(rul_labels)}), hi_labels ({len(hi_labels)})"
        
        self.signals = signals.astype(np.float32)
        self.cwt_images = cwt_images.astype(np.float32)
        self.rul_labels = rul_labels.astype(np.float32)
        self.hi_labels = hi_labels.astype(np.float32)
        
    def __len__(self):
        return len(self.rul_labels)
    
    def __getitem__(self, idx):
        # 返回 (cwt_images, signals) 作为输入，(rul_labels, hi_labels) 作为标签
        return (
            (torch.FloatTensor(self.cwt_images[idx]), 
             torch.FloatTensor(self.signals[idx]).unsqueeze(0)),
            (torch.FloatTensor([self.rul_labels[idx]]),
             torch.FloatTensor([self.hi_labels[idx]]))
        )
    
    def get_sample_weights(self, alpha=2.0):
        """
        根据HI值计算样本权重（退化越严重权重越大）
        参考孙伊萍论文：对退化后期样本赋予更高权重
        
        参数:
            alpha: 权重指数，越大后期权重越高
        """
        # 权重 = (1 - HI) ^ alpha
        weights = (1 - self.hi_labels) ** alpha
        # 归一化
        weights = weights / np.sum(weights)
        return weights


class RULDataProcessor:
    """RUL数据处理流程"""
    
    def __init__(self, window_size=1024, overlap_ratio=0.75, sampling_rate=25600):
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.feature_extractor = TimeFrequencyFeatureExtractor(sampling_rate)
        self.feature_scaler = StandardScaler()
        self.rul_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # 特征维度
        self.feature_dim = self.feature_extractor.get_feature_dimension()
        print(f"特征提取器维度: {self.feature_dim}")
    
    def create_dataset(self, full_signal, rul_array, hi_array):
        """从信号、RUL标签和HI标签创建数据集"""
        step_size = int(self.window_size * (1 - self.overlap_ratio))
        if step_size <= 0:
            step_size = self.window_size // 4
        
        # XJTU-SY数据：每个文件1.28秒，采样率25600Hz
        points_per_file = int(self.sampling_rate * 1.28)
        
        features_list = []
        rul_labels = []
        hi_labels = []
        
        start_idx = 0
        window_count = 0
        
        print(f"开始特征提取: 信号长度={len(full_signal)}, 窗口大小={self.window_size}")
        
        while start_idx + self.window_size <= len(full_signal):
            end_idx = start_idx + self.window_size
            segment = full_signal[start_idx:end_idx]
            
            # 估算当前窗口属于哪个文件
            file_idx = start_idx / points_per_file
            if file_idx >= len(rul_array):
                file_idx = len(rul_array) - 1
            
            rul_value = rul_array[int(file_idx)]
            hi_value = hi_array[int(file_idx)]
            
            try:
                # 提取特征
                features = self.feature_extractor.extract_features(segment)
                features_list.append(features)
                rul_labels.append(rul_value)
                hi_labels.append(hi_value)
                window_count += 1
                
                if window_count % 100 == 0:
                    print(f"已处理 {window_count} 个窗口...")
                    
            except Exception as e:
                print(f"处理窗口 {window_count} 时出错: {e}")
                # 使用默认特征
                default_features = self.feature_extractor._get_default_features()
                features_list.append(default_features)
                rul_labels.append(rul_value)
                hi_labels.append(hi_value)
            
            start_idx += step_size
        
        print(f"特征提取完成，共提取 {window_count} 个样本")
        
        if len(features_list) == 0:
            return np.array([]), np.array([]), np.array([])
        
        return np.array(features_list), np.array(rul_labels), np.array(hi_labels)
    
    def preprocess_features(self, X, fit=True):
        """预处理特征"""
        if len(X) == 0:
            return np.array([])
        
        if fit:
            X_processed = self.feature_scaler.fit_transform(X)
        else:
            X_processed = self.feature_scaler.transform(X)
        return X_processed
    
    def preprocess_labels(self, y, fit=True):
        """预处理标签（仅用于RUL，HI不需要归一化）"""
        if len(y) == 0:
            return np.array([])
        
        y = y.reshape(-1, 1)
        if fit:
            y_processed = self.rul_scaler.fit_transform(y)
        else:
            y_processed = self.rul_scaler.transform(y)
        return y_processed.flatten()
    
    def inverse_transform_labels(self, y_normalized):
        """反归一化标签"""
        if len(y_normalized) == 0:
            return np.array([])
        
        y_normalized = y_normalized.reshape(-1, 1)
        y_original = self.rul_scaler.inverse_transform(y_normalized)
        return y_original.flatten()
    
    def save_scalers(self, path='scalers'):
        """保存标准化器"""
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.feature_scaler, os.path.join(path, 'feature_scaler.joblib'))
        joblib.dump(self.rul_scaler, os.path.join(path, 'rul_scaler.joblib'))
    
    def load_scalers(self, path='scalers'):
        """加载标准化器"""
        self.feature_scaler = joblib.load(os.path.join(path, 'feature_scaler.joblib'))
        self.rul_scaler = joblib.load(os.path.join(path, 'rul_scaler.joblib'))
    
    def get_feature_dimension(self):
        """获取特征维度"""
        return self.feature_dim


class MultiModalDataProcessor:
    """多模态数据处理流程"""
    
    def __init__(self, 
                 window_size=1024, 
                 overlap_ratio=0.75,
                 sampling_rate=25600,
                 cwt_image_shape=(1, 64, 64),  # 包含通道维度
                 config=None):
        """
        初始化多模态数据处理器
        
        参数:
            window_size: 窗口大小
            overlap_ratio: 重叠率
            sampling_rate: 采样率
            cwt_image_shape: CWT图像形状，包含通道维度 (channels, height, width)
            config: 配置对象
        """
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.sampling_rate = sampling_rate
        self.cwt_image_shape = cwt_image_shape
        self.config = config or {}
        
        # 创建CWT特征提取器
        self.cwt_extractor = CWTFeatureExtractor(sampling_rate=sampling_rate)
        
        # 归一化器（仅用于RUL）
        self.rul_scaler = MinMaxScaler(feature_range=(0, 1))
        # HI不需要归一化，因为已经是在[0,1]范围内
    
    def process_signal_chunk(self, full_signal, rul_array, hi_array, chunk_size=1000000):
        """流式处理信号，避免内存溢出"""
        step_size = int(self.window_size * (1 - self.overlap_ratio))
        if step_size <= 0:
            step_size = self.window_size // 4
        
        points_per_file = int(self.sampling_rate * 1.28)
        total_length = len(full_signal)
        num_chunks = max(1, total_length // chunk_size)
        
        for chunk_idx in range(num_chunks):
            start_signal_idx = chunk_idx * chunk_size
            end_signal_idx = min((chunk_idx + 1) * chunk_size, total_length)
            
            signal_chunk = full_signal[start_signal_idx:end_signal_idx]
            
            signals_list = []
            cwt_images_list = []
            rul_labels_list = []
            hi_labels_list = []
            
            start_idx = 0
            while start_idx + self.window_size <= len(signal_chunk):
                end_idx = start_idx + self.window_size
                segment = signal_chunk[start_idx:end_idx]
                
                global_idx = start_signal_idx + start_idx
                file_idx = global_idx / points_per_file
                if file_idx >= len(rul_array):
                    file_idx = len(rul_array) - 1
                
                rul_value = rul_array[int(file_idx)]
                hi_value = hi_array[int(file_idx)]
                
                try:
                    cwt_image = self.cwt_extractor.cwt_to_image(
                        segment, 
                        target_shape=self.cwt_image_shape[1:]
                    )
                    
                    signal_mean = np.mean(segment)
                    signal_std = np.std(segment)
                    if signal_std > 1e-10:
                        segment_normalized = (segment - signal_mean) / signal_std
                    else:
                        segment_normalized = segment
                    
                    signals_list.append(segment_normalized.astype(np.float32))
                    cwt_images_list.append(cwt_image)
                    rul_labels_list.append(rul_value)
                    hi_labels_list.append(hi_value)
                    
                except Exception as e:
                    print(f"处理窗口时出错: {e}")
                    continue
                
                start_idx += step_size
            
            yield signals_list, cwt_images_list, rul_labels_list, hi_labels_list
    
    def create_dataset(self, full_signal, rul_array, hi_array, bearing_output_dir=None, bearing_name=None):
        """从信号、RUL标签和HI标签创建多模态数据集"""
        signals_list = []
        cwt_images_list = []
        rul_labels_list = []
        hi_labels_list = []
        
        print(f"开始多模态特征提取: 信号长度={len(full_signal)}, 窗口大小={self.window_size}")
        
        chunk_count = 0
        total_samples = 0
        
        for chunk_signals, chunk_cwt, chunk_rul, chunk_hi in self.process_signal_chunk(full_signal, rul_array, hi_array):
            signals_list.extend(chunk_signals)
            cwt_images_list.extend(chunk_cwt)
            rul_labels_list.extend(chunk_rul)
            hi_labels_list.extend(chunk_hi)
            
            chunk_count += 1
            total_samples += len(chunk_signals)
            
            if chunk_count % 5 == 0:
                print(f"已处理 {chunk_count} 个数据块，共 {total_samples} 个样本...")
        
        print(f"多模态特征提取完成，共提取 {total_samples} 个样本")
        print(f"HI标签范围: [{min(hi_labels_list):.3f}, {max(hi_labels_list):.3f}]")
        
        # 如果启用了CWT可视化，生成关键时间点的时频图
        if (self.config.get('save_cwt_images', True) and 
            bearing_output_dir is not None and 
            bearing_name is not None):
            self.cwt_extractor.generate_key_timepoint_cwt_visualizations(
                full_signal=full_signal,
                window_size=self.window_size,
                overlap_ratio=self.overlap_ratio,
                output_dir=bearing_output_dir,
                bearing_name=bearing_name,
                config=self.config
            )
        
        return signals_list, cwt_images_list, rul_labels_list, hi_labels_list
    
    def preprocess_labels(self, labels, fit=True):
        """预处理标签（仅用于RUL，HI不需要归一化）"""
        labels = np.array(labels).reshape(-1, 1)
        if fit:
            labels_scaled = self.rul_scaler.fit_transform(labels)
        else:
            labels_scaled = self.rul_scaler.transform(labels)
        return labels_scaled.flatten()
    
    def inverse_transform_labels(self, labels_normalized):
        """反归一化标签"""
        labels_normalized = np.array(labels_normalized).reshape(-1, 1)
        labels_original = self.rul_scaler.inverse_transform(labels_normalized)
        return labels_original.flatten()