"""
配置管理模块 - 负责所有与配置相关的功能
"""

import os
import yaml
from typing import Dict, Any


class BatchConfig:
    """批量处理配置类"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
    
    def _load_default_config(self) -> Dict:
        """加载默认配置"""
        return {
            'data_root': 'E:/毕业设计/data',
            'output_root': './batch_results',
            'pattern': r'.*Bearing.*_.*',
            'vibration_column': 'horizontal',
            'window_size': 4096,
            'overlap_ratio': 0.75,
            'sampling_rate': 25600,
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'learning_rate': 0.001,
            'hidden_sizes': [64, 32, 16],
            'dropout': 0.3,
            'test_size': 0.3,
            'val_split': 0.3,
            'random_seed': 42,
            'save_models': True,
            'save_scalers': True,
            'save_plots': True,
            'skip_existing': False,
            
            # 【新增】跨轴承评估配置
            'cross_bearing_eval': False,           # 是否启用跨轴承评估
            'cross_bearing_mode': 'leave_one_out',  # 留一法: 'leave_one_out', 'train_test_split'
            'cross_bearing_train_ratio': 0.8,       # 如果使用train_test_split模式，训练集比例
            'cross_bearing_visualize': True,        # 是否可视化跨轴承评估结果
            
            # 验证平滑配置
            'validation_smoothing': True,
            'validation_smoothing_alpha': 0.9,
            'validation_frequency': 2,
            
            # RUL归一化配置
            'rul_normalization_mode': 'global',
            'rul_global_max': 160,
            
            # GPU优化配置
            'use_gpu': True,
            'pin_memory': True,
            'num_workers': 4 if os.name != 'nt' else 0,
            
            # 模型对比和鲁棒性验证
            'compare_models': True,
            'models_to_compare': ['mlp', 'multimodal'],
            'robustness_test': True,
            'noise_levels': [0.0, 0.05],
            'linear_model_type': 'ridge',
            'ridge_alpha': 1.0,
            'lasso_alpha': 0.1,
            
            # CWT可视化配置
            'save_cwt_images': True,
            'cwt_visualization_points': 'default',
            'cwt_image_dpi': 150,
            'cwt_visualization_dir': 'cwt_visualizations',
            
            # 增强可视化配置
            'save_detailed_visualizations': True,
            'save_per_bearing_comparisons': True,
            'save_cross_bearing_summaries': True,
            'visualization_style': 'seaborn-v0_8',
            'visualization_dpi': 300,
        }
    
    def _load_config_file(self, config_path: str):
        """从YAML文件加载配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)
            
            if user_config:
                self.config.update(user_config)
                print(f"从 {config_path} 加载配置")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def get(self, key: str, default=None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        return self.config[key]
    
    def __setitem__(self, key: str, value: Any):
        self.config[key] = value
    
    def save(self, path: str):
        """保存配置到文件"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            print(f"配置已保存到 {path}")
        except Exception as e:
            print(f"保存配置失败: {e}")
