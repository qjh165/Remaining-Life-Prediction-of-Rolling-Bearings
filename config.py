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
            'pattern': r'.*Bearing.*_.*',  # 匹配轴承文件夹的正则表达式
            'vibration_column': 'horizontal',
            'window_size': 1024,
            'overlap_ratio': 0.75,
            'sampling_rate': 25600,
            'batch_size': 32,
            'epochs': 100,
            'patience': 15,
            'learning_rate': 0.001,
            'hidden_sizes': [64, 32, 16],
            'dropout': 0.3,
            'test_size': 0.3,
            'val_split': 0.5,
            'random_seed': 42,
            'save_models': True,
            'save_scalers': True,
            'save_plots': True,
            'skip_existing': False,  # 跳过已处理的结果
            
            # GPU优化配置
            'use_gpu': True,  # 是否使用GPU
            'pin_memory': True,  # 是否使用锁页内存
            'num_workers': 4 if os.name != 'nt' else 0,  # 数据加载工作进程数
            
            # 新增配置：模型对比和鲁棒性验证
            'compare_models': True,  # 是否进行模型对比
            'models_to_compare': ['mlp', 'linear'],  # 要对比的模型列表
            'robustness_test': True,  # 是否进行鲁棒性测试
            'noise_levels': [0.0, 0.05],  # 噪声水平（相对于特征标准差）
            'linear_model_type': 'ridge',  # 线性模型类型：linear, ridge, lasso
            'ridge_alpha': 1.0,  # Ridge回归的alpha参数
            'lasso_alpha': 0.1,  # Lasso回归的alpha参数
            
            # 新增：CWT可视化配置
            'save_cwt_images': True,  # 是否保存CWT时频图
            'cwt_visualization_points': 'default',  # 可视化点：'default'或具体索引列表
            'cwt_image_dpi': 150,  # 图像DPI
            'cwt_visualization_dir': 'cwt_visualizations',  # 可视化保存目录
            
            # 新增：增强可视化配置
            'save_detailed_visualizations': True,  # 是否保存详细可视化图表
            'save_per_bearing_comparisons': True,  # 是否保存单轴承对比图
            'save_cross_bearing_summaries': True,  # 是否保存跨轴承汇总图
            'visualization_style': 'seaborn-v0_8',  # 可视化样式
            'visualization_dpi': 300,  # 可视化DPI
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