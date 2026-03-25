"""
评估模块 - 提供XJTU-SY数据集标准评估指标
只保留 R²、MSE、RMSE、MAE
"""

import numpy as np
from typing import Dict


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   tolerance: float = 0.1) -> Dict[str, float]:
    """
    计算XJTU-SY数据集标准评估指标
    只返回 R²、MSE、RMSE、MAE
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        tolerance: 容错阈值（保留参数以兼容，但不使用）
    
    返回:
        包含4个核心指标的字典
    """
    import sklearn.metrics as metrics
    
    # 处理空数组情况
    if len(y_true) == 0 or len(y_pred) == 0:
        print("警告: 输入数组为空，返回默认指标")
        return {
            'r2': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'mae': 0.0
        }
    
    # 计算MSE
    mse = metrics.mean_squared_error(y_true, y_pred)
    
    # 计算RMSE
    rmse = np.sqrt(mse)
    
    # 计算MAE
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    # 计算R²
    try:
        r2 = metrics.r2_score(y_true, y_pred)
        if np.isnan(r2):
            r2 = 0.0
    except Exception:
        r2 = 0.0
    
    metrics_dict = {
        'r2': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae)
    }
    
    return metrics_dict


def print_metrics_summary(metrics_dict: Dict[str, float], prefix: str = ""):
    """
    打印指标汇总
    
    参数:
        metrics_dict: 指标字典
        prefix: 前缀字符串
    """
    print(f"\n{prefix}评估结果:")
    print(f"  R²: {metrics_dict.get('r2', 0):.4f}")
    print(f"  MSE: {metrics_dict.get('mse', 0):.6f}")
    print(f"  RMSE: {metrics_dict.get('rmse', 0):.4f}")
    print(f"  MAE: {metrics_dict.get('mae', 0):.4f}")
