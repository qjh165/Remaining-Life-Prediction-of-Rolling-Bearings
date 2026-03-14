"""
评估模块 - 提供全面的评估指标计算，包括PHM Score
"""

import numpy as np
from typing import Dict


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   tolerance: float = 0.1) -> Dict[str, float]:
    """
    计算全面的评估指标
    
    参数:
        y_true: 真实值数组
        y_pred: 预测值数组
        tolerance: 容错阈值（相对值，如0.1表示10%误差以内）
    
    返回:
        包含所有指标的字典
    """
    import sklearn.metrics as metrics
    
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y_true, y_pred)
    
    try:
        r2 = metrics.r2_score(y_true, y_pred)
    except Exception:
        r2 = 0.0
    
    mask = y_true != 0
    if np.any(mask):
        mape_values = np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])
        mape = np.mean(mape_values) * 100
    else:
        mape = 0.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_errors = np.abs((y_pred - y_true) / (y_true + 1e-10))
        rel_errors = np.where(np.isinf(rel_errors) | np.isnan(rel_errors), 0, rel_errors)
    
    accuracy_within_tolerance = np.mean(rel_errors <= tolerance) * 100
    
    errors = y_pred - y_true
    error_mean = np.mean(errors)
    error_std = np.std(errors)
    max_error = np.max(np.abs(errors))
    median_ae = np.median(np.abs(errors))
    
    metrics_dict = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'accuracy_within_tolerance': float(accuracy_within_tolerance),
        'error_mean': float(error_mean),
        'error_std': float(error_std),
        'max_error': float(max_error),
        'median_ae': float(median_ae)
    }
    
    return metrics_dict


def calculate_phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算PHM Score（张金超论文公式2.2）
    
    Score = sum(exp(-error/13) - 1) for error < 0
            sum(exp(error/10) - 1) for error >= 0
    
    参数:
        y_true: 真实RUL值
        y_pred: 预测RUL值
    
    返回:
        PHM Score（越小越好）
    """
    errors = y_pred - y_true
    
    # 分别计算提前预测和延迟预测的惩罚
    early_errors = errors[errors < 0]  # 预测值小于真实值（提前预测）
    late_errors = errors[errors >= 0]  # 预测值大于真实值（延迟预测）
    
    early_score = np.sum(np.exp(-early_errors / 13) - 1) if len(early_errors) > 0 else 0
    late_score = np.sum(np.exp(late_errors / 10) - 1) if len(late_errors) > 0 else 0
    
    total_score = early_score + late_score
    
    return float(total_score)


def print_metrics_summary(metrics_dict: Dict[str, float], prefix: str = ""):
    """
    打印指标汇总
    
    参数:
        metrics_dict: 指标字典
        prefix: 前缀字符串
    """
    print(f"\n{prefix}评估结果:")
    print(f"  R²: {metrics_dict.get('r2', 0):.4f}")
    print(f"  RMSE: {metrics_dict.get('rmse', 0):.4f}")
    print(f"  MAE: {metrics_dict.get('mae', 0):.4f}")
    print(f"  MAPE: {metrics_dict.get('mape', 0):.2f}%")
    print(f"  容错准确率: {metrics_dict.get('accuracy_within_tolerance', 0):.2f}%")
    print(f"  HI-R²: {metrics_dict.get('hi_r2', 0):.4f}")
    print(f"  HI-RMSE: {metrics_dict.get('hi_rmse', 0):.4f}")
    print(f"  PHM Score: {metrics_dict.get('phm_score', 0):.4f}")
    print(f"  误差均值: {metrics_dict.get('error_mean', 0):.4f}")
    print(f"  误差标准差: {metrics_dict.get('error_std', 0):.4f}")