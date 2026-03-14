"""
评估与可视化模块 - 负责模型评估、指标计算和可视化
包含健康因子曲线绘制功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
from pathlib import Path
import os
import traceback
from typing import Dict, List, Optional, Tuple, Any

# 设置matplotlib以避免绘图溢出
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 20000

from utils import selected_font


class RULEvaluator:
    """RUL评估和可视化工具类"""
    
    def __init__(self, config):
        self.config = config
        self.visualization_style = config.get('visualization_style', 'seaborn-v0_8')
        self.visualization_dpi = config.get('visualization_dpi', 300)
        self.enable_hi_visualization = config.get('enable_hi_visualization', True)
        
        # 设置可视化样式
        try:
            plt.style.use(self.visualization_style)
        except:
            plt.style.use('default')
    
    def _downsample_data(self, data, max_points=5000):
        """对数据进行降采样，避免绘图溢出"""
        if len(data) <= max_points:
            return data, np.arange(len(data))
        
        # 均匀采样
        indices = np.linspace(0, len(data)-1, max_points, dtype=int)
        return data[indices], indices
    
    def create_rul_trend_comparison(self, models_predictions, true_rul, bearing_name, output_dir):
        """
        创建RUL预测趋势对比图 (Per-Bearing RUL Trend Comparison)
        """
        if not self.config.get('save_per_bearing_comparisons', True):
            return None
        
        try:
            # 对真实RUL降采样
            true_rul_sampled, indices = self._downsample_data(true_rul)
            sample_indices = np.arange(len(true_rul_sampled))
            
            # 也对预测值进行降采样
            models_predictions_sampled = {}
            for model_name, predictions in models_predictions.items():
                if len(predictions) > 0:
                    models_predictions_sampled[model_name] = predictions[indices]
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # 绘制真实RUL值
            ax.plot(sample_indices, true_rul_sampled, 'k-', linewidth=3, label='真实RUL', alpha=0.8)
            
            # 定义颜色和线条样式
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
            line_styles = ['-', '--', '-.', ':', '-']
            
            # 绘制各模型预测值
            for idx, (model_name, predictions) in enumerate(models_predictions_sampled.items()):
                if len(predictions) > 0:
                    color = colors[idx % len(colors)]
                    line_style = line_styles[idx % len(line_styles)]
                    ax.plot(sample_indices, predictions, 
                           color=color, linestyle=line_style, linewidth=2, 
                           label=f'{model_name}预测', alpha=0.7)
            
            # 设置图形属性
            ax.set_xlabel('样本索引 (时间顺序)', fontproperties=fm.FontProperties(family=selected_font, size=12))
            ax.set_ylabel('剩余使用寿命 (RUL)', fontproperties=fm.FontProperties(family=selected_font, size=12))
            ax.set_title(f'RUL预测趋势对比 - {bearing_name}', fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            ax.legend(prop=fm.FontProperties(family=selected_font, size=10), loc='best')
            ax.grid(True, alpha=0.3)
            
            # 添加RUL阶段标注
            max_rul = np.max(true_rul_sampled)
            if max_rul > 0:
                early_threshold = max_rul * 0.7
                mid_threshold = max_rul * 0.3
                
                ax.axhline(y=early_threshold, color='green', linestyle=':', alpha=0.5, label='早期RUL阈值')
                ax.axhline(y=mid_threshold, color='orange', linestyle=':', alpha=0.5, label='中期RUL阈值')
                ax.fill_between(sample_indices, early_threshold, max_rul, 
                               alpha=0.1, color='green', label='早期阶段')
                ax.fill_between(sample_indices, mid_threshold, early_threshold, 
                               alpha=0.1, color='orange', label='中期阶段')
                ax.fill_between(sample_indices, 0, mid_threshold, 
                               alpha=0.1, color='red', label='晚期阶段')
            
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'rul_trend_comparison_{bearing_name}.png')
            plt.savefig(save_path, dpi=self.visualization_dpi, bbox_inches='tight')
            plt.close(fig)
            
            return save_path
            
        except Exception as e:
            print(f"创建RUL趋势对比图失败: {e}")
            traceback.print_exc()
            return None
    
    def create_health_indicator_curves(self, true_his, pred_his, bearing_name, output_dir, 
                                      sample_indices=None, title_suffix=""):
        """
        绘制健康因子(HI)曲线对比图
        
        参数:
            true_his: 真实HI值数组
            pred_his: 预测HI值数组
            bearing_name: 轴承名称
            output_dir: 输出目录
            sample_indices: 样本索引（可选）
            title_suffix: 标题后缀
        """
        if not self.enable_hi_visualization:
            return None
        
        try:
            # 对HI数据降采样
            true_his_sampled, indices = self._downsample_data(true_his)
            pred_his_sampled = pred_his[indices]
            sample_indices = np.arange(len(true_his_sampled))
            
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            # 1. HI曲线对比图
            ax1 = axes[0]
            ax1.plot(sample_indices, true_his_sampled, 'b-', linewidth=2.5, label='真实HI', alpha=0.8)
            ax1.plot(sample_indices, pred_his_sampled, 'r--', linewidth=2, label='预测HI', alpha=0.7)
            
            # 添加退化阶段标注
            ax1.axhline(y=0.7, color='green', linestyle=':', alpha=0.5, label='早期阈值 (0.7)')
            ax1.axhline(y=0.3, color='orange', linestyle=':', alpha=0.5, label='中期阈值 (0.3)')
            ax1.fill_between(sample_indices, 0.7, 1.0, alpha=0.1, color='green', label='早期阶段')
            ax1.fill_between(sample_indices, 0.3, 0.7, alpha=0.1, color='orange', label='中期阶段')
            ax1.fill_between(sample_indices, 0, 0.3, alpha=0.1, color='red', label='晚期阶段')
            
            ax1.set_xlabel('样本索引 (时间顺序)', fontproperties=fm.FontProperties(family=selected_font, size=12))
            ax1.set_ylabel('健康因子 (HI)', fontproperties=fm.FontProperties(family=selected_font, size=12))
            ax1.set_title(f'健康因子退化曲线对比 - {bearing_name} {title_suffix}', 
                         fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            ax1.legend(prop=fm.FontProperties(family=selected_font, size=10), loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(-0.05, 1.05)
            
            # 2. HI预测误差图
            ax2 = axes[1]
            errors = pred_his_sampled - true_his_sampled
            ax2.bar(sample_indices, errors, width=1.0, color='skyblue', edgecolor='navy', alpha=0.7, linewidth=0.5)
            ax2.axhline(y=0, color='r', linestyle='-', linewidth=1.5)
            ax2.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='±5%误差线')
            ax2.axhline(y=-0.05, color='orange', linestyle='--', alpha=0.7)
            ax2.fill_between(sample_indices, -0.05, 0.05, alpha=0.1, color='green')
            
            ax2.set_xlabel('样本索引 (时间顺序)', fontproperties=fm.FontProperties(family=selected_font, size=12))
            ax2.set_ylabel('HI预测误差', fontproperties=fm.FontProperties(family=selected_font, size=12))
            ax2.set_title('健康因子预测误差分布', 
                         fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            ax2.legend(prop=fm.FontProperties(family=selected_font, size=10), loc='best')
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            rmse = np.sqrt(np.mean(errors**2))
            
            stats_text = f'误差均值: {mean_error:.4f}\n误差标准差: {std_error:.4f}\nRMSE: {rmse:.4f}'
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    fontproperties=fm.FontProperties(family=selected_font, size=10),
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'hi_curves_{bearing_name}.png')
            plt.savefig(save_path, dpi=self.visualization_dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"健康因子曲线图已保存: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"创建健康因子曲线图失败: {e}")
            traceback.print_exc()
            return None
    
    
    def create_residual_analysis(self, model_name, predictions, true_rul, bearing_name, output_dir):
        """
        创建残差分析图 (Per-Bearing Residual Analysis)
        
        参数:
            model_name: 模型名称
            predictions: 预测值数组
            true_rul: 真实RUL值数组
            bearing_name: 轴承名称
            output_dir: 输出目录
        """
        if not self.config.get('save_per_bearing_comparisons', True):
            return None
        
        try:
            if len(predictions) != len(true_rul):
                min_len = min(len(predictions), len(true_rul))
                predictions = predictions[:min_len]
                true_rul = true_rul[:min_len]
            
            # 计算残差
            residuals = predictions - true_rul
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 1. 残差散点图
            ax1 = axes[0, 0]
            scatter = ax1.scatter(true_rul, residuals, c=true_rul, cmap='viridis', 
                                 alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
            ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7)
            ax1.set_xlabel('真实RUL值', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_ylabel('残差 (预测 - 真实)', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_title(f'{model_name}残差分析', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax1.grid(True, alpha=0.3)
            
            # 添加颜色条
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label('真实RUL值', rotation=270, labelpad=15, fontproperties=fm.FontProperties(family=selected_font))
            
            # 2. 残差直方图
            ax2 = axes[0, 1]
            ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('残差', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_ylabel('频数', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_title('残差分布', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax2.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax2.text(0.05, 0.95, f'均值: {mean_residual:.3f}\n标准差: {std_residual:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top', 
                    fontproperties=fm.FontProperties(family=selected_font, size=10),
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 3. 残差与预测值的关系
            ax3 = axes[1, 0]
            ax3.scatter(predictions, residuals, alpha=0.6, s=30)
            ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax3.set_xlabel('预测RUL值', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_ylabel('残差', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_title('残差 vs 预测值', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax3.grid(True, alpha=0.3)
            
            # 4. 残差Q-Q图
            ax4 = axes[1, 1]
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=ax4)
            ax4.set_title('残差Q-Q图', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax4.grid(True, alpha=0.3)
            
            # 设置主标题
            plt.suptitle(f'{model_name}残差分析 - {bearing_name}', fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'residual_analysis_{model_name}_{bearing_name}.png')
            plt.savefig(save_path, dpi=self.visualization_dpi, bbox_inches='tight')
            plt.close(fig)
            
            return save_path
            
        except Exception as e:
            print(f"创建残差分析图失败: {e}")
            traceback.print_exc()
            return None


class ModelComparisonVisualizer:
    """模型对比可视化类"""
    
    def __init__(self, config):
        self.config = config
        self.visualization_style = config.get('visualization_style', 'seaborn-v0_8')
        self.visualization_dpi = config.get('visualization_dpi', 300)
        
        # 设置可视化样式
        try:
            plt.style.use(self.visualization_style)
        except:
            plt.style.use('default')
    
    def create_radar_chart(self, model_metrics, title="模型性能雷达图", output_path=None):
        """
        创建模型性能雷达图 (Model Performance Radar Chart)
        
        参数:
            model_metrics: 字典，键为模型名称，值为包含指标的字典
            title: 图表标题
            output_path: 输出路径（可选）
        """
        try:
            if not model_metrics:
                return None
            
            # 提取指标名称
            all_metrics = set()
            for metrics_dict in model_metrics.values():
                all_metrics.update(metrics_dict.keys())
            
            # 选择关键指标
            key_metrics = ['r2', 'rmse', 'mae', 'mape']
            available_metrics = [m for m in key_metrics if any(m in metrics for metrics in model_metrics.values())]
            
            if not available_metrics:
                return None
            
            # 准备数据
            labels = available_metrics
            num_vars = len(labels)
            
            # 计算每个模型的数据点
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # 闭合雷达图
            
            # 创建图形
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            # 定义颜色
            colors = plt.cm.Set2(np.linspace(0, 1, len(model_metrics)))
            
            for idx, (model_name, metrics_dict) in enumerate(model_metrics.items()):
                values = []
                for metric in labels:
                    if metric in metrics_dict:
                        # 标准化指标值（R²越大越好，其他越小越好）
                        if metric == 'r2':
                            values.append(metrics_dict[metric])
                        else:
                            # 将误差指标转换为0-1范围（越小越好）
                            max_val = max(m.get(metric, 0) for m in model_metrics.values() if metric in m)
                            if max_val > 0:
                                values.append(1 - metrics_dict[metric] / max_val)
                            else:
                                values.append(0)
                    else:
                        values.append(0)
                
                # 闭合雷达图
                values += values[:1]
                
                # 绘制雷达图
                ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
                ax.fill(angles, values, alpha=0.25, color=colors[idx])
            
            # 设置刻度标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([self._get_metric_label(m) for m in labels], fontproperties=fm.FontProperties(family=selected_font, size=10))
            
            # 设置范围
            ax.set_ylim(0, 1)
            
            # 添加标题和图例
            ax.set_title(title, size=16, fontproperties=fm.FontProperties(family=selected_font, weight='bold'), pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop=fm.FontProperties(family=selected_font, size=10))
            
            # 添加网格
            ax.grid(True)
            
            plt.tight_layout()
            
            # 保存或显示
            if output_path:
                plt.savefig(output_path, dpi=self.visualization_dpi, bbox_inches='tight')
                plt.close(fig)
                return output_path
            else:
                return fig
            
        except Exception as e:
            print(f"创建雷达图失败: {e}")
            traceback.print_exc()
            return None
    
    def create_metrics_bar_chart(self, model_metrics, title="模型性能指标对比", output_path=None):
        """
        创建模型性能指标对比图 (Model Performance Metrics Bar Chart)
        
        参数:
            model_metrics: 字典，键为模型名称，值为包含指标的字典
            title: 图表标题
            output_path: 输出路径（可选）
        """
        try:
            if not model_metrics:
                return None
            
            # 提取指标
            all_metrics = set()
            for metrics_dict in model_metrics.values():
                all_metrics.update(metrics_dict.keys())
            
            # 选择要显示的指标
            key_metrics = ['r2', 'rmse', 'mae', 'mape']
            metrics_to_show = [m for m in key_metrics if m in all_metrics]
            
            if not metrics_to_show:
                return None
            
            # 创建图形
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics_to_show):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                model_names = list(model_metrics.keys())
                metric_values = [model_metrics[model].get(metric, 0) for model in model_names]
                
                # 创建柱状图
                bars = ax.bar(range(len(model_names)), metric_values, color=plt.cm.tab10(range(len(model_names))))
                ax.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
                ax.set_ylabel(self._get_metric_label(metric), fontproperties=fm.FontProperties(family=selected_font, size=11))
                ax.set_title(f'{self._get_metric_label(metric)}对比', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names, rotation=45, ha='right', fontproperties=fm.FontProperties(family=selected_font, size=10))
                ax.grid(True, alpha=0.3)
                
                # 在柱子上添加数值
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}' if metric != 'mape' else f'{height:.1f}%',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom', fontsize=9)
            
            # 隐藏多余的子图
            for idx in range(len(metrics_to_show), len(axes)):
                axes[idx].axis('off')
            
            plt.suptitle(title, fontproperties=fm.FontProperties(family=selected_font, size=16, weight='bold'))
            plt.tight_layout()
            
            # 保存或显示
            if output_path:
                plt.savefig(output_path, dpi=self.visualization_dpi, bbox_inches='tight')
                plt.close(fig)
                return output_path
            else:
                return fig
            
        except Exception as e:
            print(f"创建指标对比图失败: {e}")
            traceback.print_exc()
            return None
    
    def create_cross_bearing_boxplots(self, all_results, metrics=['r2', 'rmse', 'mae'], 
                                     title="跨轴承模型性能箱线图", output_path=None):
        """
        创建跨轴承性能箱线图 (Cross-Bearing Performance Box Plots)
        
        参数:
            all_results: 列表，每个元素是轴承的评估结果字典
            metrics: 要展示的指标列表
            title: 图表标题
            output_path: 输出路径（可选）
        """
        try:
            if not all_results:
                return None
            
            # 提取数据
            model_names = set()
            for result in all_results:
                if 'models_results' in result:
                    model_names.update(result['models_results'].keys())
            
            model_names = list(model_names)
            if not model_names:
                return None
            
            # 准备数据
            data_dict = {model: {metric: [] for metric in metrics} for model in model_names}
            
            for result in all_results:
                bearing_name = result.get('bearing_name', 'unknown')
                models_results = result.get('models_results', {})
                
                for model_name in model_names:
                    if model_name in models_results:
                        for metric in metrics:
                            if 'clean' in models_results[model_name]['results']:
                                metric_value = models_results[model_name]['results']['clean'].get(metric, 0)
                                data_dict[model_name][metric].append(metric_value)
            
            # 创建图形
            num_metrics = len(metrics)
            fig, axes = plt.subplots(1, num_metrics, figsize=(5*num_metrics, 8))
            
            if num_metrics == 1:
                axes = [axes]
            
            for idx, metric in enumerate(metrics):
                ax = axes[idx] if idx < len(axes) else None
                if ax is None:
                    continue
                
                # 准备箱线图数据
                box_data = []
                labels = []
                
                for model_name in model_names:
                    values = data_dict[model_name][metric]
                    if values:  # 只添加有数据的模型
                        box_data.append(values)
                        labels.append(model_name)
                
                if box_data:
                    # 创建箱线图
                    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                    
                    # 设置颜色
                    colors = plt.cm.Set2(np.linspace(0, 1, len(box_data)))
                    for patch, color in zip(bp['boxes'], colors):
                        patch.set_facecolor(color)
                    
                    ax.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
                    ax.set_ylabel(self._get_metric_label(metric), fontproperties=fm.FontProperties(family=selected_font, size=11))
                    ax.set_title(f'{self._get_metric_label(metric)}分布', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
                    ax.grid(True, alpha=0.3)
                    
                    # 旋转x轴标签
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontproperties=fm.FontProperties(family=selected_font, size=10))
            
            plt.suptitle(title, fontproperties=fm.FontProperties(family=selected_font, size=16, weight='bold'))
            plt.tight_layout()
            
            # 保存或显示
            if output_path:
                plt.savefig(output_path, dpi=self.visualization_dpi, bbox_inches='tight')
                plt.close(fig)
                return output_path
            else:
                return fig
            
        except Exception as e:
            print(f"创建箱线图失败: {e}")
            traceback.print_exc()
            return None
    
    def create_error_distribution_comparison(self, all_predictions, all_labels, 
                                           model_names, title="模型误差分布对比", 
                                           output_path=None):
        """
        创建模型预测误差分布对比图
        
        参数:
            all_predictions: 字典，键为模型名称，值为预测值列表的列表
            all_labels: 真实标签列表的列表
            model_names: 模型名称列表
            title: 图表标题
            output_path: 输出路径（可选）
        """
        try:
            if not all_predictions or not model_names:
                return None
            
            # 计算所有模型的绝对误差
            error_data = {}
            
            for model_name in model_names:
                if model_name in all_predictions:
                    model_errors = []
                    for pred_list, true_list in zip(all_predictions[model_name], all_labels):
                        if len(pred_list) == len(true_list):
                            errors = np.abs(np.array(pred_list) - np.array(true_list))
                            model_errors.extend(errors.tolist())
                    
                    if model_errors:
                        error_data[model_name] = model_errors
            
            if not error_data:
                return None
            
            # 创建图形
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # 1. 直方图对比
            ax1 = axes[0]
            colors = plt.cm.tab10(range(len(error_data)))
            
            for idx, (model_name, errors) in enumerate(error_data.items()):
                ax1.hist(errors, bins=30, alpha=0.6, color=colors[idx], 
                        label=model_name, edgecolor='black', linewidth=0.5)
            
            ax1.set_xlabel('绝对误差', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_ylabel('频数', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_title('绝对误差直方图', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax1.legend(prop=fm.FontProperties(family=selected_font, size=10))
            ax1.grid(True, alpha=0.3)
            
            # 2. 核密度估计图
            ax2 = axes[1]
            
            for idx, (model_name, errors) in enumerate(error_data.items()):
                from scipy.stats import gaussian_kde
                if len(errors) > 1:
                    kde = gaussian_kde(errors)
                    x_vals = np.linspace(min(errors), max(errors), 1000)
                    ax2.plot(x_vals, kde(x_vals), color=colors[idx], 
                            linewidth=2, label=model_name)
            
            ax2.set_xlabel('绝对误差', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_ylabel('概率密度', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_title('误差核密度估计', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax2.legend(prop=fm.FontProperties(family=selected_font, size=10))
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(title, fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            plt.tight_layout()
            
            # 保存或显示
            if output_path:
                plt.savefig(output_path, dpi=self.visualization_dpi, bbox_inches='tight')
                plt.close(fig)
                return output_path
            else:
                return fig
            
        except Exception as e:
            print(f"创建误差分布对比图失败: {e}")
            traceback.print_exc()
            return None
    
    def _get_metric_label(self, metric):
        """获取指标的中文标签"""
        labels = {
            'r2': 'R²分数',
            'rmse': 'RMSE',
            'mae': 'MAE',
            'mape': 'MAPE (%)',
            'mse': 'MSE'
        }
        return labels.get(metric, metric)


class VisualizationTool:
    """可视化工具类"""
    
    @staticmethod
    def visualize_model_comparison(models_results, output_dir, bearing_name):
        """创建模型对比可视化图表"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # 提取数据
            model_names = list(models_results.keys())
            noise_levels = []
            
            # 收集所有噪声水平
            for model_info in models_results.values():
                for test_desc in model_info['results'].keys():
                    if test_desc not in noise_levels:
                        noise_levels.append(test_desc)
            
            # 1. R²对比图
            ax1 = axes[0, 0]
            x = np.arange(len(model_names))
            width = 0.35
            
            for i, noise_desc in enumerate(noise_levels):
                r2_values = []
                for model_name in model_names:
                    if noise_desc in models_results[model_name]['results']:
                        r2_values.append(models_results[model_name]['results'][noise_desc]['r2'])
                    else:
                        r2_values.append(0.0)
                
                offset = (i - len(noise_levels)/2 + 0.5) * width
                bars = ax1.bar(x + offset, r2_values, width/len(noise_levels), label=noise_desc)
                
                # 在柱子上添加数值
                for bar in bars:
                    height = bar.get_height()
                    if height > 0.01:  # 只显示非零值
                        ax1.annotate(f'{height:.3f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3),  # 3 points vertical offset
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8)
            
            ax1.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_ylabel('R²分数', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax1.set_title('模型R²分数对比', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax1.set_xticks(x)
            ax1.set_xticklabels([name.upper() for name in model_names], fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax1.legend(prop=fm.FontProperties(family=selected_font, size=9))
            ax1.grid(True, alpha=0.3)
            
            # 2. RMSE对比图
            ax2 = axes[0, 1]
            for i, noise_desc in enumerate(noise_levels):
                rmse_values = []
                for model_name in model_names:
                    if noise_desc in models_results[model_name]['results']:
                        rmse_values.append(models_results[model_name]['results'][noise_desc]['rmse'])
                    else:
                        rmse_values.append(0.0)
                
                offset = (i - len(noise_levels)/2 + 0.5) * width
                ax2.bar(x + offset, rmse_values, width/len(noise_levels), label=noise_desc)
            
            ax2.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_ylabel('RMSE', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax2.set_title('模型RMSE对比', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax2.set_xticks(x)
            ax2.set_xticklabels([name.upper() for name in model_names], fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax2.legend(prop=fm.FontProperties(family=selected_font, size=9))
            ax2.grid(True, alpha=0.3)
            
            # 3. MAE对比图
            ax3 = axes[1, 0]
            for i, noise_desc in enumerate(noise_levels):
                mae_values = []
                for model_name in model_names:
                    if noise_desc in models_results[model_name]['results']:
                        mae_values.append(models_results[model_name]['results'][noise_desc]['mae'])
                    else:
                        mae_values.append(0.0)
                
                offset = (i - len(noise_levels)/2 + 0.5) * width
                ax3.bar(x + offset, mae_values, width/len(noise_levels), label=noise_desc)
            
            ax3.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_ylabel('MAE', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax3.set_title('模型MAE对比', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax3.set_xticks(x)
            ax3.set_xticklabels([name.upper() for name in model_names], fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax3.legend(prop=fm.FontProperties(family=selected_font, size=9))
            ax3.grid(True, alpha=0.3)
            
            # 4. MAPE对比图
            ax4 = axes[1, 1]
            for i, noise_desc in enumerate(noise_levels):
                mape_values = []
                for model_name in model_names:
                    if noise_desc in models_results[model_name]['results']:
                        mape_values.append(models_results[model_name]['results'][noise_desc]['mape'])
                    else:
                        mape_values.append(0.0)
                
                offset = (i - len(noise_levels)/2 + 0.5) * width
                bars = ax4.bar(x + offset, mape_values, width/len(noise_levels), label=noise_desc)
            
            ax4.set_xlabel('模型', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax4.set_ylabel('MAPE (%)', fontproperties=fm.FontProperties(family=selected_font, size=11))
            ax4.set_title('模型MAPE对比', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            ax4.set_xticks(x)
            ax4.set_xticklabels([name.upper() for name in model_names], fontproperties=fm.FontProperties(family=selected_font, size=10))
            ax4.legend(prop=fm.FontProperties(family=selected_font, size=9))
            ax4.grid(True, alpha=0.3)
            
            plt.suptitle(f'模型性能对比 - {bearing_name}', fontproperties=fm.FontProperties(family=selected_font, size=16, weight='bold'))
            plt.tight_layout()
            
            # 保存图表
            plot_path = output_dir / f"model_comparison_{bearing_name}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"创建模型对比可视化失败: {e}")
            traceback.print_exc()
            return None
    
    @staticmethod
    def visualize_results(preds, labels, metrics_dict, title="RUL预测结果"):
        """可视化预测结果"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. 预测 vs 真实值
            axes[0, 0].scatter(labels, preds, alpha=0.6, s=20)
            min_val = min(labels.min(), preds.min())
            max_val = max(labels.max(), preds.max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='理想线')
            axes[0, 0].set_xlabel('真实RUL', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[0, 0].set_ylabel('预测RUL', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[0, 0].set_title('预测结果对比', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            axes[0, 0].legend(prop=fm.FontProperties(family=selected_font, size=10))
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 预测误差分布
            axes[0, 1].hist(preds - labels, bins=30, edgecolor='black', alpha=0.7)
            axes[0, 1].axvline(x=0, color='r', linestyle='--')
            axes[0, 1].set_xlabel('预测误差', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[0, 1].set_ylabel('频数', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[0, 1].set_title('误差分布', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 预测趋势
            axes[1, 0].plot(labels[:200], 'b-', label='真实值', alpha=0.7, linewidth=2)
            axes[1, 0].plot(preds[:200], 'r-', label='预测值', alpha=0.7, linewidth=1)
            axes[1, 0].set_xlabel('样本索引', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[1, 0].set_ylabel('RUL', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[1, 0].set_title('预测趋势（前200个样本）', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            axes[1, 0].legend(prop=fm.FontProperties(family=selected_font, size=10))
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 指标汇总
            axes[1, 1].barh(range(4), [metrics_dict['r2'], metrics_dict['rmse'], 
                                      metrics_dict['mae'], metrics_dict.get('mape', 0)])
            axes[1, 1].set_yticks(range(4))
            axes[1, 1].set_yticklabels([r'R $ ^2 $ 分数', 'RMSE', 'MAE', 'MAPE(%)'], fontproperties=fm.FontProperties(family=selected_font, size=10))
            axes[1, 1].set_xlabel('值', fontproperties=fm.FontProperties(family=selected_font, size=11))
            axes[1, 1].set_title('关键指标', fontproperties=fm.FontProperties(family=selected_font, size=12, weight='bold'))
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(title, fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"可视化生成失败: {e}")
            return None