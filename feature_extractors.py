"""
特征提取模块 - 负责时频域特征提取和CWT特征提取功能
"""

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from scipy import interpolate
from scipy.signal import hilbert, welch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from utils import SCIPY_ZOOM_AVAILABLE, selected_font

try:
    from scipy.ndimage import zoom
except ImportError:
    pass  # 已经在utils中处理


class TimeFrequencyFeatureExtractor:
    """时频域特征提取器"""
    
    def __init__(self, sampling_rate=25600, wavelet='db4', level=4):
        self.sampling_rate = sampling_rate
        self.wavelet = wavelet
        self.level = level
    
    def extract_features(self, signal):
        """从信号中提取特征"""
        signal = signal.astype(np.float64)
        
        if not self._is_signal_valid(signal):
            return self._get_default_features()
        
        try:
            features = []
            
            # 1. 基础时域特征
            features.extend(self._extract_time_domain_features(signal))
            
            # 2. 统计特征
            features.extend(self._extract_statistical_features(signal))
            
            # 3. 频域特征
            features.extend(self._extract_frequency_domain_features(signal))
            
            # 4. 小波特征
            features.extend(self._extract_wavelet_features(signal))
            
            # 转换为numpy数组
            features_array = np.array(features, dtype=np.float64)
            
            # 检查是否有NaN或inf
            if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
                return self._get_default_features()
            
            return features_array
            
        except Exception as e:
            print(f"特征提取出错: {e}")
            return self._get_default_features()
    
    def _is_signal_valid(self, signal):
        """检查信号是否有效"""
        if signal is None or len(signal) == 0:
            return False
        
        if np.std(signal) < 1e-10:
            return False
            
        return True
    
    def _get_default_features(self):
        """获取默认特征（用于错误处理）"""
        return np.zeros(26, dtype=np.float64)
    
    def _extract_time_domain_features(self, signal):
        """提取时域特征"""
        features = []
        
        # 基本统计
        features.append(float(np.mean(signal)))          # 均值
        features.append(float(np.std(signal)))           # 标准差
        features.append(float(np.max(signal)))           # 最大值
        features.append(float(np.min(signal)))           # 最小值
        features.append(float(np.ptp(signal)))           # 峰峰值
        features.append(float(np.sqrt(np.mean(signal**2))))  # RMS
        features.append(float(np.mean(np.abs(signal))))      # 平均绝对值
        
        # 形状特征
        rms = features[5]
        peak = max(abs(features[2]), abs(features[3]))
        mean_abs = features[6]
        
        if rms > 0:
            features.append(float(peak / rms))  # 峰值因子
        else:
            features.append(0.0)
            
        if mean_abs > 0:
            features.append(float(rms / mean_abs))  # 形状因子
            features.append(float(peak / mean_abs))  # 脉冲因子
        else:
            features.extend([0.0, 0.0])
        
        # 统计特征
        if len(signal) > 2:
            features.append(float(skew(signal)))  # 偏度
        else:
            features.append(0.0)
            
        if len(signal) > 3:
            features.append(float(kurtosis(signal, fisher=True)))  # 峰度
        else:
            features.append(0.0)
        
        return features
    
    def _extract_statistical_features(self, signal):
        """提取统计特征"""
        features = []
        
        # 信息熵
        try:
            hist, _ = np.histogram(signal, bins=min(50, len(signal)//2))
            hist = hist[hist > 0]
            prob = hist / len(signal)
            entropy = float(-np.sum(prob * np.log(prob + 1e-10)))
            features.append(entropy)
        except:
            features.append(0.0)
        
        # 过零率
        try:
            zero_crossings = np.sum(np.diff(np.sign(signal)) != 0)
            features.append(float(zero_crossings / len(signal)))
        except:
            features.append(0.0)
        
        return features
    
    def _extract_frequency_domain_features(self, signal):
        """提取频域特征"""
        features = []
        
        try:
            n = len(signal)
            fft_vals = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(n, 1/self.sampling_rate)
            
            if len(fft_vals) > 0:
                # 频域统计
                features.append(float(np.mean(fft_vals)))    # 频域均值
                features.append(float(np.max(fft_vals)))     # 频域最大值
                features.append(float(np.sum(fft_vals)))     # 频域总和
                
                # 主频位置
                features.append(float(np.argmax(fft_vals)))  # 主频位置
                
                # 频谱质心
                total_energy = np.sum(fft_vals**2)
                if total_energy > 0:
                    spectral_centroid = float(np.sum(freqs * fft_vals**2) / total_energy)
                else:
                    spectral_centroid = 0.0
                features.append(spectral_centroid)
                
                # 频带能量比
                n_freq = len(fft_vals)
                if n_freq >= 3:
                    third = n_freq // 3
                    low_band = np.sum(fft_vals[:third]**2)
                    mid_band = np.sum(fft_vals[third:2*third]**2)
                    high_band = np.sum(fft_vals[2*third:]**2)
                    total_energy = low_band + mid_band + high_band
                    
                    if total_energy > 0:
                        features.append(float(low_band / total_energy))
                        features.append(float(mid_band / total_energy))
                        features.append(float(high_band / total_energy))
                    else:
                        features.extend([0.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 0.0, 0.0])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
        except Exception as e:
            print(f"频域特征提取出错: {e}")
            features.extend([0.0] * 9)
            
        return features
    
    def _extract_wavelet_features(self, signal):
        """提取小波特征"""
        features = []
        
        try:
            # 小波变换
            coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
            
            # 计算各层能量
            wavelet_energies = []
            for coeff in coeffs:
                if len(coeff) > 1:
                    wavelet_energies.append(float(np.sum(coeff**2)))
            
            if wavelet_energies:
                features.append(float(np.sum(wavelet_energies)))  # 总能量
                if wavelet_energies[0] > 0:
                    features.append(float(wavelet_energies[-1] / wavelet_energies[0]))  # 能量比
                else:
                    features.append(0.0)
            else:
                features.extend([0.0, 0.0])
                
        except Exception as e:
            print(f"小波特征提取出错: {e}")
            features.extend([0.0, 0.0])
            
        return features
    
    def get_feature_dimension(self):
        """获取特征维度"""
        dummy_signal = np.random.randn(512)
        features = self.extract_features(dummy_signal)
        return len(features)


class CWTFeatureExtractor:
    """连续小波变换(CWT)特征提取器"""
    
    def __init__(self, 
                 wavelet='cmor1.0-1.0',  # 使用具体的小波名称
                 scales=None,
                 sampling_rate=25600,
                 width_band=5.0,
                 normalize=True):
        """
        初始化CWT特征提取器
        
        参数:
            wavelet: 小波类型，使用具体格式如 'cmor1.0-1.0' 或 'cgau1'
            scales: 尺度参数列表，如果不提供则自动计算
            sampling_rate: 采样率
            width_band: 带宽参数（仅用于morlet小波）
            normalize: 是否对CWT结果进行归一化
        """
        self.wavelet = wavelet
        self.scales = scales
        self.sampling_rate = sampling_rate
        self.width_band = width_band
        self.normalize = normalize
        
        if scales is None:
            # 自动计算尺度参数
            self.scales = np.arange(1, 65)  # 64个尺度
    
    def apply_cwt(self, signal, scales=None):
        """
        对信号应用连续小波变换
        
        参数:
            signal: 一维振动信号
            scales: 尺度参数列表
            
        返回:
            cwt_matrix: CWT结果矩阵，形状为 (scales, time_steps)
        """
        if scales is None:
            scales = self.scales
        
        # 确保信号是一维的
        signal = np.asarray(signal).flatten()
        
        try:
            # 直接使用 PyWavelets 的 cwt 函数
            # pywt.cwt 返回 (coefficients, frequencies)
            cwt_coeffs, frequencies = pywt.cwt(signal, scales, self.wavelet)
            
            # 取绝对值，表示幅度
            cwt_matrix = np.abs(cwt_coeffs)
            
            # 归一化
            if self.normalize:
                # 避免除零
                cwt_max = np.max(cwt_matrix)
                if cwt_max > 1e-10:
                    cwt_matrix = cwt_matrix / cwt_max
                else:
                    cwt_matrix = np.zeros_like(cwt_matrix)
            
            return cwt_matrix
            
        except Exception as e:
            print(f"CWT变换出错: {e}")
            # 返回零矩阵
            return np.zeros((len(scales), len(signal)))
    
    def cwt_to_image(self, signal, target_shape=(64, 64)):
        """
        将CWT结果转换为图像格式
        
        参数:
            signal: 输入信号
            target_shape: 目标图像形状 (height, width)
            
        返回:
            cwt_image: CWT图像，形状为 (1, height, width)
        """
        # 应用CWT
        cwt_matrix = self.apply_cwt(signal)
        
        # 调整大小到目标形状
        try:
            if SCIPY_ZOOM_AVAILABLE:
                # 计算缩放因子
                zoom_factors = (
                    target_shape[0] / cwt_matrix.shape[0],
                    target_shape[1] / cwt_matrix.shape[1]
                )
                # 使用缩放插值
                cwt_resized = zoom(cwt_matrix, zoom_factors, order=1)
            else:
                # 使用 scipy 插值
                x = np.linspace(0, 1, cwt_matrix.shape[1])
                y = np.linspace(0, 1, cwt_matrix.shape[0])
                f = interpolate.interp2d(x, y, cwt_matrix, kind='linear')
                
                # 在新网格上插值
                x_new = np.linspace(0, 1, target_shape[1])
                y_new = np.linspace(0, 1, target_shape[0])
                cwt_resized = f(x_new, y_new)
                
        except Exception as e:
            print(f"图像缩放出错: {e}")
            # 如果所有方法都失败，使用简单裁剪
            cwt_resized = cwt_matrix[:min(target_shape[0], cwt_matrix.shape[0]), 
                                    :min(target_shape[1], cwt_matrix.shape[1])]
            # 如果需要填充
            if cwt_resized.shape[0] < target_shape[0] or cwt_resized.shape[1] < target_shape[1]:
                padded = np.zeros(target_shape)
                padded[:cwt_resized.shape[0], :cwt_resized.shape[1]] = cwt_resized
                cwt_resized = padded
        
        # 添加通道维度
        cwt_image = cwt_resized[np.newaxis, :, :]  # (1, H, W)
        
        return cwt_image.astype(np.float32)
    
    def extract_multiscale_features(self, signal, scales_list=None):
        """
        提取多尺度CWT特征
        
        参数:
            signal: 输入信号
            scales_list: 不同尺度的列表
            
        返回:
            multiscale_features: 多尺度特征列表
        """
        if scales_list is None:
            # 使用不同的尺度范围
            scales_list = [
                np.arange(1, 33),      # 细尺度
                np.arange(33, 65),     # 中等尺度
                np.arange(65, 97)      # 粗尺度
            ]
        
        features = []
        for scales in scales_list:
            cwt_feat = self.apply_cwt(signal, scales)
            # 计算统计特征
            features.extend([
                np.mean(cwt_feat),
                np.std(cwt_feat),
                np.max(cwt_feat),
                np.min(cwt_feat),
                np.sum(cwt_feat)
            ])
        
        return np.array(features)
    
    def visualize_and_save_cwt(self, signal_segment, save_path, title="CWT Magnitude Spectrum", 
                               dpi=150, figsize=(12, 8)):
        """
        可视化并保存CWT时频图
        
        参数:
            signal_segment: 信号片段
            save_path: 保存路径
            title: 图像标题
            dpi: 图像DPI
            figsize: 图像大小
        """
        try:
            # 创建图形
            fig, axes = plt.subplots(2, 1, figsize=figsize)
            
            # 1. 原始信号
            time = np.arange(len(signal_segment)) / self.sampling_rate if self.sampling_rate > 0 else np.arange(len(signal_segment))
            axes[0].plot(time, signal_segment, 'b-', linewidth=1)
            axes[0].set_xlabel('时间 (秒)' if self.sampling_rate > 0 else '采样点', fontproperties=fm.FontProperties(family=selected_font))
            axes[0].set_ylabel('振幅', fontproperties=fm.FontProperties(family=selected_font))
            axes[0].set_title(f'原始振动信号 (长度={len(signal_segment)})', fontproperties=fm.FontProperties(family=selected_font, size=12))
            axes[0].grid(True, alpha=0.3)
            
            # 2. CWT时频图
            cwt_matrix = self.apply_cwt(signal_segment)
            
            if cwt_matrix.shape[0] == 0 or cwt_matrix.shape[1] == 0:
                print(f"警告: CWT矩阵为空，无法可视化")
                plt.close(fig)
                return
            
            im = axes[1].imshow(cwt_matrix, aspect='auto', cmap='jet', 
                               extent=[0, len(signal_segment), 1, len(self.scales)])
            axes[1].set_xlabel('时间 (采样点)', fontproperties=fm.FontProperties(family=selected_font))
            axes[1].set_ylabel('尺度', fontproperties=fm.FontProperties(family=selected_font))
            axes[1].set_title(f'{title} - {self.wavelet.capitalize()}小波', fontproperties=fm.FontProperties(family=selected_font, size=12))
            
            # 添加颜色条
            cbar = fig.colorbar(im, ax=axes[1])
            cbar.set_label('幅度', fontproperties=fm.FontProperties(family=selected_font))
            
            # 使用fontproperties设置主标题
            plt.suptitle(f'CWT时频分析: {title}', fontproperties=fm.FontProperties(family=selected_font, size=14, weight='bold'))
            plt.tight_layout()
            
            # 保存图像
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            
            print(f"CWT时频图已保存: {save_path}")
            
        except Exception as e:
            print(f"可视化CWT时出错: {e}")
            traceback.print_exc()
            
    def generate_key_timepoint_cwt_visualizations(self, full_signal, window_size, overlap_ratio, 
                                                  output_dir, bearing_name, config):
        """
        生成关键时间点的CWT可视化
        
        参数:
            full_signal: 完整信号
            window_size: 窗口大小
            overlap_ratio: 重叠率
            output_dir: 输出目录
            bearing_name: 轴承名称
            config: 配置对象
        """
        if not config.get('save_cwt_images', True):
            print("CWT可视化功能已禁用")
            return
        
        # 计算步长
        step_size = int(window_size * (1 - overlap_ratio))
        if step_size <= 0:
            step_size = window_size // 4
        
        # 确定要可视化的时间点
        visualization_points = self._get_visualization_points(config, len(full_signal), window_size, step_size)
        
        # 创建可视化目录
        viz_dir = Path(output_dir) / config.get('cwt_visualization_dir', 'cwt_visualizations')
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"生成CWT时频图，关键时间点: {visualization_points}")
        
        for i, point in enumerate(visualization_points):
            try:
                # 提取信号片段
                start_idx = point['start_idx']
                end_idx = min(start_idx + window_size, len(full_signal))
                
                if end_idx <= start_idx:
                    print(f"警告: 无效的时间点 {start_idx}-{end_idx}")
                    continue
                
                segment = full_signal[start_idx:end_idx]
                
                # 确定标题
                if point['type'] == 'early':
                    title = f"早期 (RUL较高) - 窗口 {point['window_index']}"
                elif point['type'] == 'mid':
                    title = f"中期 (RUL中等) - 窗口 {point['window_index']}"
                elif point['type'] == 'late':
                    title = f"晚期 (RUL较低) - 窗口 {point['window_index']}"
                else:
                    title = f"自定义点 {i+1} - 窗口 {point['window_index']}"
                
                # 生成文件名
                filename = f"cwt_{bearing_name}_{point['type']}_window{point['window_index']}_idx{start_idx}-{end_idx}.png"
                save_path = viz_dir / filename
                
                # 可视化并保存
                self.visualize_and_save_cwt(
                    segment, 
                    str(save_path), 
                    title=title,
                    dpi=config.get('cwt_image_dpi', 150)
                )
                
            except Exception as e:
                print(f"处理时间点 {point} 时出错: {e}")
    
    def _get_visualization_points(self, config, signal_length, window_size, step_size):
        """
        获取要可视化的时间点
        
        参数:
            config: 配置对象
            signal_length: 信号总长度
            window_size: 窗口大小
            step_size: 步长
            
        返回:
            包含时间点信息的字典列表
        """
        # 计算可能的窗口数量
        num_windows = max(0, (signal_length - window_size) // step_size + 1)
        
        # 获取配置中的可视化点
        points_config = config.get('cwt_visualization_points', 'default')
        
        if points_config == 'default':
            # 默认：早期、中期、晚期各一个窗口
            if num_windows >= 3:
                early_idx = 0
                mid_idx = num_windows // 2
                late_idx = num_windows - 1
                
                points = [
                    {'type': 'early', 'window_index': early_idx, 'start_idx': early_idx * step_size},
                    {'type': 'mid', 'window_index': mid_idx, 'start_idx': mid_idx * step_size},
                    {'type': 'late', 'window_index': late_idx, 'start_idx': late_idx * step_size}
                ]
            elif num_windows > 0:
                # 如果窗口数少于3，只取第一个窗口
                points = [{'type': 'early', 'window_index': 0, 'start_idx': 0}]
            else:
                points = []
                
        elif isinstance(points_config, list):
            # 自定义索引列表
            points = []
            for i, idx in enumerate(points_config):
                if isinstance(idx, str) and idx.startswith('rel_'):
                    # 相对位置，如 'rel_0.0', 'rel_0.5', 'rel_1.0'
                    rel_pos = float(idx[4:])
                    window_idx = int(rel_pos * (num_windows - 1))
                else:
                    # 绝对索引
                    window_idx = int(idx) if idx >= 0 else num_windows + idx
                
                window_idx = max(0, min(window_idx, num_windows - 1))
                start_idx = window_idx * step_size
                
                point_type = f'custom_{i+1}'
                points.append({'type': point_type, 'window_index': window_idx, 'start_idx': start_idx})
        
        else:
            points = []
        
        return points