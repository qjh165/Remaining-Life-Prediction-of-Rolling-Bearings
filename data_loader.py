"""
数据加载模块 - 负责XJTU-SY轴承数据的加载功能
"""

import os
import numpy as np
import pandas as pd
import traceback
from typing import Tuple, Optional, List


class XJTUDataLoader:
    """XJTU-SY轴承数据加载器"""
    
    def __init__(self, vibration_column='horizontal'):
        self.vibration_column = vibration_column
        self.col_idx = 0 if vibration_column == 'horizontal' else 1
        
    def load_bearing_data(self, bearing_folder_path):
        """加载单个轴承的所有CSV文件数据，计算真实HI值"""
        try:
            # 获取所有CSV文件
            csv_files = []
            for f in os.listdir(bearing_folder_path):
                if f.endswith('.csv'):
                    csv_files.append(f)
            
            if not csv_files:
                print(f"警告: 文件夹 {bearing_folder_path} 中没有CSV文件")
                return None, None, None  # 修改返回值为三个
            
            # 按文件名数字排序
            def extract_number(filename):
                base_name = os.path.splitext(filename)[0]
                try:
                    return int(base_name)
                except:
                    return 0
            
            csv_files.sort(key=extract_number)
            
            total_files = len(csv_files)
            print(f"找到 {total_files} 个CSV文件")
            
            all_signals = []
            rul_values = []
            hi_values = []  # 新增HI值列表
            
            print("开始读取CSV文件...")
            
            for i, csv_file in enumerate(csv_files):
                file_path = os.path.join(bearing_folder_path, csv_file)
                
                try:
                    # 读取文件，自动检测是否有表头
                    df = self._read_csv_file(file_path)
                    
                    if df.empty:
                        print(f"警告: 文件 {csv_file} 为空")
                        continue
                    
                    # 检查列数
                    if df.shape[1] <= self.col_idx:
                        print(f"警告: 文件 {csv_file} 只有 {df.shape[1]} 列，跳过")
                        continue
                    
                    # 提取振动信号列
                    vibration_data = df.iloc[:, self.col_idx]
                    
                    # 转换为数值
                    vibration_signal = pd.to_numeric(vibration_data, errors='coerce').values
                    
                    # 删除NaN值
                    nan_count = np.sum(np.isnan(vibration_signal))
                    if nan_count > 0:
                        print(f"  文件 {csv_file} 有 {nan_count} 个NaN值，已删除")
                        vibration_signal = vibration_signal[~np.isnan(vibration_signal)]
                    
                    # 检查信号长度
                    if len(vibration_signal) < 100:
                        print(f"警告: 文件 {csv_file} 信号太短 ({len(vibration_signal)} 个点)")
                        continue
                    
                    # 处理可能的inf值
                    vibration_signal = np.nan_to_num(vibration_signal, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # 检查信号是否有效
                    if np.std(vibration_signal) < 1e-10:
                        print(f"警告: 文件 {csv_file} 信号方差太小，跳过")
                        continue
                    
                    # 标准化
                    signal_mean = np.mean(vibration_signal)
                    signal_std = np.std(vibration_signal)
                    if signal_std > 1e-10:
                        vibration_signal = (vibration_signal - signal_mean) / signal_std
                    
                    all_signals.append(vibration_signal.astype(np.float64))
                    
                    # 计算RUL（假设最后一个文件RUL=0）
                    current_rul = float(total_files - i - 1)
                    rul_values.append(current_rul)
                    
                    # 计算健康因子 HI = 1 - current_cycle / total_life
                    # current_cycle = i, total_life = total_files - 1
                    current_cycle = i
                    total_life = total_files - 1
                    hi = 1 - (current_cycle / total_life) if total_life > 0 else 0
                    hi_values.append(hi)
                    
                    # 进度显示
                    if (i + 1) % 10 == 0:
                        print(f"  已处理 {i + 1}/{total_files} 个文件")
                        
                except Exception as e:
                    print(f"处理文件 {csv_file} 时出错: {e}")
                    continue
            
            if not all_signals:
                print("错误: 没有成功加载任何信号")
                return None, None, None
            
            # 合并所有信号
            full_signal = np.concatenate(all_signals)
            
            print(f"\n数据加载完成:")
            print(f"  总信号长度: {len(full_signal):,}")
            print(f"  处理文件数: {len(all_signals)}")
            print(f"  RUL标签范围: {min(rul_values):.0f} 到 {max(rul_values):.0f}")
            print(f"  HI标签范围: {min(hi_values):.3f} 到 {max(hi_values):.3f}")
            print(f"  HI示例: 开始={hi_values[0]:.3f}, 中间={hi_values[len(hi_values)//2]:.3f}, 结束={hi_values[-1]:.3f}")
            
            return full_signal, np.array(rul_values, dtype=np.float64), np.array(hi_values, dtype=np.float64)
            
        except Exception as e:
            print(f"加载轴承数据时发生错误: {e}")
            traceback.print_exc()
            return None, None, None
    
    def _read_csv_file(self, file_path):
        """读取CSV文件，自动检测是否有表头"""
        try:
            # 先读取前3行来判断是否有表头
            df_sample = pd.read_csv(file_path, header=None, nrows=3)
            
            # 检查第一行是否包含字符串（可能是表头）
            first_row_has_string = False
            for col in range(df_sample.shape[1]):
                cell_value = df_sample.iloc[0, col]
                if isinstance(cell_value, str):
                    try:
                        float(cell_value)
                    except ValueError:
                        first_row_has_string = True
                        break
            
            if first_row_has_string:
                # 有表头，使用header=0
                df = pd.read_csv(file_path, header=0)
                df.columns = range(len(df.columns))
            else:
                # 无表头，使用header=None
                df = pd.read_csv(file_path, header=None)
            
            return df
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {e}")
            return pd.DataFrame()