"""
通用工具模块 - 负责全局设置、工具函数和辅助功能
"""

import os
import sys
import logging
import warnings
import matplotlib as mpl
import matplotlib.font_manager as fm
import torch
from datetime import datetime
from pathlib import Path
from typing import Optional

# 忽略警告
warnings.filterwarnings('ignore')

# 检查scipy.ndimage.zoom可用性
try:
    from scipy.ndimage import zoom
    SCIPY_ZOOM_AVAILABLE = True
except ImportError:
    SCIPY_ZOOM_AVAILABLE = False
    print("警告: scipy.ndimage.zoom 不可用，将使用其他图像缩放方法")


def setup_matplotlib_fonts():
    """设置matplotlib字体，解决中文乱码问题"""
    # 尝试设置中文字体
    possible_fonts = [
        'SimHei',                    # Windows黑体
        'Microsoft YaHei',           # Windows微软雅黑
        'Noto Sans CJK SC',          # Linux/Google Noto字体
        'Source Han Sans SC',        # Adobe思源黑体
        'WenQuanYi Micro Hei',       # Linux文泉驿微米黑
        'DejaVu Sans'                # 备选：支持Unicode的字体
    ]
    
    font_found = False
    selected_font = None
    
    # 方法1: 尝试通过rcParams设置
    for font_name in possible_fonts:
        try:
            # 检查字体是否可用
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            if any(font_name.lower() in f.lower() for f in available_fonts):
                mpl.rcParams['font.family'] = font_name
                selected_font = font_name
                font_found = True
                print(f"成功设置中文字体: {font_name}")
                break
        except Exception as e:
            continue
    
    # 方法2: 如果上述方法失败，尝试直接设置字体路径
    if not font_found:
        # 常见的中文字体文件路径
        font_paths = [
            'C:/Windows/Fonts/simhei.ttf',      # Windows黑体
            'C:/Windows/Fonts/msyh.ttc',        # Windows微软雅黑
            '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',  # Linux
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Linux
        ]
        
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    # 添加字体
                    fm.fontManager.addfont(font_path)
                    font_prop = fm.FontProperties(fname=font_path)
                    font_name = font_prop.get_name()
                    mpl.rcParams['font.family'] = font_name
                    selected_font = font_name
                    font_found = True
                    print(f"从文件加载中文字体: {font_path} -> {font_name}")
                    break
                except Exception as e:
                    continue
    
    # 如果仍然没有找到中文字体，使用默认字体并给出警告
    if not font_found:
        print("警告: 未找到可用的中文字体，将使用默认字体，中文可能显示为方块或乱码。")
        print("建议: 安装中文字体如 'SimHei' 或 'Noto Sans CJK'")
        # 使用支持Unicode的字体
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        selected_font = 'DejaVu Sans'
    
    # 设置其他字体相关参数
    mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    mpl.rcParams['font.size'] = 12  # 设置默认字体大小
    
    # 返回选中的字体，便于后续使用
    return selected_font


def setup_device(force_cpu=False):
    """设置全局设备（GPU/CPU）"""
    if force_cpu:
        device = torch.device('cpu')
        print(f"强制使用CPU")
        return device
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        # 可以选择特定GPU设备
        device_id = 0  # 默认使用第一个GPU
        device = torch.device(f'cuda:{device_id}')
        
        # 打印GPU信息
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"GPU设备: {torch.cuda.get_device_name(device_id)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.2f} GB")
        
        # 设置CUDA优化
        torch.backends.cudnn.benchmark = True  # 启用cuDNN自动优化
        torch.backends.cudnn.enabled = True
        
    else:
        device = torch.device('cpu')
        print(f"PyTorch版本: {torch.__version__}")
        print("CUDA不可用，使用CPU")
    
    return device


# 删除重复的show_device_info函数，只保留一个
def show_device_info():
    """显示设备信息"""
    # 注意：这里需要从全局变量获取DEVICE，但为了避免循环导入
    # 可以在调用时传入device参数，或者使用全局变量
    print("\n" + "="*50)
    print("设备信息")
    print("="*50)
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"     内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        # 注意：这里使用torch.cuda.current_device()而不是全局DEVICE
        current_device = torch.cuda.current_device()
        print(f"当前设备: cuda:{current_device}")
    else:
        print("CPU设备: 使用CPU进行训练和推理")
    print("="*50)


# 初始化字体设置
selected_font = setup_matplotlib_fonts()

# 全局设备变量
DEVICE = setup_device()


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """配置日志系统"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_processing_{timestamp}.log")
    
    logger = logging.getLogger("BatchProcessor")
    logger.setLevel(logging.INFO)
    
    # 清除现有处理器
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                      datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # 使用全局 DEVICE
    logger.info(f"使用设备: {DEVICE}")
    if DEVICE.type == 'cuda':
        logger.info(f"GPU内存信息: 总量={torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB, "
                   f"已分配={torch.cuda.memory_allocated(0) / 1024**3:.2f}GB, "
                   f"缓存={torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")
    
    return logger

def get_memory_usage():
    """【新增】获取当前内存使用情况"""
    import psutil
    import torch
    
    memory_info = {}
    
    # CPU内存
    cpu_memory = psutil.virtual_memory()
    memory_info['cpu_total_gb'] = cpu_memory.total / 1024**3
    memory_info['cpu_available_gb'] = cpu_memory.available / 1024**3
    memory_info['cpu_used_gb'] = cpu_memory.used / 1024**3
    memory_info['cpu_percent'] = cpu_memory.percent
    
    # GPU内存（如果可用）
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_info[f'gpu_{i}_total_gb'] = torch.cuda.get_device_properties(i).total_memory / 1024**3
            memory_info[f'gpu_{i}_allocated_gb'] = torch.cuda.memory_allocated(i) / 1024**3
            memory_info[f'gpu_{i}_reserved_gb'] = torch.cuda.memory_reserved(i) / 1024**3
            memory_info[f'gpu_{i}_free_gb'] = memory_info[f'gpu_{i}_total_gb'] - memory_info[f'gpu_{i}_allocated_gb']
            memory_info[f'gpu_{i}_percent'] = (memory_info[f'gpu_{i}_allocated_gb'] / memory_info[f'gpu_{i}_total_gb']) * 100
    
    return memory_info


def print_memory_usage():
    """【新增】打印内存使用情况"""
    mem_info = get_memory_usage()
    
    print("\n" + "="*50)
    print("内存使用情况")
    print("="*50)
    print(f"CPU内存: 已用={mem_info['cpu_used_gb']:.2f}GB / 总计={mem_info['cpu_total_gb']:.2f}GB ({mem_info['cpu_percent']:.1f}%)")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} 内存: 已分配={mem_info[f'gpu_{i}_allocated_gb']:.2f}GB / "
                  f"总计={mem_info[f'gpu_{i}_total_gb']:.2f}GB ({mem_info[f'gpu_{i}_percent']:.1f}%)")
    print("="*50)
