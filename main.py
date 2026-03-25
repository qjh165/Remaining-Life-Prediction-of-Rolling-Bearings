"""
程序入口模块 - 负责命令行界面和主函数调用
"""

import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 直接导入（不使用相对导入）
from config import BatchConfig
from utils import setup_logging, show_device_info, DEVICE
from runners import (
    EnhancedBatchRULProcessor, 
    EnhancedMultiModalBatchProcessor,
    create_enhanced_multimodal_config_template,
    create_basic_config_template
)
from models import MultiModalRULPredictor


def load_and_process_data(config, processor_type='standard'):
    """加载和处理数据"""
    import logging
    
    logger = logging.getLogger("DataLoader")
    logger.info("开始加载和处理数据...")
    
    # 根据处理器类型创建相应的处理器
    if processor_type == 'multimodal':
        processor = EnhancedMultiModalBatchProcessor(config, logger)
    else:
        processor = EnhancedBatchRULProcessor(config, logger)
    
    # 查找轴承文件夹
    bearing_folders = processor.find_bearing_folders()
    
    if not bearing_folders:
        logger.error("未找到任何轴承文件夹")
        return None, None
    
    logger.info(f"找到 {len(bearing_folders)} 个轴承文件夹")
    
    # 打印数据集概览
    logger.info("=" * 60)
    logger.info("数据集概览:")
    logger.info(f"  总轴承数: {len(bearing_folders)}")
    logger.info(f"  数据根目录: {config['data_root']}")
    logger.info(f"  窗口大小: {config['window_size']}")
    logger.info(f"  重叠率: {config['overlap_ratio']}")
    logger.info(f"  步长: {int(config['window_size'] * (1 - config['overlap_ratio']))}")
    logger.info(f"  预计每个轴承样本数: 约 {int(config['sampling_rate'] * 1.28 * 123 / config['window_size'])} 个")
    logger.info(f"  训练/验证/测试比例: {(1-config['test_size'])*(1-config['val_split']):.1%} / "
               f"{(1-config['test_size'])*config['val_split']:.1%} / {config['test_size']:.1%}")
    logger.info("=" * 60)
    
    return processor, bearing_folders


def run_model_experiments(processor, bearing_folders, processor_type='standard'):
    """运行模型实验"""
    import logging
    import traceback
    
    logger = logging.getLogger("ExperimentRunner")
    logger.info("开始运行模型实验...")
    
    results = []
    
    for i, bearing_folder in enumerate(bearing_folders, 1):
        logger.info(f"实验进度: {i}/{len(bearing_folders)} - {bearing_folder}")
        
        try:
            if processor_type == 'multimodal':
                # 对于多模态处理器，直接调用 process_single_bearing
                result = processor.process_single_bearing(bearing_folder)
            else:
                # 对于标准处理器，需要调用父类的方法
                result = processor.process_single_bearing(bearing_folder)
                if result:
                    summary_info, detailed_result = result
                    processor.results_summary.append(summary_info)
                    result = detailed_result
            
            if result:
                results.append(result)
            else:
                logger.warning(f"轴承 {bearing_folder} 实验失败，跳过")
                
        except Exception as e:
            logger.error(f"轴承 {bearing_folder} 实验出错: {e}")
            traceback.print_exc()
    
    return results


def generate_reports_and_visualizations(processor, results, processor_type='standard'):
    """生成报告和可视化"""
    import logging
    import pandas as pd
    from pathlib import Path
    
    logger = logging.getLogger("ReportGenerator")
    logger.info("开始生成报告和可视化...")
    
    if not results:
        logger.warning("没有实验结果可汇总")
        return
    
    # 汇总结果
    if processor_type == 'multimodal':
        # 多模态处理器手动汇总
        if results:
            # 创建汇总DataFrame
            summary_data = []
            for result in results:
                if result:
                    summary_info = {
                        'bearing_folder': result.get('bearing_folder', 'unknown'),
                        'bearing_name': result.get('bearing_name', 'unknown'),
                        'total_samples': result.get('data_info', {}).get('total_samples', 0),
                        'test_samples': result.get('data_info', {}).get('test_samples', 0),
                    }
                    
                    # 提取模型性能指标（移除 phm_score）
                    models_results = result.get('models_results', {})
                    for model_name, model_info in models_results.items():
                        if 'clean' in model_info.get('results', {}):
                            metrics = model_info['results']['clean']
                            for metric_name, metric_value in metrics.items():
                                # 【修改】只保留通用指标，移除 phm_score
                                if metric_name in ['r2', 'rmse', 'mae', 'mape', 'hi_r2', 'hi_rmse', 'hi_mae', 'hi_mape']:
                                    summary_info[f'{model_name}_{metric_name}'] = metric_value
                    
                    summary_data.append(summary_info)
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                
                # 保存为CSV
                output_root = Path(processor.config['output_root'])
                summary_csv_path = output_root / "multimodal_batch_results_summary.csv"
                df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
                
                # 保存为Excel
                summary_excel_path = output_root / "multimodal_batch_results_summary.xlsx"
                df_summary.to_excel(summary_excel_path, index=False)
                
                logger.info(f"多模态汇总文件已保存:")
                logger.info(f"  CSV: {summary_csv_path}")
                logger.info(f"  Excel: {summary_excel_path}")
                
                # 打印汇总信息
                logger.info("\n" + "="*80)
                logger.info("多模态批量处理汇总结果")
                logger.info("="*80)
                
                # 打印关键指标（移除 PHM Score）
                for col in df_summary.columns:
                    if 'hi_r2' in col:
                        logger.info(f"{col}: 平均={df_summary[col].mean():.4f}, 标准差={df_summary[col].std():.4f}")
                
                logger.info(f"详细结果已保存到汇总文件")
    else:
        # 标准处理器的汇总
        if hasattr(processor, '_summarize_results'):
            detailed_results = [r for r in results if isinstance(r, dict)]
            processor._summarize_results(detailed_results)
    
    logger.info(f"{processor_type} 实验完成!")


def run_multimodal_batch_simple():
    """简化版多模态批量处理主函数"""
    print("="*80)
    print("简化版多模态轴承剩余寿命预测系统")
    print("="*80)
    
    # 显示设备信息
    show_device_info()
    
    try:
        # 加载配置
        config_path = "multimodal_batch_config.yaml"
        config = BatchConfig(config_path)
        
        # 如果配置文件不存在，使用默认配置
        if not os.path.exists(config_path):
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            config = BatchConfig()
            config['output_root'] = "./multimodal_batch_results"
            config['use_only_35khz'] = True
        
        # 显示配置
        print("\n配置信息:")
        print(f"数据根目录: {config['data_root']}")
        print(f"输出目录: {config['output_root']}")
        print(f"窗口大小: {config['window_size']}")
        print(f"采样率: {config['sampling_rate']}")
        print(f"只使用35kHz工况: {config.get('use_only_35khz', True)}")
        
        # 创建处理器
        logger = setup_logging(log_dir="multimodal_logs")
        processor = EnhancedMultiModalBatchProcessor(config, logger)
        
        # 查找轴承文件夹
        bearing_folders = processor.find_bearing_folders()
        
        if not bearing_folders:
            logger.error("未找到任何轴承文件夹")
            return
        
        logger.info(f"找到 {len(bearing_folders)} 个轴承文件夹")
        
        # 开始处理
        start_time = datetime.now()
        logger.info(f"多模态批量处理开始时间: {start_time}")
        
        # 处理所有轴承
        all_results = []
        
        for i, bearing_folder in enumerate(bearing_folders, 1):
            logger.info(f"处理进度: {i}/{len(bearing_folders)} - {bearing_folder}")
            
            try:
                result = processor.process_single_bearing(bearing_folder)
                if result:
                    all_results.append(result)
                else:
                    logger.warning(f"轴承 {bearing_folder} 处理失败，跳过")
                    
            except Exception as e:
                logger.error(f"处理轴承 {bearing_folder} 时出错: {e}")
                traceback.print_exc()
        
        # 生成汇总报告
        if all_results:
            import pandas as pd
            from pathlib import Path
            
            # 创建汇总DataFrame
            summary_data = []
            for result in all_results:
                if result:
                    summary_info = {
                        'bearing_folder': result.get('bearing_folder', 'unknown'),
                        'bearing_name': result.get('bearing_name', 'unknown'),
                        'total_samples': result.get('data_info', {}).get('total_samples', 0),
                        'test_samples': result.get('data_info', {}).get('test_samples', 0),
                    }
                    
                    # 提取模型性能指标（【修改】移除 phm_score）
                    models_results = result.get('models_results', {})
                    for model_name, model_info in models_results.items():
                        if 'clean' in model_info.get('results', {}):
                            metrics = model_info['results']['clean']
                            for metric_name, metric_value in metrics.items():
                                # 只保留通用指标，移除 phm_score
                                if metric_name in ['r2', 'rmse', 'mae', 'mape', 'hi_r2', 'hi_rmse', 'hi_mae', 'hi_mape']:
                                    summary_info[f'{model_name}_{metric_name}'] = metric_value
                    
                    summary_data.append(summary_info)
            
            df_summary = pd.DataFrame(summary_data)
            
            # 保存为CSV
            output_root = Path(config['output_root'])
            summary_csv_path = output_root / "multimodal_batch_results_summary.csv"
            df_summary.to_csv(summary_csv_path, index=False, encoding='utf-8-sig')
            
            # 保存为Excel
            summary_excel_path = output_root / "multimodal_batch_results_summary.xlsx"
            df_summary.to_excel(summary_excel_path, index=False)
            
            # 打印汇总信息（【修改】移除 PHM Score 相关输出）
            print("\n" + "="*80)
            print("多模态批量处理汇总结果")
            print("="*80)
            
            if not df_summary.empty:
                # 只打印 HI-R² 相关指标
                hi_r2_cols = [col for col in df_summary.columns if 'hi_r2' in col]
                for col in hi_r2_cols:
                    if col in df_summary.columns:
                        r2_mean = df_summary[col].mean()
                        r2_std = df_summary[col].std()
                        print(f"{col}: 平均={r2_mean:.4f}, 标准差={r2_std:.4f}")
                
                # 【修改】移除 PHM Score 打印部分
                # 不再打印 phm_cols
                
                print(f"\n详细结果:")
                for idx, row in df_summary.iterrows():
                    bearing_name = row.get('bearing_name', f'轴承_{idx+1}')
                    hi_r2 = row.get('multimodal_hi_r2', 0)
                    # 【修改】移除 PHM Score 输出
                    print(f"{idx+1}. {bearing_name}: HI-R²={hi_r2:.4f}")
            
            print(f"\n汇总文件已保存:")
            print(f"  CSV: {summary_csv_path}")
            print(f"  Excel: {summary_excel_path}")
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"多模态批量处理结束时间: {end_time}")
        logger.info(f"总处理时间: {duration}")
        
        print("\n" + "="*80)
        print("多模态批量处理完成!")
        print(f"结果保存在: {config['output_root']}")
        print(f"处理时间: {duration}")
        print(f"使用设备: {DEVICE}")
        print("="*80)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()


def main():
    """主函数 - 默认执行多模态批量处理"""
    print("="*80)
    print("多模态轴承剩余寿命预测系统 - 增强版批量处理")
    print("版本: 7.0 - 支持健康因子(HI)联合训练")
    print("="*80)
    
    # 显示设备信息
    show_device_info()
    
    try:
        # 加载配置
        config_path = "multimodal_batch_config.yaml"
        config = BatchConfig(config_path)
        
        # 显示配置
        print("\n配置信息:")
        print(f"数据根目录: {config['data_root']}")
        print(f"输出目录: {config['output_root']}")
        print(f"窗口大小: {config['window_size']}")
        print(f"采样率: {config['sampling_rate']}")
        print(f"只使用35kHz工况: {config.get('use_only_35khz', True)}")
        
        # 【新增】检查是否启用跨轴承评估
        cross_bearing_eval = config.get('cross_bearing_eval', False)
        if cross_bearing_eval:
            print(f"\n【跨轴承评估模式】已启用")
            print(f"  评估模式: {config.get('cross_bearing_mode', 'leave_one_out')}")
        
        # 创建日志记录器
        logger = setup_logging(log_dir="multimodal_logs")
        
        # 保存当前配置
        config_save_path = Path(config['output_root']) / "multimodal_batch_config.yaml"
        config.save(str(config_save_path))
        
        # 开始处理
        start_time = datetime.now()
        logger.info(f"多模态批量处理开始时间: {start_time}")
        
        # 【新增】判断执行模式
        if cross_bearing_eval:
            # 跨轴承评估模式
            from runners import CrossBearingEvaluator
            
            # 创建处理器（仅用于查找文件夹）
            processor = EnhancedMultiModalBatchProcessor(config, logger)
            bearing_folders = processor.find_bearing_folders()
            
            if not bearing_folders:
                logger.error("未找到任何轴承文件夹")
                return
            
            # 创建跨轴承评估器
            evaluator = CrossBearingEvaluator(config, logger)
            
            # 执行跨轴承评估
            results = evaluator.evaluate_cross_bearing(bearing_folders)
            
            logger.info(f"跨轴承评估完成，共评估 {len(results)} 个测试轴承")
            
        else:
            # 标准批量处理模式
            # 1. 加载和处理数据
            processor, bearing_folders = load_and_process_data(config, processor_type='multimodal')
            
            if processor and bearing_folders:
                # 2. 运行模型实验
                results = run_model_experiments(processor, bearing_folders, processor_type='multimodal')
                
                # 3. 生成报告和可视化
                generate_reports_and_visualizations(processor, results, processor_type='multimodal')
                
                # 4. 对处理过的每个轴承生成模型对比图
                if results:
                    logger.info("开始为每个轴承生成模型性能对比图...")
                    for result in results:
                        if result:
                            bearing_name = result.get('bearing_name', '')
                            if bearing_name:
                                logger.info(f"为轴承 {bearing_name} 生成模型对比图...")
                                try:
                                    if hasattr(processor, 'generate_single_bearing_model_comparison'):
                                        comparison_path = processor.generate_single_bearing_model_comparison(bearing_name)
                                        if comparison_path:
                                            logger.info(f"轴承 {bearing_name} 的模型对比图已生成: {comparison_path}")
                                        else:
                                            logger.warning(f"轴承 {bearing_name} 的模型对比图生成失败")
                                    else:
                                        logger.warning("处理器没有 generate_single_bearing_model_comparison 方法")
                                except Exception as e:
                                    logger.error(f"生成轴承 {bearing_name} 的模型对比图时出错: {e}")
                                    traceback.print_exc()
        
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"多模态批量处理结束时间: {end_time}")
        logger.info(f"总处理时间: {duration}")
        
        print("\n" + "="*80)
        print("多模态批量处理完成!")
        print(f"结果保存在: {config['output_root']}")
        print(f"处理时间: {duration}")
        print(f"使用设备: {DEVICE}")
        print("="*80)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()

# ==================== 入口点 ====================
if __name__ == "__main__":
    # 显示欢迎信息
    print("\n" + "="*80)
    print("轴承剩余寿命预测系统 - GPU加速版")
    print("="*80)
    print("选项:")
    print("  1. 运行多模态批量处理 (默认)")
    print("  2. 运行传统特征批量处理")
    print("  3. 测试多模态模型")
    print("  4. 创建配置文件模板")
    print("  5. 显示设备信息")
    print("  6. 运行简化版多模态批量处理")
    print("  7. 创建快速测试配置文件")
    print("  8. 为指定轴承生成模型对比图")
    print("  9. 运行跨轴承评估模式")  # 【新增】跨轴承评估选项
    print("="*80)
    
    try:
        choice = input("请选择 (1-9, 默认1): ").strip()
        
        if choice == '2':
            # 运行传统特征批量处理
            print("运行传统特征批量处理...")
            config = BatchConfig("batch_config.yaml")
            logger = setup_logging()
            processor = EnhancedBatchRULProcessor(config, logger)
            processor.process_all_bearings()
            
        elif choice == '3':
            # 测试多模态模型
            print("测试多模态模型...")
            test_model = MultiModalRULPredictor(
                cwt_image_shape=(1, 64, 64),
                signal_length=1024,
                fusion_method='late'
            )
            print(f"模型创建成功: {type(test_model).__name__}")
            print(f"模型参数量: {test_model.get_parameter_count():,}")
            print("模型架构图绘制功能已简化")
            
        elif choice == '4':
            # 创建配置文件模板
            print("创建配置文件模板...")
            create_enhanced_multimodal_config_template()
            create_basic_config_template()
            
        elif choice == '5':
            # 显示设备信息
            show_device_info()
            
        elif choice == '6':
            # 运行简化版多模态批量处理
            print("运行简化版多模态批量处理...")
            run_multimodal_batch_simple()
            
        elif choice == '7':
            # 创建快速测试配置文件
            print("创建快速测试配置文件...")
            create_enhanced_multimodal_config_template()
            print("快速测试配置文件已创建，请修改 config_templates/enhanced_multimodal_config.yaml")
            
        elif choice == '8':
            # 为指定轴承生成模型对比图
            print("为指定轴承生成模型对比图...")
            
            # 首先显示设备信息
            show_device_info()
            
            # 询问用户输入
            bearing_name = input("请输入轴承名称 (如 '35Hz12kN_Bearing1_1'): ").strip()
            results_dir = input("请输入结果目录 (默认: './multimodal_batch_results'): ").strip()
            
            if not bearing_name:
                print("错误: 必须输入轴承名称")
                exit(1)
                
            if not results_dir:
                results_dir = "./multimodal_batch_results"
            
            # 加载配置
            config = BatchConfig()
            config['output_root'] = results_dir
            
            # 创建处理器
            logger = setup_logging(log_dir="model_comparison_logs")
            processor = EnhancedMultiModalBatchProcessor(config, logger)
            
            # 生成模型对比图
            print(f"\n正在为轴承 {bearing_name} 生成模型对比图...")
            print(f"扫描目录: {results_dir}")
            
            start_time = datetime.now()
            comparison_path = processor.generate_single_bearing_model_comparison(bearing_name, results_dir)
            end_time = datetime.now()
            
            duration = end_time - start_time
            
            if comparison_path:
                print(f"\n✓ 模型对比图生成成功!")
                print(f"图表保存位置: {comparison_path}")
                print(f"处理时间: {duration}")
            else:
                print(f"\n✗ 模型对比图生成失败")
                print("可能原因:")
                print("  1. 未找到轴承对应的模型文件夹")
                print("  2. 结果目录不存在")
                print("  3. 没有足够的模型进行对比")
        
        elif choice == '9':
            # 【新增】运行跨轴承评估模式
            print("运行跨轴承评估模式...")
            print("=" * 80)
            print("跨轴承评估模式说明:")
            print("  - 按工况分组（如35Hz12kN）")
            print("  - 留一法：每个轴承作为测试集，其余作为训练集")
            print("  - 评估模型在未见过的轴承上的泛化能力")
            print("=" * 80)
            
            # 加载配置
            config_path = "multimodal_batch_config.yaml"
            config = BatchConfig(config_path)
            
            # 启用跨轴承评估
            config['cross_bearing_eval'] = True
            
            # 询问用户选择评估模式
            print("\n请选择跨轴承评估模式:")
            print("  1. 留一法 (leave_one_out) - 推荐")
            print("  2. 训练/测试划分 (train_test_split)")
            mode_choice = input("请选择 (1-2, 默认1): ").strip()
            
            if mode_choice == '2':
                config['cross_bearing_mode'] = 'train_test_split'
                train_ratio = input("请输入训练集比例 (0.5-0.9, 默认0.8): ").strip()
                if train_ratio:
                    try:
                        config['cross_bearing_train_ratio'] = float(train_ratio)
                    except:
                        pass
                print(f"  模式: train_test_split, 训练集比例: {config['cross_bearing_train_ratio']}")
            else:
                config['cross_bearing_mode'] = 'leave_one_out'
                print(f"  模式: leave_one_out (留一法)")
            
            # 显示配置
            print("\n配置信息:")
            print(f"数据根目录: {config['data_root']}")
            print(f"输出目录: {config['output_root']}")
            print(f"窗口大小: {config['window_size']}")
            print(f"跨轴承评估: 已启用")
            print(f"评估模式: {config['cross_bearing_mode']}")
            
            # 创建日志记录器
            logger = setup_logging(log_dir="cross_bearing_logs")
            
            # 保存配置
            config_save_path = Path(config['output_root']) / "cross_bearing_config.yaml"
            config.save(str(config_save_path))
            
            # 开始处理
            start_time = datetime.now()
            logger.info(f"跨轴承评估开始时间: {start_time}")
            
            # 创建处理器（用于查找文件夹）
            processor = EnhancedMultiModalBatchProcessor(config, logger)
            bearing_folders = processor.find_bearing_folders()
            
            if not bearing_folders:
                logger.error("未找到任何轴承文件夹")
                exit(1)
            
            # 创建跨轴承评估器
            from runners import CrossBearingEvaluator
            evaluator = CrossBearingEvaluator(config, logger)
            
            # 执行跨轴承评估
            results = evaluator.evaluate_cross_bearing(bearing_folders)
            
            end_time = datetime.now()
            duration = end_time - start_time
            logger.info(f"跨轴承评估结束时间: {end_time}")
            logger.info(f"总处理时间: {duration}")
            
            print("\n" + "="*80)
            print("跨轴承评估完成!")
            print(f"结果保存在: {config['output_root']}/cross_bearing_evaluation")
            print(f"处理时间: {duration}")
            print(f"使用设备: {DEVICE}")
            print("="*80)
            
        else:
            # 默认运行多模态批量处理（标准模式，不启用跨轴承评估）
            print("运行多模态批量处理...")
            # 确保 cross_bearing_eval 为 False
            config = BatchConfig()
            config['cross_bearing_eval'] = False
            main()  # 调用 main() 函数
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序执行出错: {e}")
        traceback.print_exc()
