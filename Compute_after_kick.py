import json
import numpy as np
import os

def calculate_metrics_without_outliers(file_path, outlier_percent=1):
    """
    剔除最大的指定百分比误差后计算MAE和RMSE
    
    Args:
        file_path (str): 误差数据JSON文件路径
        outlier_percent (float): 要剔除的最大误差百分比，默认为1%
        
    Returns:
        dict: 包含原始和剔除异常值后的指标
    """
    # 加载数据
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"文件格式错误: {file_path}")
        return None
    
    # 提取误差信息
    error_distribution = data.get('error_distribution', {})
    errors = error_distribution.get('errors', [])
    
    if not errors:
        print("文件中没有找到误差数据")
        return None
    
    # 转换为numpy数组
    errors_array = np.array(errors)
    
    # 原始指标
    original_mae = np.mean(errors_array)
    original_rmse = np.sqrt(np.mean(errors_array**2))
    
    # 计算要剔除的阈值（最大的outlier_percent%）
    threshold = np.percentile(errors_array, 100 - outlier_percent)
    
    # 剔除大于阈值的误差
    filtered_errors = errors_array[errors_array <= threshold]
    
    # 计算剔除后的指标
    filtered_mae = np.mean(filtered_errors)
    filtered_rmse = np.sqrt(np.mean(filtered_errors**2))
    
    # 统计信息
    total_count = len(errors_array)
    filtered_count = len(filtered_errors)
    outlier_count = total_count - filtered_count
    
    # 获取模型信息
    model_name = error_distribution.get('model_name', 'Unknown')
    split = error_distribution.get('split', 'Unknown')
    
    result = {
        'model_name': model_name,
        'split': split,
        'file_path': file_path,
        'outlier_percent': outlier_percent,
        'outlier_threshold': threshold,
        'total_images': total_count,
        'outlier_count': outlier_count,
        'filtered_count': filtered_count,
        'original': {
            'mae': original_mae,
            'rmse': original_rmse
        },
        'filtered': {
            'mae': filtered_mae,
            'rmse': filtered_rmse
        },
        'improvement': {
            'mae_improvement': (original_mae - filtered_mae) / original_mae * 100,
            'rmse_improvement': (original_rmse - filtered_rmse) / original_rmse * 100
        }
    }
    
    return result

def analyze_multiple_files_without_outliers(file_paths, outlier_percent=1):
    """
    批量分析多个文件，剔除异常值后计算指标
    
    Args:
        file_paths (list): 文件路径列表
        outlier_percent (float): 要剔除的最大误差百分比
        
    Returns:
        dict: 所有文件的分析结果
    """
    results = {}
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            result = calculate_metrics_without_outliers(file_path, outlier_percent)
            if result:
                results[os.path.basename(file_path)] = result
    
    return results

def print_comparison_table(results):
    """
    打印原始指标与剔除异常值后指标的对比表格
    
    Args:
        results (dict): 分析结果
    """
    if not results:
        print("没有可用的结果")
        return
    
    print("\n" + "="*100)
    print(f"{'模型':<20} {'数据集':<10} {'原始MAE':<10} {'剔除后MAE':<12} {'改进%':<8} {'原始RMSE':<10} {'剔除后RMSE':<12} {'改进%':<8} {'异常值数':<10}")
    print("="*100)
    
    for file_name, result in results.items():
        orig = result['original']
        filt = result['filtered']
        impr = result['improvement']
        
        print(f"{result['model_name']:<20} {result['split']:<10} "
              f"{orig['mae']:<10.3f} {filt['mae']:<12.3f} {impr['mae_improvement']:<8.2f} "
              f"{orig['rmse']:<10.3f} {filt['rmse']:<12.3f} {impr['rmse_improvement']:<8.2f} "
              f"{result['outlier_count']:<10}")
    
    print("="*100)

def save_results_to_csv(results, output_file):
    """
    将结果保存到CSV文件
    
    Args:
        results (dict): 分析结果
        output_file (str): 输出CSV文件路径
    """
    import csv
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow([
            'Model', 'Split', 'Outlier_Percent', 'Outlier_Threshold',
            'Total_Images', 'Outlier_Count', 'Filtered_Count',
            'Original_MAE', 'Original_RMSE',
            'Filtered_MAE', 'Filtered_RMSE',
            'MAE_Improvement_Percent', 'RMSE_Improvement_Percent'
        ])
        
        # 写入数据
        for file_name, result in results.items():
            writer.writerow([
                result['model_name'],
                result['split'],
                result['outlier_percent'],
                result['outlier_threshold'],
                result['total_images'],
                result['outlier_count'],
                result['filtered_count'],
                result['original']['mae'],
                result['original']['rmse'],
                result['filtered']['mae'],
                result['filtered']['rmse'],
                result['improvement']['mae_improvement'],
                result['improvement']['rmse_improvement']
            ])
    
    print(f"结果已保存到CSV文件: {output_file}")

def plot_comparison(results, save_path=None):
    """
    绘制原始指标与剔除异常值后指标的对比图
    
    Args:
        results (dict): 分析结果
        save_path (str, optional): 图片保存路径
    """
    import matplotlib.pyplot as plt
    
    if not results:
        print("没有可用的结果")
        return
    
    # 准备数据
    labels = []
    original_mae = []
    filtered_mae = []
    original_rmse = []
    filtered_rmse = []
    
    for file_name, result in results.items():
        labels.append(f"{result['model_name']}\n({result['split']})")
        original_mae.append(result['original']['mae'])
        filtered_mae.append(result['filtered']['mae'])
        original_rmse.append(result['original']['rmse'])
        filtered_rmse.append(result['filtered']['rmse'])
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # MAE对比
    x = np.arange(len(labels))
    width = 0.35
    
    ax1.bar(x - width/2, original_mae, width, label='Original MAE', color='lightcoral', alpha=0.7)
    ax1.bar(x + width/2, filtered_mae, width, label='Modified MAE', color='lightgreen', alpha=0.7)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('MAE')
    ax1.set_title('MAEComparison (remove the top 1%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (orig, filt) in enumerate(zip(original_mae, filtered_mae)):
        ax1.text(i - width/2, orig + max(original_mae)*0.01, f'{orig:.2f}', ha='center', va='bottom', fontsize=9)
        ax1.text(i + width/2, filt + max(filtered_mae)*0.01, f'{filt:.2f}', ha='center', va='bottom', fontsize=9)
    
    # RMSE对比
    ax2.bar(x - width/2, original_rmse, width, label='Original RMSE', color='lightcoral', alpha=0.7)
    ax2.bar(x + width/2, filtered_rmse, width, label='Modified RMSE', color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE Comparison (remove the top 1%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (orig, filt) in enumerate(zip(original_rmse, filtered_rmse)):
        ax2.text(i - width/2, orig + max(original_rmse)*0.01, f'{orig:.2f}', ha='center', va='bottom', fontsize=9)
        ax2.text(i + width/2, filt + max(filtered_rmse)*0.01, f'{filt:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 示例文件路径 - 替换为实际路径
    file_paths = [
        "GeCo1_val_count_errors.json",
        "GeCo1_test_count_errors.json",
        "GeCo_best_val_count_errors.json",
        "GeCo_best_test_count_errors.json"
    ]
    
    # 只处理存在的文件
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if not existing_files:
        print("未找到任何误差文件，请检查文件路径")
        print("当前目录中的JSON文件:")
        for file in os.listdir('.'):
            if file.endswith('.json'):
                print(f"  - {file}")
    else:
        print(f"找到 {len(existing_files)} 个误差文件:")
        for file in existing_files:
            print(f"  - {file}")
        
        # 分析剔除最大1%误差后的指标
        results = analyze_multiple_files_without_outliers(existing_files, outlier_percent=1)
        
        # 打印详细结果
        print("\n=== 剔除最大1%误差后的指标对比 ===")
        for file_name, result in results.items():
            print(f"\n{file_name}:")
            print(f"  模型: {result['model_name']}")
            print(f"  数据集: {result['split']}")
            print(f"  总图像数: {result['total_images']}")
            print(f"  剔除的异常值数量: {result['outlier_count']} (阈值: {result['outlier_threshold']:.2f})")
            print(f"  原始 MAE: {result['original']['mae']:.3f}, RMSE: {result['original']['rmse']:.3f}")
            print(f"  剔除后 MAE: {result['filtered']['mae']:.3f}, RMSE: {result['filtered']['rmse']:.3f}")
            print(f"  MAE 改进: {result['improvement']['mae_improvement']:.2f}%")
            print(f"  RMSE 改进: {result['improvement']['rmse_improvement']:.2f}%")
        
        # 打印对比表格
        print_comparison_table(results)
        
        # 保存结果到CSV
        save_results_to_csv(results, "metrics_without_outliers.csv")
        
        # 绘制对比图
        plot_comparison(results, save_path="metrics_comparison.png")
        
        # 可以尝试不同的异常值剔除比例
        print("\n=== 不同异常值剔除比例的比较 ===")
        for percent in [0.5, 1, 2, 5]:
            print(f"\n剔除最大 {percent}% 误差:")
            temp_results = analyze_multiple_files_without_outliers(existing_files[:1], outlier_percent=percent)
            if temp_results:
                for file_name, result in temp_results.items():
                    print(f"  {file_name}:")
                    print(f"    原始 MAE: {result['original']['mae']:.3f}")
                    print(f"    剔除后 MAE: {result['filtered']['mae']:.3f}")
                    print(f"    改进: {result['improvement']['mae_improvement']:.2f}%")