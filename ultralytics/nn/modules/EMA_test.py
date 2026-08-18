# alpha_comparison_experiment.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
from datetime import datetime
from ultralytics import YOLO
import warnings
import sys
import os

warnings.filterwarnings('ignore')


class AlphaComparison:
    """
    完整的Alpha值比较实验
    通过修改损失函数文件中的α值来比较不同α值的性能
    """

    def __init__(self, model_path, data_yaml,
                 alpha_values=[2.5, 3.0, 3.5],
                 save_dir="runs/alpha_comparison",
                 epochs=30, batch_size=16, imgsz=640):
        """
        初始化实验

        Args:
            model_path: 预训练模型路径
            data_yaml: 数据集配置文件路径
            alpha_values: 要比较的α值列表
            save_dir: 结果保存目录
        """
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.alpha_values = alpha_values
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 实验配置
        self.config = {
            'epochs': epochs,
            'batch_size': batch_size,
            'imgsz': imgsz,
            'patience': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        }

        # 存储结果
        self.results = {}

        # 备份原始损失函数文件
        self.backup_loss_file()

        print(f"\n{'=' * 60}")
        print(f"Alpha值比较实验")
        print(f"{'=' * 60}")
        print(f"模型: {model_path}")
        print(f"数据集: {data_yaml}")
        print(f"比较的α值: {alpha_values}")
        print(f"结果保存目录: {self.save_dir}")
        print(f"{'=' * 60}")

    def backup_loss_file(self):
        """备份原始的损失函数文件"""
        # 查找损失函数文件
        possible_paths = [
            "E:/improvedyolo/ultralytics/utils/loss.py",  # 你的路径
            "ultralytics/utils/loss.py",
            "loss.py",
        ]

        self.loss_file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.loss_file_path = Path(path)
                break

        if self.loss_file_path:
            # 创建备份
            backup_path = self.save_dir / "original_loss_backup.py"
            shutil.copy2(self.loss_file_path, backup_path)
            print(f"✓ 已备份损失函数文件: {backup_path}")
        else:
            print("✗ 未找到损失函数文件，请手动指定路径")
            self.loss_file_path = input("请输入损失函数文件路径: ")

    def modify_alpha_in_loss_file(self, alpha):
        """
        修改损失函数文件中的α值

        Args:
            alpha: 要设置的α值
        """
        if not self.loss_file_path or not os.path.exists(self.loss_file_path):
            print("✗ 损失函数文件不存在")
            return False

        try:
            # 读取文件内容
            with open(self.loss_file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 查找并替换α值
            # 查找 alpha= 后面跟着数字的模式
            import re

            # 修改BboxLoss类中的alpha默认值
            pattern1 = r'def __init__\(self, reg_max=.*?, alpha=([0-9.]+)\)'
            replacement1 = f'def __init__(self, reg_max=16, alpha={alpha})'
            content = re.sub(pattern1, replacement1, content)

            # 修改bbox_alpha_iou函数中的alpha默认值
            pattern2 = r'def bbox_alpha_iou\(.*?, alpha=([0-9.]+), '
            replacement2 = f'def bbox_alpha_iou(box1, box2, xywh=True, alpha={alpha}, CIoU=False, DIoU=True, eps=1e-7):'
            content = re.sub(pattern2, replacement2, content)

            # 写入修改后的内容
            with open(self.loss_file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✓ 已将损失函数中的α值修改为 {alpha}")
            return True

        except Exception as e:
            print(f"✗ 修改损失函数失败: {e}")
            return False

    def restore_loss_file(self):
        """恢复原始的损失函数文件"""
        if not self.loss_file_path:
            return

        backup_path = self.save_dir / "original_loss_backup.py"
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, self.loss_file_path)
            print("✓ 已恢复原始损失函数文件")

    def train_single_alpha(self, alpha):
        """
        训练单个α值的模型

        Args:
            alpha: α值

        Returns:
            metrics: 评估指标
        """
        print(f"\n开始训练 α={alpha} ...")

        # 修改损失函数中的α值
        if not self.modify_alpha_in_loss_file(alpha):
            return None

        try:
            # 创建保存目录
            alpha_dir = self.save_dir / f"alpha_{alpha:.1f}"
            alpha_dir.mkdir(exist_ok=True)

            # 加载模型
            model = YOLO(self.model_path)

            # 训练参数
            train_args = {
                'data': self.data_yaml,
                'epochs': self.config['epochs'],
                'imgsz': self.config['imgsz'],
                'batch': self.config['batch_size'],
                'project': str(self.save_dir),
                'name': f"alpha_{alpha:.1f}",
                'exist_ok': True,
                'save': True,
                'val': True,
                'plots': True,
                'device': self.config['device'],
                'patience': self.config['patience'],
                'verbose': True,
                'amp': True,
                'workers': 4,
            }

            # 训练模型
            print(f"训练中...")
            train_results = model.train(**train_args)

            # 验证模型
            print(f"评估中...")
            val_results = model.val(data=self.data_yaml, split='val')

            # 提取指标
            metrics = {
                'alpha': alpha,
                'map50_95': val_results.box.map,  # mAP50-95
                'map50': val_results.box.map50,  # mAP50
                'precision': val_results.box.p,
                'recall': val_results.box.r,
                'save_dir': str(alpha_dir),
                'timestamp': datetime.now().isoformat(),
            }

            # 计算F1分数
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                  (metrics['precision'] + metrics['recall'] + 1e-9)

            # 保存结果
            self.save_alpha_results(alpha, metrics, alpha_dir)

            print(f"✓ α={alpha} 训练完成")
            print(f"  mAP50-95: {metrics['map50_95']:.4f}")
            print(f"  mAP50: {metrics['map50']:.4f}")

            return metrics

        except Exception as e:
            print(f"✗ α={alpha} 训练失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_alpha_results(self, alpha, metrics, save_dir):
        """保存单个α值的结果"""
        # 保存为JSON
        json_path = save_dir / "results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        # 保存为CSV
        df = pd.DataFrame([metrics])
        csv_path = save_dir / "results.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')

    def run_experiment(self):
        """运行完整的实验"""
        print("\n开始Alpha值比较实验...")

        for i, alpha in enumerate(self.alpha_values):
            print(f"\n{'=' * 50}")
            print(f"实验 {i + 1}/{len(self.alpha_values)}: α = {alpha}")
            print(f"{'=' * 50}")

            # 训练并评估
            metrics = self.train_single_alpha(alpha)

            if metrics:
                self.results[alpha] = metrics
            else:
                self.results[alpha] = {'error': '训练失败'}

        # 恢复原始损失函数文件
        self.restore_loss_file()

        # 保存和分析结果
        summary_df = self.save_summary_results()

        if summary_df is not None and not summary_df.empty:
            self.analyze_results(summary_df)
            self.create_visualization(summary_df)

        print(f"\n{'=' * 60}")
        print(f"实验完成！")
        print(f"详细结果保存在: {self.save_dir}")
        print(f"{'=' * 60}")

        return self.results

    def save_summary_results(self):
        """保存所有α值的汇总结果"""
        # 收集所有有效结果
        summary_data = []

        for alpha, result in self.results.items():
            if 'error' in result:
                continue

            if all(k in result for k in ['map50_95', 'map50']):
                summary_data.append({
                    'alpha': alpha,
                    'mAP50-95': result['map50_95'],
                    'mAP50': result['map50'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'save_dir': result['save_dir']
                })

        if summary_data:
            # 创建DataFrame
            df = pd.DataFrame(summary_data)
            df = df.sort_values('alpha')

            # 保存为CSV
            csv_path = self.save_dir / 'alpha_comparison_summary.csv'
            df.to_csv(csv_path, index=False, float_format='%.4f', encoding='utf-8')

            # 保存为JSON
            json_path = self.save_dir / 'alpha_comparison_summary.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(df.to_dict('records'), f, indent=2, ensure_ascii=False)

            print(f"\n结果汇总已保存:")
            print(f"  CSV: {csv_path}")
            print(f"  JSON: {json_path}")

            return df

        print("\n没有有效的结果数据")
        return None

    def analyze_results(self, df):
        """分析比较结果"""
        print(f"\n{'=' * 60}")
        print(f"Alpha值比较分析")
        print(f"{'=' * 60}")

        # 找到最佳α值
        best_idx = df['mAP50-95'].idxmax()
        best_alpha = df.loc[best_idx, 'alpha']
        best_map50_95 = df.loc[best_idx, 'mAP50-95']
        best_map50 = df.loc[best_idx, 'mAP50']

        print(f"\n1. 最佳α值分析:")
        print(f"   最佳α值: {best_alpha:.2f}")
        print(f"   最佳mAP50-95: {best_map50_95:.4f}")
        print(f"   最佳mAP50: {best_map50:.4f}")

        # 比较所有α值
        print(f"\n2. 所有α值性能:")
        for _, row in df.iterrows():
            marker = "★" if row['alpha'] == best_alpha else " "
            print(f"   {marker} α={row['alpha']:.2f}: "
                  f"mAP50-95={row['mAP50-95']:.4f}, "
                  f"mAP50={row['mAP50']:.4f}")

        # 性能提升分析
        print(f"\n3. 性能提升分析:")
        for i in range(len(df) - 1):
            alpha1 = df.loc[i, 'alpha']
            alpha2 = df.loc[i + 1, 'alpha']
            map1 = df.loc[i, 'mAP50-95']
            map2 = df.loc[i + 1, 'mAP50-95']

            if map1 > 0:
                improvement = ((map2 - map1) / map1) * 100
                direction = "提升" if improvement > 0 else "下降"
                print(f"   α={alpha1:.2f} → α={alpha2:.2f}: "
                      f"{abs(improvement):.2f}% {direction}")

        print(f"\n4. 推荐:")
        print(f"   ✓ 建议使用 α={best_alpha:.2f}")
        print(f"   ✓ 可获得最佳性能: mAP50-95={best_map50_95:.4f}")
        print(f"{'=' * 60}")

        # 保存分析结果
        analysis = {
            'best_alpha': float(best_alpha),
            'best_map50_95': float(best_map50_95),
            'best_map50': float(best_map50),
            'analysis_time': datetime.now().isoformat(),
            'all_results': df.to_dict('records')
        }

        analysis_path = self.save_dir / 'analysis_results.json'
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

        return analysis

    def create_visualization(self, df):
        """创建可视化图表"""
        if df.empty:
            print("没有有效数据，无法创建可视化图表")
            return

        # 创建图表
        plt.figure(figsize=(14, 10))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 子图1: α值 vs mAP50-95
        plt.subplot(2, 2, 1)
        plt.plot(df['alpha'], df['mAP50-95'], 'bo-', linewidth=2, markersize=10)

        # 标记最佳点
        best_idx = df['mAP50-95'].idxmax()
        plt.scatter(df.loc[best_idx, 'alpha'], df.loc[best_idx, 'mAP50-95'],
                    color='red', s=200, zorder=5,
                    label=f'最佳 α={df.loc[best_idx, "alpha"]:.2f}')

        plt.xlabel('Alpha值')
        plt.ylabel('mAP50-95')
        plt.title('Alpha值 vs mAP50-95')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 添加数值标签
        for _, row in df.iterrows():
            plt.annotate(f"{row['mAP50-95']:.3f}",
                         (row['alpha'], row['mAP50-95']),
                         textcoords="offset points", xytext=(0, 10),
                         ha='center', fontsize=10)

        # 子图2: α值 vs mAP50
        plt.subplot(2, 2, 2)
        bars = plt.bar([f"α={a:.1f}" for a in df['alpha']], df['mAP50'],
                       color=['green' if i == best_idx else 'skyblue'
                              for i in range(len(df))])

        plt.xlabel('Alpha值')
        plt.ylabel('mAP50')
        plt.title('Alpha值 vs mAP50')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        # 在柱子上添加数值
        for bar, value in zip(bars, df['mAP50']):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=10)

        # 子图3: 精确率-召回率
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(df['precision'], df['recall'],
                              s=df['mAP50-95'] * 500,
                              c=df['alpha'], cmap='viridis',
                              edgecolors='black', alpha=0.8)

        # 添加α值标签
        for i, row in df.iterrows():
            plt.annotate(f"α={row['alpha']:.1f}",
                         (row['precision'], row['recall']),
                         textcoords="offset points", xytext=(5, 5),
                         ha='left', fontsize=10)

        plt.xlabel('精确率 (Precision)')
        plt.ylabel('召回率 (Recall)')
        plt.title('精确率-召回率分布')
        plt.grid(True, alpha=0.3)
        plt.colorbar(scatter, label='Alpha值')

        # 子图4: 雷达图对比
        plt.subplot(2, 2, 4, projection='polar')

        # 归一化指标
        metrics = ['mAP50-95', 'mAP50', 'precision', 'recall', 'f1_score']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        for i, alpha_val in enumerate(df['alpha']):
            values = []
            for metric in metrics:
                if metric in df.columns:
                    # 归一化到0-1
                    col_values = df[metric].values
                    norm_value = (col_values[i] - col_values.min()) / (col_values.max() - col_values.min() + 1e-9)
                    values.append(norm_value)

            values += values[:1]  # 闭合
            plt.plot(angles, values, 'o-', linewidth=2, label=f'α={alpha_val:.1f}')
            plt.fill(angles, values, alpha=0.1)

        plt.xticks(angles[:-1], metrics)
        plt.title('归一化指标雷达图')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        # 保存图表
        plot_path = self.save_dir / 'alpha_comparison_visualization.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n可视化图表已保存: {plot_path}")


# 主函数
def main():
    """主函数"""
    print("YOLO Alpha值比较实验")
    print("=" * 60)

    # 用户输入参数
    model_path = input("请输入模型路径 (默认: yolov8n.pt): ") or "yolov8n.pt"
    data_yaml = input("请输入数据集配置文件路径 (默认: data.yaml): ") or "data.yaml"

    alpha_input = input("请输入要比较的α值，用逗号分隔 (默认: 2.5,3.0): ") or "2.5,3.0"
    alpha_values = [float(x.strip()) for x in alpha_input.split(',')]

    save_dir = input("请输入结果保存目录 (默认: runs/alpha_comparison): ") or "runs/alpha_comparison"

    epochs = int(input("请输入训练轮数 (默认: 30): ") or "30")
    batch_size = int(input("请输入批次大小 (默认: 16): ") or "16")
    imgsz = int(input("请输入图像尺寸 (默认: 640): ") or "640")

    print("\n实验配置:")
    print(f"  模型: {model_path}")
    print(f"  数据集: {data_yaml}")
    print(f"  α值: {alpha_values}")
    print(f"  保存目录: {save_dir}")
    print(f"  训练轮数: {epochs}")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {imgsz}")

    confirm = input("\n是否开始实验? (y/n): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return

    # 创建并运行实验
    experiment = AlphaComparison(
        model_path=model_path,
        data_yaml=data_yaml,
        alpha_values=alpha_values,
        save_dir=save_dir,
        epochs=epochs,
        batch_size=batch_size,
        imgsz=imgsz
    )

    # 运行实验
    results = experiment.run_experiment()

    print("\n" + "=" * 60)
    print("实验总结:")
    print("=" * 60)

    # 显示最终结果
    summary_path = Path(save_dir) / 'alpha_comparison_summary.csv'
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty:
            print("\n最终结果:")
            print(df.to_string(index=False))

            best_idx = df['mAP50-95'].idxmax()
            print(f"\n推荐使用: α={df.loc[best_idx, 'alpha']:.2f}")
            print(f"可获得 mAP50-95: {df.loc[best_idx, 'mAP50-95']:.4f}")

    print(f"\n所有结果保存在: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()