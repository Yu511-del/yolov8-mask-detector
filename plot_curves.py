import pandas as pd
import matplotlib.pyplot as plt

# 读取 results.csv
df = pd.read_csv('experiments/scale/scale_m/results.csv')

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 图1：mAP 曲线
ax1.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='blue')
ax1.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='orange')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('mAP')
ax1.set_title('mAP 训练曲线')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：Loss 曲线
ax2.plot(df['epoch'], df['train/box_loss'], label='Box Loss', color='red')
ax2.plot(df['epoch'], df['train/cls_loss'], label='Cls Loss', color='green')
ax2.plot(df['epoch'], df['train/dfl_loss'], label='DFL Loss', color='purple')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('训练损失曲线')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
print('✅ 训练曲线图已生成: training_curves.png')