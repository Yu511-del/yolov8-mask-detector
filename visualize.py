from ultralytics import YOLO
from pathlib import Path

# 加载模型
model = YOLO('experiments/scale/scale_m/weights/best.pt')

# 创建输出目录
output_dir = Path('visualization_results')
output_dir.mkdir(exist_ok=True)

# 对 test_images 里的所有图片做检测
image_dir = Path('test_images')
image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpeg'))

if not image_files:
    print('❌ 没找到图片！请把图片放到 test_images 文件夹')
else:
    for img_path in image_files:
        print(f'正在处理: {img_path.name}')
        results = model(str(img_path))
        results[0].save(str(output_dir / img_path.name))
    
    print(f'✅ 完成！结果在 {output_dir} 文件夹')