import sys
import os
from pathlib import Path

# 将父目录添加到模块搜索路径中
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from k_chart_fetcher import k_chart_fetcher

if __name__ == '__main__':
    print(k_chart_fetcher('002594', '399001', '1y', 2, 1.5))

