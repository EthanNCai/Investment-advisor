import sys
import os
from pathlib import Path

from back_end.peudo_backend.get_stock_data.get_stock_trends_data import StockTrendsData

# 将父目录添加到模块搜索路径中
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from k_chart_fetcher import k_chart_fetcher

if __name__ == '__main__':
    trends_data = StockTrendsData('01810')
    print(trends_data.get_trends())
    data = trends_data.get_trends()
    print(data['name'])
    print(trends_data.format_klines(data['trends']))

    # print(k_chart_fetcher('002594', '399001', '1y', 2, 1.5))

