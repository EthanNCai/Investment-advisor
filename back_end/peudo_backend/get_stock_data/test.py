
import os
import sys

from back_end.peudo_backend.get_stock_data.get_stock_trends_data import StockTrendsData

# 添加项目根目录到python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

if __name__ == '__main__':
    data_trends = StockTrendsData('NYAuTN06')
    print(data_trends.get_trends())


