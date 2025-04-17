
import sys
import os
import time

from back_end.peudo_backend.get_stock_data.get_stock_data_A_and_G import EastMoneyKLineSpider
from back_end.peudo_backend.get_stock_data.stock_data_base import StockKlineDatabase

# 添加项目根目录到python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)

if __name__ == '__main__':
    splider = EastMoneyKLineSpider('AUTD')
    data = splider.get_klines()
    data_list = splider.format_klines(data['klines'])
    db = StockKlineDatabase()
    db.insert_kline_data(data_list, data['code'], data['name'], data['type'])
    result = db.query_kline(
        stock_code='AUTD',
        start_date='2023-07-10',
        end_date='2025-04-13'
    )
    print(result)


