import os
import sys

from back_end.peudo_backend.get_stock_data.get_stock_data_A_and_G import EastMoneyKLineSpider
from back_end.peudo_backend.get_stock_data.stock_data_base import StockKlineDatabase

# # 添加项目根目录到python路径
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
# sys.path.append(project_root)
# # 添加back_end/peudo_backend目录到python路径
# peudo_backend_dir = os.path.abspath(os.path.join(current_dir, "../"))
# sys.path.append(peudo_backend_dir)


# 现在使用相对于项目根目录或peudo_backend目录的导入路径


if __name__ == '__main__':
    db = StockKlineDatabase()
    EastMoneyKLineSpider = EastMoneyKLineSpider('SPX')
    east_money_kline_data = EastMoneyKLineSpider.get_klines()
    db.insert_kline_data(EastMoneyKLineSpider.format_klines(east_money_kline_data['klines']))
    pass
