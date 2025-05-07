import os
import sys

# 添加项目根目录到python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
sys.path.append(project_root)
# 添加back_end/peudo_backend目录到python路径
peudo_backend_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(peudo_backend_dir)


# 现在使用相对于项目根目录或peudo_backend目录的导入路径


if __name__ == '__main__':
    # close_a_, close_b_, dates_a_, dates_b_ = get_stock_data_pair('000001', '000902')
    # close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)
    # print(f"close_a: {close_a}")
    # print(f"close_b: {close_b}")
    # print(f"dates: {dates}")
    # db = StockKlineDatabase()
    # kline_a = db.query_kline('000001', '2015-01-01', datetime.now().strftime('%Y-%m-%d'))
    # kline_b = db.query_kline('000902', '2015-01-01', datetime.now().strftime('%Y-%m-%d'))
    # aligned_data = align_kline_dates(kline_a, kline_b)
    # dates, close_a, close_b = aligned_data
    # print(f"dates: {dates}")
    # print(f"close_a: {close_a}")
    # print(f"close_b: {close_b}")
    pass
