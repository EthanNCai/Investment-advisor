import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from get_stock_data.stock_data_base import StockKlineDatabase

durations = {'maximum': -1, '5y': 1825, '2y': 730, '1y': 365, '1q': 120, '1m': 30}


def compare_dates(a, b):
    date_format = "%Y-%m-%d"

    date_a = datetime.strptime(a, date_format)
    date_b = datetime.strptime(b, date_format)

    return date_a > date_b


def date_alignment(close_a_, close_b_, dates_a_, dates_b_):
    assert len(close_a_) == len(dates_a_)
    assert len(dates_b_) == len(close_b_)

    date_intersection = set(dates_a_).intersection(set(dates_b_))

    close_a = []
    close_b = []
    dates = []

    for i in range(len(dates_a_)):
        if dates_a_[i] in date_intersection:
            dates.append(dates_a_[i])
            close_a.append(close_a_[i])
    for i in range(len(dates_b_)):
        if dates_b_[i] in date_intersection:
            close_b.append(close_b_[i])

    return close_a, close_b, dates


def convert_dates_to_splitters(date_list) -> (list, list):
    if not date_list:
        return [], []

    # 将字符串日期转换为datetime对象并排序
    dates = [datetime.strptime(date, "%Y-%m-%d") for date, _ in date_list]
    raw_flag = [isPos for _, isPos in date_list]

    is_range_pos_flag = []
    ranges = []
    start_date = dates[0]
    prev_date = start_date

    for i, current_date in enumerate(dates[1:]):
        if (current_date - prev_date).days > 1:
            end_date = prev_date + timedelta(days=1)
            ranges.append(start_date.strftime("%Y-%m-%d"))
            ranges.append(end_date.strftime("%Y-%m-%d"))
            is_range_pos_flag.append(raw_flag[i + 1])
            start_date = current_date
        prev_date = current_date

    end_date = dates[-1] + timedelta(days=1)
    ranges.append(start_date.strftime("%Y-%m-%d"))
    ranges.append(end_date.strftime("%Y-%m-%d"))
    is_range_pos_flag.append(raw_flag[-1])

    return ranges, is_range_pos_flag


def get_stock_data_pair(code_a: str, code_b: str) -> tuple:
    """
    从数据库获取两只股票的日期和收盘价
    返回格式：(dates_a, close_a, dates_b, close_b)
    """

    def fetch_single_stock(code: str) -> tuple:
        """获取单只股票数据，返回 (日期列表, 收盘价列表)"""
        db = StockKlineDatabase()
        # 查询全部历史数据
        records = db.query_kline(
            stock_code=code,
            start_date='2015-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d')
        )

        if not records:
            raise ValueError(f"股票 {code} 无可用数据")

        # 按日期升序排序
        sorted_records = sorted(records, key=lambda x: x[1])

        # 拆分成日期和收盘价列表
        dates = [record[1] for record in sorted_records]  # date字段位置
        closes = [float(record[3]) for record in sorted_records]  # close字段位置
        if len(dates) != len(closes):
            raise ValueError(f"股票 {code} 数据异常：日期与收盘价数量不匹配")

        if len(dates) == 0:
            raise ValueError(f"股票 {code} 无有效数据")
        return dates, closes

    # 获取两只股票数据
    dates_a, close_a = fetch_single_stock(code_a)
    dates_b, close_b = fetch_single_stock(code_b)

    return close_a, close_b, dates_a, dates_b


def k_chart_fetcher(code_a, code_b, duration_in, degree, threshold_arg=2.0):
    """
    获取K线图数据，计算两只股票的价格比值和异常检测
    
    Args:
        code_a: 股票A代码
        code_b: 股票B代码
        duration_in: 时间跨度
        degree: 多项式拟合的次数
        threshold_arg: 异常检测阈值系数，默认为2.0
        
    Returns:
        Dict: 包含K线图数据的字典
    """
    # 第一步：拿到需要的两个股票的收盘价list
    close_a_, close_b_, dates_a_, dates_b_ = get_stock_data_pair(code_a, code_b)

    # 需要注意的是，两个股票不一定交易日是重叠的，所以我们只取二者交易日的交集最终计算ratio，我将其称之为日期的alignment
    close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)

    assert len(close_a) == len(close_b)
    assert len(close_a) == len(dates)
    duration_days = durations[duration_in]
    if duration_days == -1 or duration_days >= len(close_a):
        duration_days = len(close_a)

    close_a = close_a[-duration_days:]
    close_b = close_b[-duration_days:]
    dates = dates[-duration_days:]

    ratio = [float(a) / float(b) for a, b in zip(close_a, close_b)]

    # 第二步：计算拟合曲线
    y = np.array(ratio)
    x = np.arange(len(ratio))
    coefficients = np.polyfit(x, y, int(degree))
    poly = np.poly1d(coefficients)

    fitting_line = poly(x).tolist()
    delta = [r - f for r, f in zip(ratio, fitting_line)]

    # 第三步：计算标准差
    yd = np.array(delta)
    std_dev = np.std(yd)
    # 将前端传入的阈值系数应用于标准差
    # 这里我们不直接修改std_dev，而是将它作为基准值返回
    # 让前端和其他函数根据需要使用这个基准值和阈值系数进行计算
    threshold = std_dev

    return {
        "close_a": close_a,
        "close_b": close_b,
        "dates": dates,
        "ratio": ratio,
        "fitting_line": fitting_line,
        "delta": delta,
        "threshold": threshold  # 原始标准差
    }
