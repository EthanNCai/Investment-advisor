import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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
        return [],[]

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


def k_chart_fetcher(code_a, code_b,duration_in, degree, threshold_arg):
    # 第一步：拿到需要的两个股票的收盘价list

    with open('../peudo_backend/stock_info_base.json', 'r', encoding='utf-8') as file:
        stock_info_base = json.load(file)

    # 从数据库里面读取 两只股票的dates和close (日期&收盘价)
    close_a_: list = stock_info_base[code_a]['close']
    close_b_: list = stock_info_base[code_b]['close']
    dates_a_: list = stock_info_base[code_a]['dates']
    dates_b_: list = stock_info_base[code_b]['dates']

    # 需要注意的是，两个股票不一定交易日是重叠的，所以我们只取二者交易日的交集最终计算ratio，我将其称之为日期的alignment
    close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)

    duration_days = durations[duration_in]

    if duration_days == -1 or duration_days >= len(close_a):
        duration_days = len(close_a)

    close_a = close_a[:duration_days]
    close_b = close_b[:duration_days]
    dates = dates[:duration_days]

    first_date = dates[0]
    last_date = dates[-1]
    ratio = [float(a) / float(b) for a, b in zip(close_a, close_b)]

    # 第二步：计算拟合曲线
    y = np.array(ratio)
    x = np.arange(len(ratio))
    coefficients = np.polyfit(x, y, int(degree))
    poly = np.poly1d(coefficients)
    fitting_line = poly(x).tolist()

    delta = [r - f for r, f in zip(ratio, fitting_line)]

    # 第三步：计算离群点
    yd = np.array(delta)
    mean = np.mean(yd)
    std_dev = np.std(yd)
    threshold = threshold_arg * std_dev

    # 找到离群点的index

    xo = []
    for i, y in enumerate(yd):
        if y - mean > threshold:
            xo.append((i, True))
        elif y - mean < -threshold:
            xo.append((i, False))

    # 获得离群点的dates
    n_outliers = len(xo)
    outlier_dates = [(dates[i], isPos) for i, isPos in xo]

    if not outlier_dates:
        return {"close_a": close_a,
                "close_b": close_b,
                "dates": dates,
                "ratio": ratio,
                "outlier_date_splitters": [],
                "colors": ['gray'],
                "fitting_line": fitting_line,
                "delta":delta,
                "thres":threshold
                }

    # 这一步很有意思，吧离散的date用区间的概念体现出来，返回的是这些区间的分割点
    outlier_date_splitters, outlier_date_pos_flags = convert_dates_to_splitters(outlier_dates)

    assert len(outlier_date_splitters) == len(outlier_date_pos_flags) * 2
    # print(outlier_date_splitters)
    # print(outlier_date_pos_flags)

    pos_outlier_color = 'blue'
    normal_color = 'black'
    neg_outlier_color = 'red'

    colors = []
    for outlier_date_pos_flag in outlier_date_pos_flags:
        colors.append(normal_color)
        if outlier_date_pos_flag:
            colors.append(pos_outlier_color)
        if not outlier_date_pos_flag:
            colors.append(neg_outlier_color)
    colors.append(normal_color)  # tail color

    # 处理极端情况(outliers区间刚好在始末)
    if outlier_date_splitters[0] == first_date:
        outlier_date_splitters.pop(0)
        colors.pop(0)
    if outlier_date_splitters[-1] == last_date:
        outlier_date_splitters.pop(-1)
        colors.pop(-1)

    assert len(colors) == len(outlier_date_splitters) + 1

    # print(close_a)
    # print(close_b)
    # print(dates)
    # print(ratio)
    # print(outlier_date_splitters)
    # print(colors)

    return {"close_a":close_a,
            "close_b":close_b,
            "dates":dates,
            "ratio":ratio,
            "outlier_date_splitters":outlier_date_splitters,
            "colors":colors,
            "fitting_line":fitting_line,
            "delta":delta,
            "thres":threshold}

    # i = 0
    # while i< max(n_neg_outliers,n_pos_outliers):
    #     pass
    # 可视化时间序列及离群点

    # plt.figure(figsize=(12, 6))
    # plt.plot(x, fitting_line, label='Time Series Data')
    # plt.plot(x, ratio, label='Time Series Data')
    # plt.plot([i if y - mean < -threshold else None for i, y in enumerate(yd)], ratio, 'ro', markersize=8, label='Outliers')
    # plt.plot([i if y - mean > threshold else None for i, y in enumerate(yd)], ratio, 'bo', markersize=8, label='Outliers')
    # plt.xlabel('Time')
    # plt.ylabel('Value')
    # plt.title('Outliers Detection in Time Series Data using 3 Sigma Method')
    # plt.legend()
    # plt.grid(True)
    # plt.show()


# k_chart_fetcher('5y', 2, 'HSI', 'SPX', 1.5)
