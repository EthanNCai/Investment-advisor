from typing import List, Dict, Any
from datetime import datetime


def process_kline_by_type(records, kline_type: str) -> List[Dict[str, Any]]:
    """
    根据K线类型处理原始K线数据
    
    参数:
        records: 数据库查询结果
        kline_type: K线类型 ("daily", "weekly", "monthly", "yearly")
    
    返回:
        处理后的K线数据列表
    """
    # 按日期排序
    sorted_records = sorted(records, key=lambda x: x[1])

    # 转换为字典格式
    daily_klines = []
    for record in sorted_records:
        kline = {
            "date": record[1],
            "open": float(record[2]),
            "close": float(record[3]),
            "high": float(record[4]),
            "low": float(record[5]),
            "volume": int(record[6]),
            "amount": float(record[7]) if len(record) > 7 else 0.0,
        }
        daily_klines.append(kline)

    # 如果是日K，直接返回
    if kline_type == "daily":
        return daily_klines

    # 处理周K、月K、年K
    if kline_type == "weekly":
        return aggregate_klines(daily_klines, '%Y-%U')  # 按年和周聚合
    elif kline_type == "monthly":
        return aggregate_klines(daily_klines, '%Y-%m')  # 按年和月聚合
    elif kline_type == "yearly":
        return aggregate_klines(daily_klines, '%Y')  # 按年聚合

    # 默认返回日K
    return daily_klines


def aggregate_klines(daily_klines: List[Dict[str, Any]], date_format: str) -> List[Dict[str, Any]]:
    """
    聚合K线数据
    
    参数:
        daily_klines: 日K数据
        date_format: 日期格式，用于分组
    
    返回:
        聚合后的K线数据
    """
    if not daily_klines:
        return []

    # 按日期格式分组
    groups = {}
    for kline in daily_klines:
        date_obj = datetime.strptime(kline['date'], '%Y-%m-%d')
        period_key = date_obj.strftime(date_format)

        if period_key not in groups:
            groups[period_key] = []

        groups[period_key].append(kline)

    # 聚合每个分组的数据
    aggregated_klines = []
    for period_key, klines in sorted(groups.items()):
        if not klines:
            continue

        # 该周期的第一个交易日开盘价
        open_price = klines[0]['open']

        # 该周期的最后一个交易日收盘价
        close_price = klines[-1]['close']

        # 该周期的最高价和最低价
        high_price = max(k['high'] for k in klines)
        low_price = min(k['low'] for k in klines)

        # 该周期的总成交量和成交额
        total_volume = sum(k['volume'] for k in klines)
        total_amount = sum(k['amount'] for k in klines)

        # 使用该周期的第一个交易日作为日期
        date = klines[0]['date']

        aggregated_klines.append({
            "date": date,
            "open": open_price,
            "close": close_price,
            "high": high_price,
            "low": low_price,
            "volume": total_volume,
            "amount": total_amount
        })

    return aggregated_klines
