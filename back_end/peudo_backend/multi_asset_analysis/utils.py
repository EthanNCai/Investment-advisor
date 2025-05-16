import json
from typing import List, Dict, Tuple, Any, Union
import os
import numpy as np
from datetime import datetime


def load_stock_list() -> List[Dict[str, str]]:
    """
    加载股票列表
    
    Returns:
        股票信息列表
    """
    stock_list_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_list.json')
    try:
        with open(stock_list_path, 'r', encoding='utf-8') as file:
            stock_info_json = json.load(file)
            return stock_info_json['stocks']
    except Exception as e:
        print(f"加载股票列表失败: {e}")
        return []


def get_stock_names(assets: List[str]) -> List[str]:
    """
    从股票列表文件中获取股票名称
    
    Args:
        assets: 股票代码列表
        
    Returns:
        股票名称列表
    """
    try:
        with open('stock_list.json', 'r', encoding='utf-8') as file:
            stock_info_json = json.load(file)
            stock_info_list = stock_info_json['stocks']

            # 构建代码到名称的映射
            code_to_name = {}
            for stock in stock_info_list:
                code_to_name[stock['code']] = stock['name']

            # 返回资产列表中每个代码对应的名称
            return [code_to_name.get(code, code) for code in assets]
    except Exception as e:
        print(f"读取股票名称时出错: {e}")
        return assets  # 如果发生错误，直接返回原始代码作为名称


def process_kline_by_type(
        data: Union[List[float], List[List[float]], np.ndarray],
        dates: List[str],
        kline_type: str
) -> Tuple[List[float], List[str]]:
    """
    根据K线类型处理价格数据和日期
    
    Args:
        data: 原始价格数据或价格比值数据
        dates: 对应的日期列表
        kline_type: K线类型 ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        处理后的(数据, 日期)元组
    """
    if len(data) == 0 or len(dates) == 0:
        return [], []

    # 确保数据和日期长度相同
    if len(data) != len(dates):
        raise ValueError("数据和日期长度不匹配")

    # 日K不需要处理
    if kline_type == 'daily':
        return data, dates

    # 解析日期，用于周、月、年K线的聚合
    parsed_dates = [datetime.strptime(date, '%Y-%m-%d') for date in dates]

    # 周K线 - 每周取一个数据点(默认取每周最后一个交易日)
    if kline_type == 'weekly':
        processed_data = []
        processed_dates = []
        current_week = -1

        for i in range(len(parsed_dates)):
            week = parsed_dates[i].isocalendar()[1]
            if week != current_week:
                current_week = week
                processed_data.append(data[i])
                processed_dates.append(dates[i])

        return processed_data, processed_dates

    # 月K线 - 每月取一个数据点
    elif kline_type == 'monthly':
        processed_data = []
        processed_dates = []
        current_month = -1

        for i in range(len(parsed_dates)):
            month = parsed_dates[i].month
            year = parsed_dates[i].year
            month_id = year * 12 + month

            if month_id != current_month:
                current_month = month_id
                processed_data.append(data[i])
                processed_dates.append(dates[i])

        return processed_data, processed_dates

    # 年K线 - 每年取一个数据点
    elif kline_type == 'yearly':
        processed_data = []
        processed_dates = []
        current_year = -1

        for i in range(len(parsed_dates)):
            year = parsed_dates[i].year

            if year != current_year:
                current_year = year
                processed_data.append(data[i])
                processed_dates.append(dates[i])

        return processed_data, processed_dates

    # 不支持的K线类型，返回原始数据
    return data, dates


def process_ratio_by_kline_type(
        result: Dict[str, Any],
        kline_type: str
) -> Dict[str, Any]:
    """
    根据K线类型处理价格比值结果
    
    Args:
        result: k_chart_fetcher返回的结果
        kline_type: K线类型 ('daily', 'weekly', 'monthly', 'yearly')
        
    Returns:
        处理后的结果字典
    """
    # 如果是日K或结果为空，直接返回
    if kline_type == 'daily' or not result or 'ratio' not in result or 'dates' not in result:
        return result

    # 处理价格比值和日期
    processed_ratio, processed_dates = process_kline_by_type(
        result['ratio'],
        result['dates'],
        kline_type
    )

    # 更新结果中的关键数据
    processed_result = result.copy()
    processed_result['ratio'] = processed_ratio
    processed_result['dates'] = processed_dates

    # 对应处理其他相关数据
    if 'fitting_line' in result and len(result['fitting_line']) == len(result['ratio']):
        processed_fitting_line, _ = process_kline_by_type(
            result['fitting_line'],
            result['dates'],
            kline_type
        )
        processed_result['fitting_line'] = processed_fitting_line

    if 'delta' in result and len(result['delta']) == len(result['ratio']):
        processed_delta, _ = process_kline_by_type(
            result['delta'],
            result['dates'],
            kline_type
        )
        processed_result['delta'] = processed_delta

    if 'close_a' in result and len(result['close_a']) == len(result['ratio']):
        processed_close_a, _ = process_kline_by_type(
            result['close_a'],
            result['dates'],
            kline_type
        )
        processed_result['close_a'] = processed_close_a

    if 'close_b' in result and len(result['close_b']) == len(result['ratio']):
        processed_close_b, _ = process_kline_by_type(
            result['close_b'],
            result['dates'],
            kline_type
        )
        processed_result['close_b'] = processed_close_b

    # 重新计算拟合曲线的标准差，因为变化了数据点
    if 'fitting_line' in processed_result and 'ratio' in processed_result:
        fitting_line = np.array(processed_result['fitting_line'])
        ratio = np.array(processed_result['ratio'])
        delta = ratio - fitting_line
        processed_result['delta'] = delta.tolist()
        processed_result['threshold'] = float(np.std(delta))

    return processed_result
