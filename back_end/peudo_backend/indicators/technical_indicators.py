import numpy as np
from typing import List, Dict, Any, Optional


def calculate_indicators(kline_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算各种技术指标
    
    参数:
        kline_data: K线数据列表
    
    返回:
        包含各项技术指标的字典
    """
    # 提取收盘价数组
    closes = np.array([k['close'] for k in kline_data])
    highs = np.array([k['high'] for k in kline_data])
    lows = np.array([k['low'] for k in kline_data])

    # 计算移动平均线
    ma5 = calculate_ma(closes, 5)
    ma10 = calculate_ma(closes, 10)
    ma20 = calculate_ma(closes, 20)
    ma60 = calculate_ma(closes, 60)

    # 计算MACD
    macd_dict = calculate_macd(closes)

    # 计算RSI
    rsi_dict = calculate_rsi(closes)
    
    # 计算KDJ
    kdj_dict = calculate_kdj(highs, lows, closes)

    return {
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20,
        "ma60": ma60,
        "macd": macd_dict,
        "rsi": rsi_dict,
        "kdj": kdj_dict
    }


def calculate_ma(prices: np.ndarray, window: int) -> List[float]:
    """
    计算移动平均线
    
    参数:
        prices: 价格数组
        window: 窗口大小
        
    返回:
        移动平均线数组
    """
    if len(prices) < window:
        # 如果数据不足，填充None
        return [None] * len(prices)

    # 计算移动平均线
    ma = []
    for i in range(len(prices)):
        if i < window - 1:
            ma.append(None)  # 数据不足时填充None
        else:
            ma.append(float(np.mean(prices[i - window + 1:i + 1])))

    return [round(x, 2) if x is not None else None for x in ma]


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
    """
    计算MACD指标
    
    参数:
        prices: 价格数组
        fast: 快线周期
        slow: 慢线周期
        signal: 信号线周期
        
    返回:
        包含DIF, DEA和MACD的字典
    """
    # 计算EMA
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    # 计算DIF (MACD Line)
    dif = []
    for i in range(len(prices)):
        if ema_fast[i] is None or ema_slow[i] is None:
            dif.append(None)
        else:
            dif.append(ema_fast[i] - ema_slow[i])

    # 计算DEA (Signal Line)
    dea = calculate_ema(np.array([d if d is not None else 0 for d in dif]), signal)

    # 计算MACD柱状图
    macd = []
    for i in range(len(dif)):
        if dif[i] is None or dea[i] is None:
            macd.append(None)
        else:
            # MACD柱状图 = (DIF - DEA) * 2
            macd.append((dif[i] - dea[i]) * 2)

    return {
        "dif": [round(x, 2) if x is not None else None for x in dif],
        "dea": [round(x, 2) if x is not None else None for x in dea],
        "macd": [round(x, 2) if x is not None else None for x in macd]
    }


def calculate_ema(prices: np.ndarray, window: int) -> List[float]:
    """
    计算指数移动平均线 (EMA)
    
    参数:
        prices: 价格数组
        window: 窗口大小
        
    返回:
        EMA数组
    """
    if len(prices) < window:
        return [None] * len(prices)

    ema = [None] * (window - 1)
    # 初始EMA等于前window天的简单平均值
    ema.append(float(np.mean(prices[:window])))

    # 计算后续的EMA
    multiplier = 2.0 / (window + 1)
    for i in range(window, len(prices)):
        ema.append(prices[i] * multiplier + ema[-1] * (1 - multiplier))

    return ema


def calculate_rsi(prices: np.ndarray, windows: List[int] = [6, 12, 24]) -> Dict[str, List[float]]:
    """
    计算RSI指标
    
    参数:
        prices: 价格数组
        windows: RSI周期列表
        
    返回:
        包含不同周期RSI的字典
    """
    result = {}

    for window in windows:
        rsi_values = []

        if len(prices) <= window:
            # 数据不足
            rsi_values = [None] * len(prices)
        else:
            # 计算价格变化
            deltas = np.diff(prices)

            # 计算每个窗口的RSI
            for i in range(len(prices)):
                if i < window:
                    rsi_values.append(None)  # 数据不足
                else:
                    # 获取窗口内的价格变化
                    window_deltas = deltas[i - window:i]

                    # 计算上涨和下跌的平均值
                    gains = np.sum(window_deltas[window_deltas > 0]) / window if any(window_deltas > 0) else 0
                    losses = -np.sum(window_deltas[window_deltas < 0]) / window if any(window_deltas < 0) else 0

                    if losses == 0:
                        rsi = 100.0  # 避免除以零
                    else:
                        rs = gains / losses
                        rsi = 100.0 - (100.0 / (1.0 + rs))

                    rsi_values.append(round(rsi, 2))

        result[f"rsi{window}"] = rsi_values

    return result


def calculate_kdj(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, n: int = 9, m1: int = 3, m2: int = 3) -> Dict[str, List[float]]:
    """
    计算KDJ指标
    
    参数:
        highs: 最高价数组
        lows: 最低价数组
        closes: 收盘价数组
        n: RSV计算周期，默认9
        m1: K值平滑因子，默认3
        m2: D值平滑因子，默认3
    
    返回:
        包含K、D、J值的字典
    """
    if len(closes) < n:
        # 数据不足
        return {
            "k": [None] * len(closes),
            "d": [None] * len(closes),
            "j": [None] * len(closes)
        }
    
    # 计算RSV (Raw Stochastic Value)
    rsv = []
    for i in range(len(closes)):
        if i < n - 1:
            rsv.append(None)
        else:
            # 获取n日内的最高价和最低价
            highest_high = np.max(highs[i-n+1:i+1])
            lowest_low = np.min(lows[i-n+1:i+1])
            
            # 计算RSV值
            if highest_high == lowest_low:
                rsv.append(50)  # 避免除以零
            else:
                rsv_value = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
                rsv.append(rsv_value)
    
    # 计算K值（初始K值为50）
    k_values = [50.0 if i < n - 1 else None for i in range(n - 1)]
    
    # 计算第一个有效的K值
    if rsv[n-1] is not None:
        k_values.append((2/3) * 50 + (1/3) * rsv[n-1])
    else:
        k_values.append(50.0)
    
    # 计算其余K值
    for i in range(n, len(closes)):
        if rsv[i] is not None:
            k_value = (m1-1)/m1 * k_values[-1] + 1/m1 * rsv[i]
            k_values.append(k_value)
        else:
            k_values.append(k_values[-1])
    
    # 计算D值（初始D值为50）
    d_values = [50.0 if i < n - 1 else None for i in range(n - 1)]
    
    # 计算第一个有效的D值
    if k_values[n-1] is not None:
        d_values.append((2/3) * 50 + (1/3) * k_values[n-1])
    else:
        d_values.append(50.0)
    
    # 计算其余D值
    for i in range(n, len(closes)):
        if k_values[i] is not None:
            d_value = (m2-1)/m2 * d_values[-1] + 1/m2 * k_values[i]
            d_values.append(d_value)
        else:
            d_values.append(d_values[-1])
    
    # 计算J值
    j_values = []
    for i in range(len(closes)):
        if i < n - 1 or k_values[i] is None or d_values[i] is None:
            j_values.append(None)
        else:
            j_value = 3 * k_values[i] - 2 * d_values[i]
            j_values.append(j_value)
    
    # 四舍五入到2位小数
    return {
        "k": [round(x, 2) if x is not None else None for x in k_values],
        "d": [round(x, 2) if x is not None else None for x in d_values],
        "j": [round(x, 2) if x is not None else None for x in j_values]
    }


def calculate_price_ratio_anomaly(ratio_data: List[float], delta_data: List[float], threshold: float) -> Dict:
    """
    计算价差异常值并生成预警信息，同时考虑Z分数和偏离度
    
    Args:
        ratio_data: 价差数据列表
        delta_data: 与拟合曲线的差值列表
        threshold: 异常值检测阈值（标准差）
        
    Returns:
        Dict: 包含异常值检测结果的字典
    """
    if not ratio_data or not delta_data:
        return {
            "mean": 0,
            "std": 0,
            "anomalies": [],
            "warning_level": "normal",
            "upper_bound": 0,
            "lower_bound": 0
        }
    
    # 计算基础统计值
    mean_ratio = sum(ratio_data) / len(ratio_data)
    std = threshold  # 使用传入的标准差
    
    # 检测异常值 - 同时考虑Z分数和偏离度
    anomalies = []
    # 先收集所有可能的异常点及其指标
    potential_anomalies = []
    
    for i, (ratio, delta) in enumerate(zip(ratio_data, delta_data)):
        # 计算Z分数 - 基于与拟合线的偏差
        z_score = abs(delta / std) if std != 0 else 0
        
        # 计算偏离度 - 基于与均值的相对偏差（百分比）
        deviation = (ratio - mean_ratio) / mean_ratio if mean_ratio != 0 else 0
        deviation_pct = abs(deviation * 100)  # 偏离百分比的绝对值
        
        # 使用基础阈值初步筛选潜在异常点
        if z_score > 2.0 or deviation_pct > 5.0:
            potential_anomalies.append({
                "index": i,
                "value": ratio,
                "z_score": z_score,
                "deviation": deviation,
                "deviation_pct": deviation_pct,
                # 综合分数 = Z分数 * (1 + 归一化的偏离度影响)
                "combined_score": z_score * (1 + min(deviation_pct / 20, 1.0))
            })
    
    # 对潜在异常点按综合分数排序
    potential_anomalies.sort(key=lambda x: x["combined_score"], reverse=True)
    
    # 取分数最高的点作为确认的异常点（最多取原始数据的10%或至少6个点）
    max_anomalies = max(min(len(ratio_data) // 10, 20), 6)
    for anomaly in potential_anomalies[:max_anomalies]:
        if anomaly["combined_score"] > 2.0:  # 综合分数阈值
            anomalies.append({
                "index": anomaly["index"],
                "value": anomaly["value"],
                "z_score": anomaly["z_score"],
                "deviation": anomaly["deviation"]
            })
    
    # 确定预警级别 - 使用综合评分而非仅Z分数
    warning_level = "normal"
    if anomalies:
        # 找出偏离度最大的异常点
        max_deviation_anomaly = max(anomalies, key=lambda x: abs(x["deviation"]))
        max_z_score_anomaly = max(anomalies, key=lambda x: x["z_score"])
        
        max_deviation_pct = abs(max_deviation_anomaly["deviation"] * 100)
        max_z_score = max_z_score_anomaly["z_score"]
        
        # 根据偏离度和Z分数综合判断风险级别
        if (max_z_score > 3.0 and max_deviation_pct > 15.0) or max_deviation_pct > 25.0:
            warning_level = "high"
        elif (max_z_score > 2.5 and max_deviation_pct > 10.0) or max_deviation_pct > 15.0 or max_z_score > 3.0:
            warning_level = "medium"
    
    # 返回结果
    return {
        "mean": mean_ratio,
        "std": std,
        "anomalies": anomalies,
        "warning_level": warning_level,
        "upper_bound": mean_ratio + 2.0 * std,
        "lower_bound": mean_ratio - 2.0 * std
    }
