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
