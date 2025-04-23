"""
投资信号生成与分析模块
"""
import bisect
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from indicators.technical_indicators import calculate_price_ratio_anomaly
from k_chart_fetcher import get_stock_data_pair, date_alignment, durations


def generate_investment_signals(
        close_a: List[float],
        close_b: List[float],
        dates: List[str],
        degree: int,
        threshold_multiplier: float
) -> List[Dict[str, Any]]:
    """
    生成投资信号
    
    参数:
        close_a: 股票A的收盘价列表
        close_b: 股票B的收盘价列表
        dates: 对应的日期列表
        degree: 多项式拟合的次数
        threshold_multiplier: 异常检测的阈值系数
    
    返回:
        投资信号列表
    """
    # 1. 计算价格比值
    ratio = [float(a) / float(b) for a, b in zip(close_a, close_b)]

    # 2. 使用多项式拟合比值曲线
    y = np.array(ratio)
    x = np.arange(len(ratio))
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    fitting_line = poly(x).tolist()

    # 3. 计算实际比值与拟合值的偏差
    delta = [r - f for r, f in zip(ratio, fitting_line)]

    # 4. 计算标准差作为阈值基准
    yd = np.array(delta)
    std_dev = np.std(yd)

    # 5. 使用异常检测算法识别异常点
    anomaly_info = calculate_price_ratio_anomaly(ratio, delta, threshold_multiplier, std_dev)

    # 6. 将异常点转化为投资信号
    signals = []
    for i, anomaly in enumerate(anomaly_info["anomalies"]):
        signal_id = i + 1
        signal_date = dates[anomaly["index"]]
        signal_ratio = float(ratio[anomaly["index"]])

        # 获取对应的原始delta值（未取绝对值前的偏差）
        original_delta = delta[anomaly["index"]]

        # 使用z_score绝对值表示偏差程度，但保留原值用于计算其他指标
        z_score = float(anomaly["z_score"])
        deviation = abs(float(anomaly["deviation"]) * 100)
        # 根据原始delta的正负确定信号类型，而不是使用z_score的正负
        signal_type = "positive" if original_delta > 0 else "negative"

        # 确定信号强度（基于z_score的绝对值和偏离程度）
        if z_score > 2.2 or deviation > 15.0:
            strength = "strong"
        elif z_score > 1.5 or deviation > 10.0:
            strength = "medium"
        else:
            strength = "weak"

        # 生成描述和建议
        if signal_type == "positive":
            description = f"价格比值显著高于拟合曲线(Z值:{z_score:.2f})，可能存在超买状态。"
            recommendation = "考虑卖出股票A或买入股票B进行获利。"
        else:
            description = f"价格比值显著低于拟合曲线(Z值:{z_score:.2f})，可能存在超卖状态。"
            recommendation = "考虑买入股票A或卖出股票B进行获利。"

        signals.append({
            "id": signal_id,
            "date": signal_date,
            "ratio": signal_ratio,
            "z_score": z_score,
            "type": signal_type,
            "strength": strength,
            "description": description,
            "recommendation": recommendation
        })

    return signals


def get_latest_price_ratio(
        code_a: str,
        code_b: str,
        trends_a: List[Dict[str, Any]],
        trends_b: List[Dict[str, Any]]
) -> Optional[float]:
    """
    从最新趋势数据中获取当前价格比值
    
    参数:
        code_a: 股票A代码
        code_b: 股票B代码
        trends_a: 股票A的趋势数据
        trends_b: 股票B的趋势数据
    
    返回:
        当前价格比值，如果无法获取则返回None
    """
    if not trends_a or not trends_b:
        return None

    # 获取最新价格
    latest_price_a = trends_a[-1].get('current_price') if trends_a else None
    latest_price_b = trends_b[-1].get('current_price') if trends_b else None

    if latest_price_a is not None and latest_price_b is not None and latest_price_b != 0:
        return float(latest_price_a) / float(latest_price_b)

    return None


def analyze_current_position(
        code_a: str,
        code_b: str,
        close_a: List[float],
        close_b: List[float],
        dates: List[str],
        signals: List[Dict[str, Any]],
        trends_a: List[Dict[str, Any]],
        trends_b: List[Dict[str, Any]],
        degree: int
) -> Dict[str, Any]:
    """
    分析当前价格比值位置
    
    参数:
        code_a: 股票A代码
        code_b: 股票B代码
        close_a: 股票A的收盘价列表
        close_b: 股票B的收盘价列表
        dates: 对应的日期列表
        signals: 投资信号列表
        trends_a: 股票A的趋势数据
        trends_b: 股票B的趋势数据
        degree: 多项式拟合的次数
    
    返回:
        当前位置分析结果
    """
    # 计算价格比值
    ratio = [float(a) / float(b) for a, b in zip(close_a, close_b)]

    # 获取当前最新比值
    current_ratio = get_latest_price_ratio(code_a, code_b, trends_a, trends_b)

    # 如果无法获取当前比值，使用历史最后一个
    if current_ratio is None and close_a and close_b and close_b[-1] != 0:
        current_ratio = close_a[-1] / close_b[-1]

    if current_ratio is None:
        # 无法获取最新数据时的默认值
        return {
            "current_ratio": 0,
            "nearest_signal_id": None,
            "similarity_score": None,
            "percentile": None,
            "is_extreme": False,
            "recommendation": "无法获取最新数据，请稍后再试。"
        }

    # 寻找最相似的历史信号
    nearest_signal_id = None
    max_similarity = 0
    similarity_score = None

    if signals:
        for signal in signals:
            # 使用比值的相对差异来计算相似度
            signal_ratio = signal["ratio"]
            diff = abs(signal_ratio - current_ratio) / max(signal_ratio, current_ratio)
            similarity = 1 - min(diff, 1.0)  # 限制在0-1之间

            if similarity > max_similarity:
                max_similarity = similarity
                nearest_signal_id = signal["id"]
                similarity_score = similarity

    # 计算百分位数
    if ratio:
        sorted_ratios = sorted(ratio)
        rank = bisect.bisect_left(sorted_ratios, current_ratio)
        percentile = rank / len(sorted_ratios)
    else:
        percentile = None

    # 计算与拟合曲线的偏离程度
    y = np.array(ratio)
    x = np.arange(len(ratio))
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)

    # 计算标准差
    fitting_line = poly(x).tolist()
    delta = [r - f for r, f in zip(ratio, fitting_line)]
    yd = np.array(delta)
    std_dev = np.std(yd)

    # 判断是否处于极端位置
    is_extreme = False
    current_z_score = None

    if std_dev > 0:
        mean_ratio = np.mean(ratio)
        current_z_score = (current_ratio - mean_ratio) / std_dev
        is_extreme = abs(current_z_score) > 2.0
        is_extreme = bool(is_extreme)

    # 生成推荐建议
    if is_extreme and current_z_score is not None:
        if current_z_score > 0:
            recommendation = "当前比值处于历史高位，建议关注做空套利机会（卖出股票A或买入股票B）。"
        else:
            recommendation = "当前比值处于历史低位，建议关注做多套利机会（买入股票A或卖出股票B）。"
    elif similarity_score and similarity_score > 0.8:
        nearest_signal = next((s for s in signals if s["id"] == nearest_signal_id), None)
        if nearest_signal:
            recommendation = f"当前比值与历史信号({nearest_signal['date']})高度相似，请参考该信号的表现。"
        else:
            recommendation = "当前比值在历史正常范围内，暂无明显异常。"
    else:
        recommendation = "当前比值在历史正常范围内，暂无明显异常。"

    return {
        "current_ratio": float(current_ratio) if current_ratio is not None else 0,
        "nearest_signal_id": int(nearest_signal_id) if nearest_signal_id is not None else None,
        "similarity_score": float(similarity_score) if similarity_score is not None else None,
        "percentile": float(percentile) if percentile is not None else None,
        "is_extreme": is_extreme,
        "recommendation": recommendation
    }




