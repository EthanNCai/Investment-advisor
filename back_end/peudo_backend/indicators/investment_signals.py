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
            "z_score_mark": anomaly["z_score_mark"],
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
        当前位置分析结果，包含:
        - current_ratio: 当前价格比值
        - nearest_signals: 最相似的多个历史信号(Top-3)
        - similarity_score: 最高相似度得分
        - percentile: 当前比值在历史分布中的百分位
        - deviation_from_trend: 相对于趋势线的偏离程度
        - volatility_level: 波动幅度级别 (low, medium, high)
        - is_extreme: 是否处于极端位置
        - z_score: 当前比值的Z得分
        - historical_signal_pattern: 历史信号模式分析
        - recommendation: 综合推荐建议
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
            "nearest_signals": [],
            "similarity_score": None,
            "percentile": None,
            "deviation_from_trend": None,
            "volatility_level": None,
            "is_extreme": False,
            "z_score": None,
            "historical_signal_pattern": None,
            "recommendation": "无法获取最新数据，请稍后再试。"
        }

    # 1. 多项式拟合比值曲线
    y = np.array(ratio)
    x = np.arange(len(ratio))
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)
    fitting_line = poly(x).tolist()

    # 计算拟合曲线的延伸
    future_point = len(ratio)  # 当前点位于时间序列的末尾
    trend_direction = poly(future_point) - poly(future_point - 1)
    trend_projection = poly(future_point)

    # 2. 计算偏差和标准差
    delta = [r - f for r, f in zip(ratio, fitting_line)]
    yd = np.array(delta)
    std_dev = np.std(yd)
    mean_ratio = np.mean(ratio)

    # 3. 计算当前比值的Z分数
    current_z_score = None
    if std_dev > 0:
        current_z_score = (current_ratio - trend_projection) / std_dev

    # 4. 计算波动性水平
    recent_volatility = None
    if len(ratio) >= 30:  # 只有当有足够的数据时才计算
        # 计算最近30个交易日的波动率
        recent_ratios = ratio[-30:]
        recent_volatility = np.std(recent_ratios) / np.mean(recent_ratios)

    volatility_level = None
    if recent_volatility is not None:
        if recent_volatility < 0.05:
            volatility_level = "low"
        elif recent_volatility < 0.10:
            volatility_level = "medium"
        else:
            volatility_level = "high"

    # 5. 寻找最相似的历史信号（优化为多个相似信号）
    similarity_threshold = 0.4
    similar_signals = []
    res = []
    if signals:
        # 计算每个历史信号与当前比值的相似度
        signal_similarities = []
        for signal in signals:
            signal_ratio = signal["ratio"]

            # 计算比值差异的新方法
            # 1. 首先计算比值的百分比差异
            absolute_diff = abs(signal_ratio - current_ratio)
            relative_diff = absolute_diff / max(signal_ratio, current_ratio)
            # 2. 放大小差异以便更好区分高相似度值
            # 使用适中的指数函数，避免相似度值过低
            ratio_similarity = np.exp(-10 * relative_diff)

            # 3. 方向匹配加权（更强的加权效果）
            direction_match = 1.0
            if current_z_score is not None and "z_score_mark" in signal:
                if (current_z_score > 0 and signal["z_score_mark"] > 0) or (
                        current_z_score < 0 and signal["z_score_mark"] < 0):
                    direction_match = 1.25  # 方向一致性加分提高
                else:
                    direction_match = 0.85  # 方向不一致性减分降低

            # 4. 时间接近度（较新的信号可能更有参考价值）- 使时间权重差异更大
            time_weight = 1.0
            if len(dates) > 0 and "date" in signal:
                try:
                    signal_index = dates.index(signal["date"])
                    time_factor = signal_index / len(dates)  # 0为最早的信号，1为最新的信号
                    time_weight = 0.85 + (0.15 * time_factor)
                except ValueError:
                    # 信号日期不在当前日期列表中时，维持默认权重
                    pass

            # 5. Z值相似性（添加新的相似度因子）
            z_score_similarity = 1.0
            if current_z_score is not None and "z_score_mark" in signal:
                z_diff = abs(current_z_score - signal["z_score_mark"])
                z_score_similarity = np.exp(-0.5 * z_diff)

            # 6. 信号强度相似性
            strength_similarity = 1.0
            if "strength" in signal:
                strength_map = {"weak": 1, "medium": 2, "strong": 3}
                # 基于当前Z分数估计当前强度
                current_strength = "medium"
                if current_z_score is not None:
                    if abs(current_z_score) > 2.0:
                        current_strength = "strong"
                    elif abs(current_z_score) < 1.5:
                        current_strength = "weak"

                if current_strength and signal["strength"]:
                    strength_diff = abs(strength_map.get(current_strength, 2) - strength_map.get(signal["strength"], 2))
                    strength_similarity = 1.0 - (strength_diff * 0.2)

            # 7. 添加基础相似度以确保不会太低
            base_similarity = 0.3

            # 8. 综合相似度得分并确保最终相似度不超过1.0
            raw_similarity = base_similarity + (
                    1 - base_similarity) * ratio_similarity * direction_match * time_weight * z_score_similarity * strength_similarity
            res.append(round(raw_similarity, 2)) if raw_similarity > 0.95 else None
            similarity = min(raw_similarity, 0.98)  # 限制最大相似度为0.98
            # 加入到待排序列表
            signal_similarities.append((signal, similarity))
        # 按相似度从高到低排序
        signal_similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取前3个最相似的信号（满足最低相似度要求）
        top_similar_signals = [
                                  {
                                      "id": s[0]["id"],
                                      "date": s[0]["date"],
                                      "ratio": s[0]["ratio"],
                                      "similarity": float(s[1]),
                                      "type": s[0]["type"],
                                      "strength": s[0]["strength"]
                                  }
                                  for s in signal_similarities if s[1] >= similarity_threshold
                              ][:3]

        similar_signals = top_similar_signals

    # 设置最高相似度得分
    similarity_score = similar_signals[0]["similarity"] if similar_signals else None
    nearest_signal_id = similar_signals[0]["id"] if similar_signals else None

    # 6. 计算百分位数（优化：使用加权累积分布函数）
    percentile = None
    if ratio:
        # 考虑最近数据的权重更大
        weights = np.linspace(0.5, 1.0, len(ratio))  # 线性递增权重
        sorted_indices = np.argsort(ratio)

        # 找出当前比值在排序后的位置
        current_pos = bisect.bisect_left(sorted(ratio), current_ratio)

        # 计算加权累积分布
        if current_pos == 0:
            percentile = 0.0
        elif current_pos >= len(ratio):
            percentile = 1.0
        else:
            # 计算加权百分位数
            weight_sum = np.sum(weights)
            lower_weight_sum = np.sum(weights[sorted_indices[:current_pos]])
            percentile = lower_weight_sum / weight_sum

    # 7. 计算当前偏离趋势的程度
    deviation_from_trend = None
    if current_ratio is not None and trend_projection is not None:
        deviation_from_trend = (current_ratio - trend_projection) / trend_projection * 100

    # 8. 判断是否处于极端位置
    is_extreme = False
    if current_z_score is not None:
        is_extreme = abs(current_z_score) > 1.8

    # 9. 分析历史信号模式
    historical_signal_pattern = None
    if signals and len(signals) >= 3:
        # 获取最近3个信号
        recent_signals = sorted(signals, key=lambda s: s["date"], reverse=True)[:3]

        # 检查信号类型模式
        signal_types = [s["type"] for s in recent_signals]

        if all(s == "positive" for s in signal_types):
            historical_signal_pattern = "连续超买"
        elif all(s == "negative" for s in signal_types):
            historical_signal_pattern = "连续超卖"
        elif signal_types[0] != signal_types[1]:
            historical_signal_pattern = "震荡切换"
        else:
            historical_signal_pattern = "混合模式"

    # 10. 生成综合推荐建议
    recommendation = ""

    # 基于当前Z分数的极端值判断
    if is_extreme and current_z_score is not None:
        if current_z_score > 0:
            recommendation = f"当前比值处于历史高位(Z值:{current_z_score:.2f})，建议关注做空套利机会（卖出股票A或买入股票B）。"
        else:
            recommendation = f"当前比值处于历史低位(Z值:{current_z_score:.2f})，建议关注做多套利机会（买入股票A或卖出股票B）。"

    # 考虑趋势与当前位置的综合分析
    elif trend_direction is not None and deviation_from_trend is not None:
        if trend_direction > 0 and deviation_from_trend < -5:
            recommendation = "当前比值低于上升趋势线，可能存在回归机会。考虑买入股票A或卖出股票B。"
        elif trend_direction < 0 and deviation_from_trend > 5:
            recommendation = "当前比值高于下降趋势线，可能存在回归机会。考虑卖出股票A或买入股票B。"
        elif abs(deviation_from_trend) < 3:
            recommendation = "当前比值贴近趋势线，建议观望或继续跟踪趋势发展。"

    # 与历史信号的比较分析
    elif similarity_score and similarity_score > 0.8 and nearest_signal_id is not None:
        nearest_signal = next((s for s in signals if s["id"] == nearest_signal_id), None)
        if nearest_signal:
            signal_desc = "超买" if nearest_signal["type"] == "positive" else "超卖"
            strength_desc = {"weak": "弱", "medium": "中等", "strong": "强"}[nearest_signal["strength"]]
            recommendation = f"当前比值与历史{nearest_signal['date']}的{signal_desc}信号高度相似(相似度:{similarity_score:.2f})，当时为{strength_desc}信号。请参考该信号后续的市场表现。"

    # 考虑历史模式
    elif historical_signal_pattern:
        if historical_signal_pattern == "连续超买" and percentile and percentile > 0.7:
            recommendation = "近期历史信号显示连续超买状态，当前比值位于较高位置，建议保持谨慎。"
        elif historical_signal_pattern == "连续超卖" and percentile and percentile < 0.3:
            recommendation = "近期历史信号显示连续超卖状态，当前比值位于较低位置，可能存在机会。"
        elif historical_signal_pattern == "震荡切换":
            recommendation = "近期市场处于震荡状态，建议关注价格比值突破重要阈值的情况。"

    # 没有明显信号时的建议
    else:
        recommendation = "当前比值在历史正常范围内，暂无明显异常。"

    # 添加波动性提示
    if volatility_level == "high":
        recommendation += " 注意：当前波动性较高，交易需谨慎。"

    return {
        "current_ratio": float(current_ratio) if current_ratio is not None else 0,
        "nearest_signals": similar_signals,
        "similarity_score": float(similarity_score) if similarity_score is not None else None,
        "percentile": float(percentile) if percentile is not None else None,
        "deviation_from_trend": float(deviation_from_trend) if deviation_from_trend is not None else None,
        "volatility_level": volatility_level,
        "is_extreme": is_extreme,
        "z_score": float(current_z_score) if current_z_score is not None else None,
        "historical_signal_pattern": historical_signal_pattern,
        "recommendation": recommendation
    }

