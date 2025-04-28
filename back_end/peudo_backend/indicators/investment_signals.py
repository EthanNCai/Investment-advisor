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
            "id": int(signal_id),
            "date": signal_date,
            "ratio": float(signal_ratio),
            "z_score": float(z_score),
            "z_score_mark": float(anomaly["z_score_mark"]),
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
        - trend_strength: 趋势强度
        - support_resistance: 支撑和阻力位
        - mean_reversion_probability: 均值回归概率
        - cycle_position: 在周期中的位置
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
            "trend_strength": None,
            "support_resistance": None,
            "mean_reversion_probability": None,
            "cycle_position": None,
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
        # 计算较长期的波动率（如90天）作为比较基准
        long_term_ratios = ratio[-min(90, len(ratio)):]
        long_term_volatility = np.std(long_term_ratios) / np.mean(long_term_ratios)

        # 比较近期波动率与长期波动率
        relative_volatility = recent_volatility / long_term_volatility if long_term_volatility else 1.0

    volatility_level = None
    if recent_volatility is not None:
        if recent_volatility < 0.035:
            volatility_level = "low"
        elif recent_volatility < 0.08:
            volatility_level = "medium"
        else:
            volatility_level = "high"

    # 5. 寻找最相似的历史信号（优化为多个相似信号）
    similarity_threshold = 0.5
    similar_signals = []
    res = []
    if signals:
        # 计算每个历史信号与当前比值的相似度
        signal_similarities = []
        for signal in signals:
            signal_ratio = signal["ratio"]

            # 1. 比值相似度计算改进
            # 使用对数差异计算，对不同数量级的比值更公平
            log_current = np.log(current_ratio) if current_ratio > 0 else 0
            log_signal = np.log(signal_ratio) if signal_ratio > 0 else 0
            log_diff = abs(log_current - log_signal)
            # 使用sigmoid函数平滑相似度曲线，保持值在0-1之间
            ratio_similarity = 1 / (1 + np.exp(5 * log_diff))

            # 2. 考虑相对于趋势线的位置（新增）
            position_similarity = 1.0
            if current_z_score is not None and "z_score_mark" in signal:
                # 如果两者相对于趋势线的位置类似（同侧且幅度接近），增加相似度
                z_diff = abs(current_z_score - signal["z_score_mark"])
                position_similarity = np.exp(-0.3 * z_diff)  # 指数衰减，但衰减速度较慢

            # 3. 改进方向匹配权重：考虑Z值大小，而不仅是正负
            direction_match = 1.0
            if current_z_score is not None and "z_score_mark" in signal:
                # 方向一致性
                direction_same = (current_z_score > 0 and signal["z_score_mark"] > 0) or (
                        current_z_score < 0 and signal["z_score_mark"] < 0)

                # 极端程度匹配（两者都是极端值或都是中性值）
                extreme_match = (abs(current_z_score) > 1.8 and abs(signal["z_score_mark"]) > 1.8) or (
                        abs(current_z_score) <= 1.8 and abs(signal["z_score_mark"]) <= 1.8)

                if direction_same and extreme_match:
                    direction_match = 1.3  # 方向和极端程度都匹配时加分最多
                elif direction_same:
                    direction_match = 1.15  # 仅方向匹配时适度加分
                else:
                    direction_match = 0.8  # 方向不匹配时更大幅度减分

            # 4. 动态时间权重（新增）
            # 根据市场环境变化调整时间权重重要性
            time_weight = 1.0
            if len(dates) > 0 and "date" in signal:
                try:
                    signal_index = dates.index(signal["date"])
                    # 计算基础时间因子（0为最早信号，1为最新信号）
                    time_factor = signal_index / len(dates)

                    # 动态调整：在波动性高的市场中，最近的信号更重要
                    if volatility_level == "high":
                        time_weight = 0.7 + (0.3 * time_factor)  # 最新信号权重达到1.0，最旧信号为0.7
                    elif volatility_level == "medium":
                        time_weight = 0.8 + (0.2 * time_factor)  # 最新信号权重达到1.0，最旧信号为0.8
                    else:  # low volatility
                        time_weight = 0.9 + (0.1 * time_factor)  # 最新信号权重达到1.0，最旧信号为0.9
                except ValueError:
                    # 信号日期不在当前日期列表中时，维持默认权重
                    pass

            # 5. 市场环境相似性
            market_similarity = 1.0

            # 计算当前趋势方向
            current_trend_direction = None
            if len(fitting_line) >= 2:
                # 使用拟合线的最近部分估计趋势
                window_size = min(30, len(fitting_line))
                recent_trend = fitting_line[-window_size:]
                if len(recent_trend) >= 2:
                    slope = (recent_trend[-1] - recent_trend[0]) / (len(recent_trend) - 1)
                    current_trend_direction = "上升" if slope > 0 else "下降"

            if current_trend_direction and "type" in signal and "z_score_mark" in signal:
                # 推断历史信号的市场环境
                signal_trend = None

                # 基于信号类型和Z分数推断当时的趋势方向
                if signal["type"] == "positive":  # 高于趋势线
                    signal_trend = "上升" if signal["z_score_mark"] > 0 else "下降"
                else:  # 负向信号
                    signal_trend = "下降" if signal["z_score_mark"] < 0 else "上升"

                # 趋势方向匹配时增加相似度
                if signal_trend == current_trend_direction:
                    market_similarity = 1.15  # 趋势方向相同
                else:
                    market_similarity = 0.9  # 趋势方向相反

                # 进一步考虑Z分数相似性
                if current_z_score is not None and "z_score_mark" in signal:
                    # 信号类型相同且Z分数绝对值相近时，增加相似度
                    if (current_z_score * signal["z_score_mark"] > 0 and  # 同号
                            abs(abs(current_z_score) - abs(signal["z_score_mark"])) < 1.0):  # 幅度相近
                        market_similarity *= 1.05  # 额外加分

            # 6. 改进的信号强度相似性
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
                    # 使用指数函数使得差异更显著
                    strength_similarity = np.exp(-0.4 * strength_diff)

            # 添加：7. 价格形态相似性（比较历史价格比率模式）
            pattern_similarity = 1.0
            if len(dates) > 0 and "date" in signal:
                try:
                    # 寻找信号在历史序列中的位置
                    signal_index = dates.index(signal["date"])

                    # 确保有足够的数据点进行比较
                    if signal_index >= 10 and len(ratio) - 10 >= 0:
                        # 提取历史信号前后的价格比率模式（共21个点，信号在中间）
                        signal_pattern_start = max(0, signal_index - 10)
                        signal_pattern_end = min(len(ratio), signal_index + 11)
                        signal_pattern = ratio[signal_pattern_start:signal_pattern_end]

                        # 提取当前比值前的价格比率模式
                        current_pattern_end = len(ratio)
                        current_pattern_start = max(0, current_pattern_end - 21)
                        current_pattern = ratio[current_pattern_start:current_pattern_end]

                        # 标准化两个模式以便比较形状而非绝对值
                        if signal_pattern and current_pattern:
                            norm_signal_pattern = [
                                (x - min(signal_pattern)) / (max(signal_pattern) - min(signal_pattern) + 1e-10)
                                for x in signal_pattern]
                            norm_current_pattern = [
                                (x - min(current_pattern)) / (max(current_pattern) - min(current_pattern) + 1e-10)
                                for x in current_pattern]

                            # 计算形态相似度（使用二者长度的较小值）
                            min_length = min(len(norm_signal_pattern), len(norm_current_pattern))
                            if min_length > 5:  # 确保有足够的点进行比较
                                # 计算模式差异（均方根误差）
                                pattern_diff_sum = sum((norm_signal_pattern[i] - norm_current_pattern[i]) ** 2
                                                       for i in range(min_length))
                                pattern_diff = (pattern_diff_sum / min_length) ** 0.5

                                # 将差异转换为相似度值
                                pattern_similarity = np.exp(-5 * pattern_diff)  # 指数衰减，差异越大，相似度越低
                except (ValueError, IndexError):
                    # 处理各种潜在异常，保持默认相似度
                    pass

            # 原来的基础相似度动态调整
            base_similarity = 0.25

            # 调整动态权重合成，加入新的形态相似性
            total_weight = 4.0  # 更新权重总和

            # 更新权重分配
            ratio_weight = 1.1 / total_weight  # 比值相似度权重
            position_weight = 0.8 / total_weight  # 位置相似度
            direction_weight = 0.6 / total_weight  # 方向匹配
            time_weight_factor = 0.5 / total_weight  # 时间因素
            market_weight = 0.3 / total_weight  # 市场环境相似度
            strength_weight = 0.1 / total_weight  # 信号强度相似度
            pattern_weight = 0.6 / total_weight  # 新增：价格形态相似度

            # 更新最终相似度计算
            weighted_similarity = (
                    base_similarity +
                    (1 - base_similarity) * (
                            ratio_similarity * ratio_weight +
                            position_similarity * position_weight +
                            direction_match * direction_weight +
                            time_weight * time_weight_factor +
                            market_similarity * market_weight +
                            strength_similarity * strength_weight +
                            pattern_similarity * pattern_weight
                    )
            )

            # 对非常相似的信号进行记录（用于调试）
            if weighted_similarity > 0.85:
                res.append(round(weighted_similarity, 2))

            # 限制最大相似度为0.95，保留一定区分度
            similarity = min(weighted_similarity, 0.95)

            # 加入到待排序列表
            signal_similarities.append((signal, similarity))

        # 按相似度从高到低排序
        signal_similarities.sort(key=lambda x: x[1], reverse=True)

        # 优化获取最相似信号的策略，提高多样性和覆盖性
        # 步骤1：先选择相似度最高的信号
        selected_signals = []
        unique_patterns = set()  # 用于跟踪已选择的信号模式类型

        # 首先添加最相似的信号
        if signal_similarities and signal_similarities[0][1] >= similarity_threshold:
            top_signal = signal_similarities[0]
            selected_signals.append(top_signal)
            # 记录其类型和强度组合，用于确保多样性
            if "type" in top_signal[0] and "strength" in top_signal[0]:
                unique_patterns.add((top_signal[0]["type"], top_signal[0]["strength"]))

        # 步骤2：遍历剩余信号，尝试添加不同类型/强度组合的信号
        remaining_candidates = signal_similarities[1:] if signal_similarities else []
        for signal_tuple in remaining_candidates:
            if len(selected_signals) >= 3:  # 最多选3个信号
                break

            signal, sim = signal_tuple

            # 只考虑达到阈值的信号
            if sim < similarity_threshold:
                continue

            # 检查是否是新的类型/强度组合
            if "type" in signal and "strength" in signal:
                pattern_key = (signal["type"], signal["strength"])
                if pattern_key not in unique_patterns:
                    selected_signals.append(signal_tuple)
                    unique_patterns.add(pattern_key)
                    continue

            # 如果还没有足够的信号，再检查时间跨度
            # 优先选择与现有选择在时间上有显著差异的信号
            if len(selected_signals) < 3 and "date" in signal:
                # 检查与已选信号的时间差异
                signal_date = signal["date"]
                min_time_diff = float('inf')

                for selected in selected_signals:
                    if "date" in selected[0]:
                        try:
                            selected_date = selected[0]["date"]
                            # 简化处理：计算日期字符串在排序中的距离
                            date_diff = abs(dates.index(signal_date) - dates.index(selected_date))
                            min_time_diff = min(min_time_diff, date_diff)
                        except (ValueError, IndexError):
                            continue

                # 如果时间差异足够大(至少间隔30个交易日)，也加入选择
                if min_time_diff > 30:
                    selected_signals.append(signal_tuple)

        # 步骤3：如果通过多样性筛选后仍未满足3个信号，则补充相似度最高的信号
        if len(selected_signals) < 3:
            for signal_tuple in remaining_candidates:
                if len(selected_signals) >= 3:
                    break

                signal, sim = signal_tuple
                # 避免重复添加
                if signal_tuple not in selected_signals and sim >= similarity_threshold:
                    selected_signals.append(signal_tuple)

        # 将选择的信号转换为前端所需格式
        top_similar_signals = [
            {
                "id": s[0]["id"],
                "date": s[0]["date"],
                "ratio": s[0]["ratio"],
                "similarity": float(s[1]),
                "type": s[0]["type"],
                "strength": s[0]["strength"]
            }
            for s in selected_signals
        ]

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

    # 10. 新增：计算趋势强度
    trend_strength = None
    if len(fitting_line) >= 2:
        # 使用拟合线的斜率计算趋势强度
        recent_window = min(30, len(fitting_line))
        recent_fit = fitting_line[-recent_window:]

        if recent_window >= 2:
            trend_slope = (recent_fit[-1] - recent_fit[0]) / (recent_window - 1)
            trend_norm = abs(trend_slope) / mean_ratio  # 标准化斜率

            # 根据标准化斜率确定趋势强度
            if trend_norm < 0.001:  # 几乎没有趋势
                trend_strength = {
                    "value": round(float(trend_norm * 100), 3),
                    "level": "无明显趋势",
                    "direction": "平稳"
                }
            elif trend_norm < 0.003:  # 弱趋势
                trend_strength = {
                    "value": round(float(trend_norm * 100), 3),
                    "level": "弱趋势",
                    "direction": "上升" if trend_slope > 0 else "下降"
                }
            elif trend_norm < 0.008:  # 中等趋势
                trend_strength = {
                    "value": round(float(trend_norm * 100), 3),
                    "level": "中等趋势",
                    "direction": "上升" if trend_slope > 0 else "下降"
                }
            else:  # 强趋势
                trend_strength = {
                    "value": round(float(trend_norm * 100), 3),
                    "level": "强趋势",
                    "direction": "上升" if trend_slope > 0 else "下降"
                }

    # 11. 新增：计算支撑位和阻力位
    support_resistance = None
    if len(ratio) >= 30:
        # 使用历史价格分布识别支撑位和阻力位
        sorted_ratios = sorted(ratio)
        q1 = np.percentile(sorted_ratios, 25)  # 第一四分位数作为强支撑位
        q3 = np.percentile(sorted_ratios, 75)  # 第三四分位数作为强阻力位

        # 找出最近的局部最低点作为近期支撑
        recent_window = min(60, len(ratio))
        recent_ratios = ratio[-recent_window:]

        # 计算局部最小值和最大值（至少间隔5个点）
        local_mins = []
        local_maxs = []
        for i in range(5, len(recent_ratios) - 5):
            if recent_ratios[i] == min(recent_ratios[i - 5:i + 6]):
                local_mins.append(recent_ratios[i])
            if recent_ratios[i] == max(recent_ratios[i - 5:i + 6]):
                local_maxs.append(recent_ratios[i])

        # 找出当前比值最近的支撑位和阻力位
        nearby_support = None
        nearby_resistance = None

        if local_mins:
            # 找出低于当前比值的最高支撑位
            lower_supports = [r for r in local_mins if r < current_ratio]
            if lower_supports:
                nearby_support = max(lower_supports)

        if local_maxs:
            # 找出高于当前比值的最低阻力位
            higher_resistances = [r for r in local_maxs if r > current_ratio]
            if higher_resistances:
                nearby_resistance = min(higher_resistances)

        # 如果没有找到本地支撑/阻力位，使用统计分布
        if nearby_support is None:
            nearby_support = q1
        if nearby_resistance is None:
            nearby_resistance = q3

        support_resistance = {
            "strong_support": round(float(q1), 3),
            "nearby_support": round(float(nearby_support), 3),
            "nearby_resistance": round(float(nearby_resistance), 3),
            "strong_resistance": round(float(q3), 3)
        }

    # 12. 新增：计算均值回归概率
    mean_reversion_probability = None
    if current_z_score is not None:
        # 根据Z分数计算均值回归概率
        # Z分数越高，均值回归概率越大
        if abs(current_z_score) > 2.3:
            prob = 0.85  # 极端偏离，高概率回归
        elif abs(current_z_score) > 1.8:
            prob = 0.7  # 显著偏离，较高概率回归
        elif abs(current_z_score) > 1.0:
            prob = 0.55  # 中等偏离，中等概率回归
        else:
            prob = 0.3  # 轻微偏离，低概率回归

        # 考虑趋势因素调整概率
        if trend_strength and trend_strength["level"] != "无明显趋势":
            trend_factor = 0.2 if trend_strength["level"] == "强趋势" else 0.1

            # 如果偏离方向与趋势方向一致，降低回归概率；反之则提高
            if (current_z_score > 0 and trend_strength["direction"] == "上升") or \
                    (current_z_score < 0 and trend_strength["direction"] == "下降"):
                prob -= trend_factor
            else:
                prob += trend_factor

        # 确保概率在有效范围内
        mean_reversion_probability = max(0.1, min(0.95, prob))

    # 13. 新增：分析在周期中的位置
    cycle_position = None
    if percentile is not None and trend_strength:
        if percentile > 0.85:
            position = "顶部区域"
        elif percentile > 0.65:
            position = "上升区域"
        elif percentile > 0.35:
            position = "中间区域"
        elif percentile > 0.15:
            position = "下降区域"
        else:
            position = "底部区域"

        # 结合趋势方向
        if trend_strength["direction"] == "上升":
            if position in ["底部区域", "下降区域"]:
                cycle_status = "可能开始新一轮上升周期"
            elif position in ["中间区域"]:
                cycle_status = "处于上升周期中段"
            else:
                cycle_status = "上升周期接近尾声"
        elif trend_strength["direction"] == "下降":
            if position in ["顶部区域", "上升区域"]:
                cycle_status = "可能开始新一轮下降周期"
            elif position in ["中间区域"]:
                cycle_status = "处于下降周期中段"
            else:
                cycle_status = "下降周期接近尾声"
        else:
            cycle_status = "当前处于盘整阶段"

        cycle_position = {
            "position": position,
            "status": cycle_status
        }

    # 14. 生成综合推荐建议
    recommendation = ""

    # 基于当前Z分数的极端值判断
    if is_extreme and current_z_score is not None:
        if current_z_score > 0:
            recommendation = f"当前比值处于历史高位(Z值:{current_z_score:.2f})，建议关注做空套利机会（卖出股票A或买入股票B）。"

            # 添加均值回归概率信息
            if mean_reversion_probability:
                recommendation += f" 均值回归概率约为{mean_reversion_probability:.0%}。"

            # 添加支撑位信息
            if support_resistance:
                recommendation += f" 注意防守{support_resistance['nearby_support']}的支撑位。"
        else:
            recommendation = f"当前比值处于历史低位(Z值:{current_z_score:.2f})，建议关注做多套利机会（买入股票A或卖出股票B）。"

            # 添加均值回归概率信息
            if mean_reversion_probability:
                recommendation += f" 均值回归概率约为{mean_reversion_probability:.0%}。"

            # 添加阻力位信息
            if support_resistance:
                recommendation += f" 上方{support_resistance['nearby_resistance']}可能存在阻力。"

    # 考虑趋势与当前位置的综合分析
    elif trend_strength and trend_strength["level"] != "无明显趋势":
        if trend_strength["direction"] == "上升" and deviation_from_trend and deviation_from_trend < -5:
            recommendation = f"当前比值低于上升趋势线，可能存在回归机会。考虑买入股票A或卖出股票B。价格比值处于{percentile:.0%}百分位水平。"

            # 添加支撑阻力位信息
            if support_resistance:
                recommendation += f" 近期支撑位:{support_resistance['nearby_support']}，阻力位:{support_resistance['nearby_resistance']}。"
        elif trend_strength["direction"] == "下降" and deviation_from_trend and deviation_from_trend > 5:
            recommendation = f"当前比值高于下降趋势线，可能存在回归机会。考虑卖出股票A或买入股票B。价格比值处于{percentile:.0%}百分位水平。"

            # 添加支撑阻力位信息
            if support_resistance:
                recommendation += f" 近期支撑位:{support_resistance['nearby_support']}，阻力位:{support_resistance['nearby_resistance']}。"
        elif abs(deviation_from_trend or 0) < 3:
            recommendation = f"当前比值贴近{trend_strength['level']}的{trend_strength['direction']}趋势线，建议顺势操作或观望。"

            # 添加周期位置信息
            if cycle_position:
                recommendation += f" {cycle_position['status']}。"
        else:
            recommendation = f"当前处于{trend_strength['level']}的{trend_strength['direction']}趋势中，"

            if trend_strength["direction"] == "上升":
                recommendation += f"整体偏多操作为宜。价格比值处于{percentile:.0%}百分位水平。"
            else:
                recommendation += f"整体偏空操作为宜。价格比值处于{percentile:.0%}百分位水平。"

    # 与历史信号的比较分析
    elif similarity_score and similarity_score > 0.8 and nearest_signal_id is not None:
        nearest_signal = next((s for s in signals if s["id"] == nearest_signal_id), None)
        if nearest_signal:
            signal_desc = "超买" if nearest_signal["type"] == "positive" else "超卖"
            strength_desc = {"weak": "弱", "medium": "中等", "strong": "强"}[nearest_signal["strength"]]
            recommendation = f"当前比值与历史{nearest_signal['date']}的{signal_desc}信号高度相似(相似度:{similarity_score:.2f})，当时为{strength_desc}信号。参考历史表现，"

            # 检查该信号的历史表现
            similar_signal_record = None
            for signal in signals:
                if signal["id"] == nearest_signal_id and "record_id" in signal:
                    # 可能需要加载信号记录来获取更详细的历史表现
                    # 这里可以简化为直接使用当前信号的类型来推断
                    if signal["type"] == "positive":
                        recommendation += "可能面临价格回落。建议卖出股票A或买入股票B。"
                    else:
                        recommendation += "可能面临价格反弹。建议买入股票A或卖出股票B。"
                    break

            # 如果无法获取历史表现，给出通用建议
            if "可能面临" not in recommendation:
                recommendation += "注意关注后续价格走势的相似性。"

    # 考虑历史模式
    elif historical_signal_pattern:
        if historical_signal_pattern == "连续超买" and percentile and percentile > 0.7:
            recommendation = "近期历史信号显示连续超买状态，当前比值位于较高位置，建议保持谨慎。"

            # 添加支撑位信息
            if support_resistance:
                recommendation += f" 若跌破{support_resistance['nearby_support']}支撑位，可能加速下跌。"
        elif historical_signal_pattern == "连续超卖" and percentile and percentile < 0.3:
            recommendation = "近期历史信号显示连续超卖状态，当前比值位于较低位置，可能存在机会。"

            # 添加阻力位信息
            if support_resistance:
                recommendation += f" 突破{support_resistance['nearby_resistance']}阻力位后，可能加速上涨。"
        elif historical_signal_pattern == "震荡切换":
            recommendation = "近期市场处于震荡状态，建议关注价格比值突破重要阈值的情况。"

            # 添加支撑阻力区间
            if support_resistance:
                recommendation += f" 当前交易区间可能在{support_resistance['nearby_support']}至{support_resistance['nearby_resistance']}之间。"

    # 没有明显信号时，基于周期位置和支撑阻力给出建议
    elif cycle_position:
        recommendation = f"当前比值在历史分布中处于{cycle_position['position']}，{cycle_position['status']}。"

        if support_resistance:
            recommendation += f" 近期支撑位:{support_resistance['nearby_support']}，阻力位:{support_resistance['nearby_resistance']}。"

        # 添加百分位信息
        if percentile is not None:
            recommendation += f" 价格比值处于{percentile:.0%}百分位水平。"

    # 最后的兜底建议
    else:
        recommendation = "当前比值在历史正常范围内，暂无明显异常。建议继续观察市场动态。"

        # 添加百分位信息
        if percentile is not None:
            recommendation += f" 价格比值处于{percentile:.0%}百分位水平。"

    # 添加波动性提示
    if volatility_level == "high":
        recommendation += " 注意：当前波动性较高，交易需谨慎。"
    elif volatility_level == "low" and trend_strength and trend_strength["level"] != "无明显趋势":
        recommendation += f" 当前波动性较低，可能即将迎来波动性扩大，注意{trend_strength['direction']}趋势延续性。"

    return {
        "current_ratio": float(current_ratio) if current_ratio is not None else 0,
        "nearest_signals": similar_signals,
        "similarity_score": float(similarity_score) if similarity_score is not None else None,
        "percentile": float(percentile) if percentile is not None else None,
        "deviation_from_trend": float(deviation_from_trend) if deviation_from_trend is not None else None,
        "volatility_level": volatility_level,
        "is_extreme": bool(is_extreme),
        "z_score": float(current_z_score) if current_z_score is not None else None,
        "historical_signal_pattern": historical_signal_pattern,
        "trend_strength": trend_strength,
        "support_resistance": support_resistance,
        "mean_reversion_probability": float(
            mean_reversion_probability) if mean_reversion_probability is not None else None,
        "cycle_position": cycle_position,
        "recommendation": recommendation
    }
