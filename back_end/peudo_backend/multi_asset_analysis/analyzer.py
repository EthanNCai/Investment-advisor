import numpy as np
from typing import List, Dict, Tuple, Any
import pandas as pd
from k_chart_fetcher import k_chart_fetcher, get_stock_data_pair, date_alignment, durations
from indicators.technical_indicators import calculate_price_ratio_anomaly
from get_stock_data.stock_trends_base import StockTrendsDatabase
from .utils import process_ratio_by_kline_type


def calculate_correlation_matrix(assets: List[str], asset_names: List[str],
                                 duration: str, kline_type: str,
                                 polynomial_degree: int,
                                 threshold_multiplier: float) -> Dict[str, Any]:
    """
    计算多资产间的相关性矩阵
    
    Args:
        assets: 资产代码列表
        asset_names: 资产名称列表
        duration: 时间跨度
        kline_type: K线类型
        polynomial_degree: 多项式拟合次数
        threshold_multiplier: 阈值倍数
    
    Returns:
        Dict: 包含相关性分析结果的字典
    """
    n = len(assets)
    relations_matrix = np.zeros((n, n))

    # 所有资产对的价格比值
    all_pair_ratios = {}

    # 计算资产对之间的关系
    for i in range(n):
        for j in range(n):
            if i == j:
                relations_matrix[i][j] = 0
                continue

            # 获取两个资产的价格比值
            result = k_chart_fetcher(
                code_a=assets[i],
                code_b=assets[j],
                duration_in=duration,
                degree=polynomial_degree
            )

            # 根据K线类型处理结果
            if kline_type != 'daily':
                result = process_ratio_by_kline_type(result, kline_type)

            # 保存中间结果以便复用
            pair_key = f"{assets[i]}_{assets[j]}"
            all_pair_ratios[pair_key] = {
                "ratio": result["ratio"],
                "dates": result["dates"],
                "fitting_line": result["fitting_line"],
                "delta": result["delta"],
                "threshold": result["threshold"],
                "close_a": result["close_a"],
                "close_b": result["close_b"]
            }

            # 计算异常值分析
            ratio_data = result["ratio"]
            delta_data = result["delta"]
            threshold = result["threshold"]

            anomaly_result = calculate_price_ratio_anomaly(
                ratio_data=ratio_data,
                delta_data=delta_data,
                threshold_multiplier=threshold_multiplier,
                std_value=threshold
            )

            # 获取异常点列表
            anomalies = anomaly_result.get("anomalies", [])

            if not anomalies or len(ratio_data) == 0:
                # 没有异常点，设置为0
                relations_matrix[i][j] = 0
                continue

            # 获取当前价格比值
            current_ratio = get_latest_price_ratio(assets[i], assets[j])
            if current_ratio is None:
                current_ratio = ratio_data[-1]

            # 计算拟合线的延伸值
            # 使用多项式拟合获得的系数
            x = np.arange(len(ratio_data))
            coefficients = np.polyfit(x, ratio_data, polynomial_degree)
            poly = np.poly1d(coefficients)
            # 计算历史百分位
            sorted_ratios = sorted(ratio_data)
            rank = sum(1 for r in sorted_ratios if r <= current_ratio)
            historical_percentile = (rank / len(sorted_ratios)) * 100

            # 当前点位于时间序列的末尾
            current_x = len(ratio_data) - 1
            current_fitting = poly(current_x)
            current_delta = current_ratio - current_fitting

            # 如果当前差值为0或极小，则无信号
            if abs(current_delta) < threshold * 0.5:
                relations_matrix[i][j] = 0
                continue

            green_points = []  # 绿色区域点（比值高于拟合线，差值为正）
            red_points = []  # 红色区域点（比值低于拟合线，差值为负）

            for anomaly in anomalies:
                idx = anomaly["index"]
                if idx < len(ratio_data) and idx < len(result["fitting_line"]):
                    diff = ratio_data[idx] - result["fitting_line"][idx]
                    z_score = anomaly.get("z_score_mark", 0)

                    if diff > 0:
                        green_points.append({"diff": diff, "z_score": z_score})
                    else:
                        red_points.append({"diff": diff, "z_score": z_score})

            # 使用差值百分位法计算信号强度
            signal_strength = 0

            if current_delta > 0:  # 当前在绿色区域（比值高于拟合线）
                if green_points:
                    # 计算绿色区域点差值范围
                    min_green = min(point["diff"] for point in green_points)
                    max_green = max(point["diff"] for point in green_points)

                    if max_green > min_green and min_green <= current_delta <= max_green:
                        # 计算当前差值在绿色区域点中的百分位
                        strength_pct = (current_delta - min_green) / (max_green - min_green) * 100
                        # 绿色区域表示做空第一个资产，做多第二个资产（负信号强度）
                        if historical_percentile > 75:
                            signal_strength = -1 * (min(max(strength_pct, 0), 100) + historical_percentile * 0.55)
                        else:
                            signal_strength = -1 * min(max(strength_pct, 0), 100)
                    else:
                        print("设置了中等强度在绿色区域，但当前点位于绿色区域外")
                        signal_strength = -15  # 赋予很弱强度
            else:  # 当前在红色区域（比值低于拟合线）
                if red_points:
                    # 计算红色区域点差值范围
                    min_red = min(point["diff"] for point in red_points)  # 最小的负差值，绝对值最大
                    max_red = max(point["diff"] for point in red_points)  # 最大的负差值，绝对值最小

                    if min_red < max_red and abs(max_red) <= abs(current_delta) <= abs(min_red):
                        # 计算当前差值在红色区域点中的百分位
                        # 红色区域点越小（绝对值越大），信号越强
                        strength_pct = (abs(current_delta) - abs(max_red)) / (abs(min_red) - abs(max_red)) * 100
                        # 红色区域表示做多第一个资产，做空第二个资产（正信号强度）
                        if historical_percentile < 25:
                            signal_strength = min(max(strength_pct, 0), 100) + (25 - historical_percentile) * 3
                        else:
                            signal_strength = min(max(strength_pct, 0), 100)
                    else:
                        print("设置了中等强度在红色区域，但当前点位于红色区域外")
                        signal_strength = 15

                        # 根据信号强度分级
            if abs(signal_strength) < 20:
                signal_strength = 0  # 极弱信号视为无信号

            # 保存到矩阵中
            relations_matrix[i][j] = round(signal_strength, 1)

    # 计算每个资产的综合强弱得分
    strength_scores = np.sum(relations_matrix, axis=1)

    # 归一化强弱得分到 -100 到 100 之间
    # max_abs_score = max(abs(np.max(strength_scores)), abs(np.min(strength_scores)))
    # if max_abs_score > 0:
    #     normalized_scores = (strength_scores / max_abs_score) * 100
    # else:
    #     normalized_scores = strength_scores
    strength_scores = np.clip(strength_scores, -100, 100)
    # 创建资产强弱排名
    asset_strength_ranking = []
    for i in range(n):
        asset_strength_ranking.append({
            "assetIndex": i,
            "score": round(float(strength_scores[i]), 1)
        })

    # 按得分从高到低排序
    asset_strength_ranking = sorted(asset_strength_ranking, key=lambda x: x["score"], reverse=True)

    return {
        "assets": assets,
        "assetNames": asset_names,
        "relationsMatrix": relations_matrix.tolist(),
        "assetStrengthRanking": asset_strength_ranking,
        "allPairRatios": all_pair_ratios
    }


def find_optimal_trading_pairs(correlation_matrix: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从相关性矩阵中找出最有交易价值的资产对
    
    Args:
        correlation_matrix: 资产相关性矩阵结果
        
    Returns:
        List: 最优交易对列表
    """
    assets = correlation_matrix["assets"]
    asset_names = correlation_matrix["assetNames"]
    relations_matrix = correlation_matrix["relationsMatrix"]

    # 提取所有非零的资产对交易信号
    pairs = []
    n = len(assets)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            signal_strength = relations_matrix[i][j]
            if abs(signal_strength) >= 20:  # 只考虑信号强度达到一定阈值的对
                # 确定交易方向
                direction = 1 if signal_strength > 0 else -1

                # 估算预期收益 (简化模型)
                expected_return = abs(signal_strength) / 5  # 简单假设信号强度每5分对应1%预期收益

                pairs.append({
                    "assetAIndex": i,
                    "assetBIndex": j,
                    "signalStrength": signal_strength,
                    "direction": direction,
                    "expectedReturn": round(expected_return, 2)
                })

    # 按信号强度绝对值从高到低排序
    pairs = sorted(pairs, key=lambda x: abs(x["signalStrength"]), reverse=True)

    # 限制返回的对数
    max_pairs = min(25, len(pairs))
    return pairs[:max_pairs]


def get_asset_pair_analysis(code_a: str, code_b: str, duration: str,
                            polynomial_degree: int, threshold_multiplier: float,
                            kline_type: str = "daily") -> Dict[str, Any]:
    """
    获取两个资产间的比值分析结果
    
    Args:
        code_a: 第一个资产代码
        code_b: 第二个资产代码
        duration: 时间跨度
        polynomial_degree: 多项式拟合次数
        threshold_multiplier: 阈值倍数
        kline_type: K线类型，默认为日K
        
    Returns:
        Dict: 包含比值分析结果的字典
    """
    # 获取K线数据和比值
    result = k_chart_fetcher(
        code_a=code_a,
        code_b=code_b,
        duration_in=duration,
        degree=polynomial_degree
    )

    # 根据K线类型处理结果
    if kline_type != 'daily':
        result = process_ratio_by_kline_type(result, kline_type)
    # 提取结果数据
    ratio = result["ratio"]
    dates = result["dates"]
    fitting_line = result["fitting_line"]
    delta = result["delta"]
    threshold = result["threshold"]
    close_a = result["close_a"]
    close_b = result["close_b"]

    # 计算异常值分析
    anomaly_result = calculate_price_ratio_anomaly(
        ratio_data=ratio,
        delta_data=delta,
        threshold_multiplier=threshold_multiplier,
        std_value=threshold
    )

    # 提取异常点信息
    anomalies = anomaly_result.get("anomalies", [])
    is_anomaly = [False] * len(ratio)
    z_scores = [0.0] * len(ratio)

    for anomaly in anomalies:
        idx = anomaly["index"]
        if 0 <= idx < len(ratio):
            is_anomaly[idx] = True
            z_scores[idx] = anomaly["z_score_mark"]

    # 计算阈值上下限
    upper_threshold = [f + threshold_multiplier * threshold for f in fitting_line]
    lower_threshold = [f - threshold_multiplier * threshold for f in fitting_line]

    # 计算最近数据的分析结果
    recent_window = min(90, len(ratio))
    if recent_window > 0:
        current_ratio = get_latest_price_ratio(code_a, code_b)
        if current_ratio is None:
            current_ratio = ratio[-1]

        # 计算90日均值，但保持键名为ma10_ratio
        ma10_ratio = sum(ratio[-recent_window:]) / recent_window

        # 计算历史百分位
        sorted_ratios = sorted(ratio)
        if len(sorted_ratios) > 0:
            rank = sum(1 for r in sorted_ratios if r <= current_ratio)
            historical_percentile = (rank / len(sorted_ratios)) * 100
        else:
            historical_percentile = 50

        # 使用差值百分位法计算信号强度
        # 计算当前拟合线的值
        x = np.arange(len(ratio))
        coefficients = np.polyfit(x, ratio, polynomial_degree)
        poly = np.poly1d(coefficients)
        current_x = len(ratio) - 1  # 当前点位于时间序列的末尾
        current_fitting = poly(current_x)
        current_delta = current_ratio - current_fitting

        # 分析异常点中的绿色点和红色点
        green_points = []  # 绿色区域点（比值高于拟合线，差值为正）
        red_points = []  # 红色区域点（比值低于拟合线，差值为负）

        for anomaly in anomalies:
            idx = anomaly["index"]
            if idx < len(ratio) and idx < len(fitting_line):
                diff = ratio[idx] - fitting_line[idx]
                z_score = anomaly.get("z_score_mark", 0)

                if diff > 0:  # 绿色点
                    green_points.append({"diff": diff, "z_score": z_score})
                else:  # 红色点
                    red_points.append({"diff": diff, "z_score": z_score})

        # 计算信号强度
        signal_strength = 0

        if current_delta > 0:  # 当前在绿色区域（比值高于拟合线）
            if green_points:
                min_green = min(point["diff"] for point in green_points)
                max_green = max(point["diff"] for point in green_points)

                if max_green > min_green and min_green <= current_delta <= max_green:
                    # 计算当前差值在绿色区域点中的百分位
                    strength_pct = (current_delta - min_green) / (max_green - min_green) * 100
                    # 绿色区域表示做空第一个资产，做多第二个资产（负信号强度）
                    if historical_percentile > 75:
                        signal_strength = -1 * (min(max(strength_pct, 0), 100) + historical_percentile * 0.55)
                    else:
                        signal_strength = -1 * min(max(strength_pct, 0), 100)
                else:
                    signal_strength = -15
        else:  # 当前在红色区域（比值低于拟合线）
            if red_points:
                min_red = min(point["diff"] for point in red_points)  # 最小的负差值，绝对值最大
                max_red = max(point["diff"] for point in red_points)  # 最大的负差值，绝对值最小

                if min_red < max_red and abs(max_red) <= abs(current_delta) <= abs(min_red):
                    # 计算当前差值在红色区域点中的百分位
                    # 红色区域点越小（绝对值越大），信号越强
                    strength_pct = (abs(current_delta) - abs(max_red)) / (abs(min_red) - abs(max_red)) * 100
                    # 红色区域表示做多第一个资产，做空第二个资产（正信号强度）
                    if historical_percentile < 25:
                        signal_strength = min(max(strength_pct, 0), 100) + (25 - historical_percentile) * 3
                    else:
                        signal_strength = min(max(strength_pct, 0), 100)
                else:
                    signal_strength = 15

        # 根据信号强度分级，如果信号强度太弱则视为观望
        if abs(signal_strength) < 20:
            signal_strength = 0
            recommendation = "观望"
        else:
            # 根据信号强度生成交易建议
            if signal_strength > 0:
                recommendation = f"做多{code_a}做空{code_b}"
            else:
                recommendation = f"做空{code_a}做多{code_b}"

            # 根据强度添加修饰语
            if abs(signal_strength) >= 80:
                recommendation = "强烈建议" + recommendation
            elif abs(signal_strength) >= 60:
                recommendation = "建议" + recommendation
            elif abs(signal_strength) >= 40:
                recommendation = "可考虑" + recommendation

    else:
        # 数据太少，无法提供分析
        current_ratio = 0
        ma10_ratio = 0
        historical_percentile = 50
        signal_strength = 0
        recommendation = "数据不足，无法分析"

    # 整合分析结果
    analysis = {
        "current_ratio": current_ratio,
        "ma10_ratio": ma10_ratio,
        "historical_percentile": historical_percentile,
        "signal_strength": round(signal_strength, 1),
        "recommendation": recommendation
    }

    # 组合最终结果
    return {
        "code_a": code_a,
        "code_b": code_b,
        "dates": dates,
        "ratio": ratio,
        "fitted_curve": fitting_line,
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "anomaly_info": {
            "is_anomaly": is_anomaly,
            "z_scores": z_scores
        },
        "analysis": analysis
    }


# 获取最新价格比值
def get_latest_price_ratio(code_a: str, code_b: str):
    try:
        # 获取两只股票最新的价格趋势数据
        db = StockTrendsDatabase()
        trends_a = db.query_trends(stock_code=code_a)
        trends_b = db.query_trends(stock_code=code_b)

        # 确保有趋势数据
        if trends_a and trends_b:
            # 找到时间最接近的一对数据点
            latest_time_a = max(trends_a, key=lambda x: x['date'])['date']
            latest_time_b = max(trends_b, key=lambda x: x['date'])['date']
            current_ratio = None
            # 获取最新价格
            latest_price_a = next((t['current_price'] for t in trends_a if t['date'] == latest_time_a), None)
            latest_price_b = next((t['current_price'] for t in trends_b if t['date'] == latest_time_b), None)

            # 如果两者都有最新价格，计算实时比值
            if latest_price_a is not None and latest_price_b is not None and latest_price_b != 0:
                current_ratio = float(latest_price_a) / float(latest_price_b)
                return current_ratio

    except Exception as e:
        print(f"获取实时价格数据失败: {e}")
        return None


# 获取所有资产对的比值图表数据
def get_all_pair_charts(assets: List[str], duration: str, polynomial_degree: int,
                        threshold_multiplier: float, kline_type: str = "daily") -> Dict[str, Any]:
    """
    获取所有资产对的比值图表数据
    
    Args:
        assets: 资产代码列表
        duration: 时间跨度
        polynomial_degree: 多项式拟合次数
        threshold_multiplier: 阈值倍数
        kline_type: K线类型，默认为日K
        
    Returns:
        Dict: 包含所有资产对比值图表数据的字典
    """
    n = len(assets)
    all_charts = {}

    for i in range(n):
        for j in range(i + 1, n):  # 只计算一次每对资产
            code_a = assets[i]
            code_b = assets[j]

            pair_key = f"{code_a}_{code_b}"

            # 获取比值数据
            try:
                # 获取K线数据和比值
                result = k_chart_fetcher(
                    code_a=code_a,
                    code_b=code_b,
                    duration_in=duration,
                    degree=polynomial_degree
                )

                # 根据K线类型处理结果
                if kline_type != 'daily':
                    result = process_ratio_by_kline_type(result, kline_type)

                # 提取基本数据
                ratio = result["ratio"]
                dates = result["dates"]
                fitting_line = result["fitting_line"]
                delta = result["delta"]
                threshold = result["threshold"]

                # 计算异常值分析
                anomaly_result = calculate_price_ratio_anomaly(
                    ratio_data=ratio,
                    delta_data=delta,
                    threshold_multiplier=threshold_multiplier,
                    std_value=threshold
                )

                # 提取异常点信息
                anomalies = anomaly_result.get("anomalies", [])
                green_points = []  # 绿色区域点（比值高于拟合线，差值为正）
                red_points = []  # 红色区域点（比值低于拟合线，差值为负）

                for anomaly in anomalies:
                    idx = anomaly["index"]
                    if idx < len(ratio) and idx < len(fitting_line):
                        diff = ratio[idx] - fitting_line[idx]

                        if diff > 0:
                            green_points.append({
                                "date": dates[idx],
                                "value": ratio[idx]
                            })
                        else:
                            red_points.append({
                                "date": dates[idx],
                                "value": ratio[idx]
                            })

                # 获取当前比值
                current_ratio = get_latest_price_ratio(code_a, code_b)
                if current_ratio is None:
                    current_ratio = ratio[-1]

                # 简化的图表数据
                chart_data = {
                    "code_a": code_a,
                    "code_b": code_b,
                    "dates": dates,
                    "ratio": ratio,
                    "fitting_line": fitting_line,
                    "green_points": green_points,
                    "red_points": red_points,
                    "current_ratio": current_ratio
                }

                all_charts[pair_key] = chart_data

            except Exception as e:
                print(f"获取资产对 {code_a}/{code_b} 的图表数据失败: {e}")
                continue

    return all_charts
