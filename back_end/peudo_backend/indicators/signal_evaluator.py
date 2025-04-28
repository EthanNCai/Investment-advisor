"""
投资信号质量评分与追踪验证模块
"""
import json
import os
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

# 信号评估结果缓存文件
SIGNAL_RECORDS_FILE = "model_cache/signal_records.json"


def init_signal_records_file():
    """初始化信号记录文件"""
    if not os.path.exists("model_cache"):
        os.makedirs("model_cache")
    
    if not os.path.exists(SIGNAL_RECORDS_FILE):
        with open(SIGNAL_RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump({"signal_records": []}, f, ensure_ascii=False)


def load_signal_records() -> List[Dict[str, Any]]:
    """加载历史信号记录"""
    init_signal_records_file()
    
    try:
        with open(SIGNAL_RECORDS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("signal_records", [])
    except Exception as e:
        print(f"加载信号记录失败: {e}")
        return []


def save_signal_records(records: List[Dict[str, Any]]):
    """保存信号记录"""
    init_signal_records_file()
    
    try:
        with open(SIGNAL_RECORDS_FILE, "w", encoding="utf-8") as f:
            json.dump({"signal_records": records}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存信号记录失败: {e}")


def evaluate_signal_quality(
    signal: Dict[str, Any],
    code_a: str,
    code_b: str,
    historical_records: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    评估信号质量并计算可信度评分
    
    参数:
        signal: 投资信号数据
        code_a: 股票A代码
        code_b: 股票B代码
        historical_records: 历史信号记录数据，如果为None则自动加载
        
    返回:
        信号质量评估结果，包含:
        - quality_score: 信号质量得分(0-100)
        - confidence_level: 信号可信度级别(low, medium, high)
        - historical_accuracy: 历史准确率
        - expected_return: 预期收益率
        - risk_ratio: 风险比率
        - risk_return_ratio: 风险收益比
        - factors: 影响因素明细
    """
    if historical_records is None:
        historical_records = load_signal_records()
    
    # 1. 计算历史信号准确率
    historical_accuracy = calculate_historical_accuracy(
        signal["type"], 
        code_a, 
        code_b, 
        historical_records
    )
    
    # 2. 根据Z值评估信号强度因子(0-30分)
    z_score = abs(signal["z_score"])
    strength_factor = min(30, z_score * 10)
    
    # 3. 评估市场条件因子(0-25分)
    market_factor = evaluate_market_condition(signal, code_a, code_b)
    
    # 4. 评估信号一致性因子(0-25分)
    consistency_factor = evaluate_signal_consistency(signal, code_a, code_b, historical_records)
    
    # 5. 评估技术指标支持因子(0-20分)
    technical_factor = evaluate_technical_support(signal, code_a, code_b)
    
    # 6. 计算综合评分(0-100分)
    quality_score = strength_factor + market_factor + consistency_factor + technical_factor
    
    # 7. 确定可信度级别
    confidence_level = "low"
    if quality_score >= 80:
        confidence_level = "high"
    elif quality_score >= 60:
        confidence_level = "medium"
    
    # 8. 计算预期收益率和风险比率
    expected_return = calculate_expected_return(signal, historical_records)
    risk_ratio = calculate_risk_ratio(signal, historical_records)
    
    # 确保风险收益比不会除以零
    risk_return_ratio = expected_return / max(0.01, risk_ratio) if risk_ratio > 0 else 0
    
    # 收集各因素得分明细
    factors = {
        "strength": float(strength_factor),
        "market_condition": float(market_factor),
        "consistency": float(consistency_factor),
        "technical_support": float(technical_factor)
    }
    
    return {
        "quality_score": float(round(quality_score, 2)),
        "confidence_level": confidence_level,
        "historical_accuracy": float(round(historical_accuracy * 100, 2)) if historical_accuracy is not None else None,
        "expected_return": float(round(expected_return * 100, 2)),
        "risk_ratio": float(round(risk_ratio * 100, 2)),
        "risk_return_ratio": float(round(risk_return_ratio, 2)),
        "factors": factors
    }


def calculate_historical_accuracy(
    signal_type: str,
    code_a: str,
    code_b: str,
    historical_records: List[Dict[str, Any]]
) -> Optional[float]:
    """
    计算历史信号准确率
    
    参数:
        signal_type: 信号类型 (positive/negative)
        code_a: 股票A代码
        code_b: 股票B代码
        historical_records: 历史信号记录
        
    返回:
        历史准确率(0-1)，如果没有足够数据则返回None
    """
    # 过滤相关股票对的历史记录
    relevant_records = [
        r for r in historical_records 
        if r["code_a"] == code_a and r["code_b"] == code_b and r["signal_type"] == signal_type
    ]
    
    # 如果记录少于3条则认为数据不足
    if len(relevant_records) < 3:
        return None
    
    # 计算成功比例
    successful_records = [r for r in relevant_records if r["was_profitable"]]
    accuracy = len(successful_records) / len(relevant_records) if relevant_records else 0
    
    return float(accuracy)


def evaluate_market_condition(
    signal: Dict[str, Any],
    code_a: str,
    code_b: str
) -> float:
    """
    评估市场条件因子(0-25分)
    
    参数:
        signal: 投资信号
        code_a: 股票A代码
        code_b: 股票B代码
        
    返回:
        市场条件评分(0-25)
    """
    # 这里可以根据实际情况进行市场评估
    # 在实际应用中，可以引入更多外部数据源如大盘行情、行业指数等
    
    # 简化版本：基于信号强度和Z值波动评估
    score = 0
    
    # 1. 信号强度影响(0-10分)
    if signal["strength"] == "strong":
        score += 10
    elif signal["strength"] == "medium":
        score += 7
    else:
        score += 4
    
    # 2. Z值稳定性影响(0-15分)
    z_score = abs(signal["z_score"])
    if 1.5 <= z_score <= 2.5:
        # 适中的Z值最有可能回归均值
        z_score_factor = 15
    elif z_score > 2.5:
        # 极端值可能是异常，但也可能意味着强趋势
        z_score_factor = 8
    else:
        # 较小的Z值可能信号不明显
        z_score_factor = 5
    
    score += z_score_factor
    
    return float(min(25, score))


def evaluate_signal_consistency(
    signal: Dict[str, Any],
    code_a: str,
    code_b: str,
    historical_records: List[Dict[str, Any]]
) -> float:
    """
    评估信号一致性因子(0-25分)
    
    参数:
        signal: 当前信号
        code_a: 股票A代码
        code_b: 股票B代码
        historical_records: 历史信号记录
        
    返回:
        信号一致性评分(0-25)
    """
    # 查找近期相同类型的信号
    recent_signals = [
        r for r in historical_records 
        if r["code_a"] == code_a and r["code_b"] == code_b
    ]
    
    # 如果没有历史记录则返回中等分数
    if not recent_signals:
        return 15.0
    
    # 按日期排序
    recent_signals.sort(key=lambda x: x["signal_date"], reverse=True)
    
    # 提取最近5个信号记录
    recent_signals = recent_signals[:5]
    
    # 计算与当前信号类型相同的比例
    same_type_count = sum(1 for r in recent_signals if r["signal_type"] == signal["type"])
    type_consistency = same_type_count / len(recent_signals) if recent_signals else 0
    
    # 如果信号与最近趋势一致性高，得分高
    consistency_score = type_consistency * 25
    
    return float(consistency_score)


def evaluate_technical_support(
    signal: Dict[str, Any],
    code_a: str,
    code_b: str
) -> float:
    """
    评估技术指标支持因子(0-20分)
    
    参数:
        signal: 投资信号
        code_a: 股票A代码
        code_b: 股票B代码
        
    返回:
        技术指标支持评分(0-20)
    """
    # 简化版实现，实际应用中可以添加更多技术指标判断
    # 例如MACD、RSI、布林带等指标的验证
    
    # 基础分数10分
    base_score = 10
    
    # 通过信号强度调整
    if signal["strength"] == "strong":
        strength_adjust = 10
    elif signal["strength"] == "medium":
        strength_adjust = 7
    else:
        strength_adjust = 3
    
    technical_score = base_score + strength_adjust
    
    return float(min(20, technical_score))


def calculate_expected_return(
    signal: Dict[str, Any],
    historical_records: List[Dict[str, Any]]
) -> float:
    """
    计算预期收益率
    
    参数:
        signal: 投资信号
        historical_records: 历史信号记录
        
    返回:
        预期收益率(小数表示)
    """
    # 过滤相似的历史信号
    similar_signals = [
        r for r in historical_records 
        if r["signal_type"] == signal["type"] and r["signal_strength"] == signal["strength"]
    ]
    
    # 如果没有足够的历史数据，使用默认值
    if len(similar_signals) < 3:
        # 根据信号强度设置默认预期收益
        if signal["strength"] == "strong":
            return 0.05  # 5%
        elif signal["strength"] == "medium":
            return 0.03  # 3%
        else:
            return 0.015  # 1.5%
    
    # 计算历史信号的平均收益率
    total_return = sum(r["actual_return"] for r in similar_signals if "actual_return" in r)
    avg_return = total_return / len(similar_signals)
    
    # 应用信号特定调整
    if signal["strength"] == "strong":
        # 强信号可能有更高收益
        return float(avg_return * 1.2)
    elif signal["strength"] == "medium":
        return float(avg_return)
    else:
        # 弱信号可能收益较低
        return float(avg_return * 0.8)


def calculate_risk_ratio(
    signal: Dict[str, Any],
    historical_records: List[Dict[str, Any]]
) -> float:
    """
    计算风险比率
    
    参数:
        signal: 投资信号
        historical_records: 历史信号记录
        
    返回:
        风险比率(小数表示)
    """
    # 过滤相似的历史信号
    similar_signals = [
        r for r in historical_records 
        if r["signal_type"] == signal["type"] and r["signal_strength"] == signal["strength"]
    ]
    
    # 如果没有足够的历史数据，使用默认值
    if len(similar_signals) < 3:
        # 根据信号强度设置默认风险
        if signal["strength"] == "strong":
            return 0.03  # 3%
        elif signal["strength"] == "medium":
            return 0.02  # 2%
        else:
            return 0.01  # 1%
    
    # 计算历史信号的最大回撤平均值作为风险指标
    total_risk = sum(r.get("max_drawdown", 0) for r in similar_signals)
    avg_risk = total_risk / len(similar_signals)
    
    # 应用信号特定调整
    if signal["strength"] == "strong":
        # 强信号风险可能更高
        return float(avg_risk * 1.1)
    elif signal["strength"] == "weak":
        # 弱信号风险可能更低
        return float(avg_risk * 0.9)
    else:
        return float(avg_risk)


def record_new_signal(
    signal: Dict[str, Any],
    code_a: str,
    code_b: str,
    evaluation: Dict[str, Any]
) -> Dict[str, Any]:
    """
    记录新的信号并保存到历史记录中
    
    参数:
        signal: 投资信号
        code_a: 股票A代码
        code_b: 股票B代码
        evaluation: 信号质量评估结果
        
    返回:
        带有唯一ID的信号记录
    """
    # 加载现有记录
    records = load_signal_records()
    
    # 查找是否已存在完全相同的信号记录（相同股票对、相同日期、相同信号ID）
    existing_record = None
    for record in records:
        if (record["code_a"] == code_a and 
            record["code_b"] == code_b and 
            record["signal_date"] == signal["date"] and
            record["signal_id"] == signal["id"]):
            existing_record = record
            break
    
    # 如果找到完全相同的记录，更新它而不是创建新记录
    if existing_record:
        signal_id = existing_record["record_id"]
    else:
        # 生成新的唯一ID
        signal_id = len(records) + 1
    
    # 创建新记录
    signal_record = {
        "record_id": int(signal_id),
        "signal_id": signal["id"],
        "code_a": code_a,
        "code_b": code_b,
        "signal_date": signal["date"],
        "signal_type": signal["type"],
        "signal_strength": signal["strength"],
        "z_score": float(signal["z_score"]),
        "quality_score": float(evaluation["quality_score"]),
        "confidence_level": evaluation["confidence_level"],
        "expected_return": float(evaluation["expected_return"]),
        "risk_ratio": float(evaluation["risk_ratio"]),
        "risk_return_ratio": float(evaluation["risk_return_ratio"]),
        "creation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "active",
        "was_profitable": None,
        "actual_return": None,
        "max_drawdown": None,
        "validation_completed": False,
        "followup_data": []
    }
    
    # 如果找到现有记录，就替换它；否则添加新记录
    if existing_record:
        for i, record in enumerate(records):
            if record["record_id"] == signal_id:
                records[i] = signal_record
                break
    else:
        records.append(signal_record)
    
    # 保存记录
    save_signal_records(records)
    
    return signal_record


def update_signal_performance(
    record_id: int,
    close_a: List[float],
    close_b: List[float],
    dates: List[str],
    days_to_track: int = 30
) -> Dict[str, Any]:
    """
    更新信号表现跟踪数据
    
    参数:
        record_id: 信号记录ID
        close_a: 股票A收盘价列表
        close_b: 股票B收盘价列表
        dates: 日期列表
        days_to_track: 跟踪天数
        
    返回:
        更新后的信号记录
    """
    # 加载记录
    records = load_signal_records()
    
    # 查找特定记录
    target_record = None
    for record in records:
        if record["record_id"] == record_id:
            target_record = record
            break
    
    if not target_record:
        return None
    
    # 查找信号日期在历史数据中的位置
    signal_date = target_record["signal_date"]
    if signal_date not in dates:
        return target_record
    
    signal_idx = dates.index(signal_date)
    
    # 如果信号是最新的，无法评估表现
    if signal_idx >= len(dates) - 1:
        return target_record
    
    # 计算跟踪开始和结束的位置
    start_idx = signal_idx
    end_idx = min(signal_idx + days_to_track, len(dates) - 1)
    
    # 如果数据不足，提前结束
    if end_idx <= start_idx:
        return target_record
    
    # 计算跟踪期内的收益率
    ratio_at_signal = close_a[start_idx] / close_b[start_idx]
    
    # 初始化跟踪数据
    followup_data = []
    max_drawdown = 0
    max_profit = 0
    min_ratio = ratio_at_signal
    max_ratio = ratio_at_signal
    
    # 信号类型决定预期方向
    expected_direction = 1 if target_record["signal_type"] == "negative" else -1
    
    # 计算每日表现并记录
    for i in range(start_idx + 1, end_idx + 1):
        current_ratio = close_a[i] / close_b[i]
        current_date = dates[i]
        
        # 对于负向信号(比值回升)，比值下降是盈利；对于正向信号(比值回落)，比值上升是盈利
        daily_return = (current_ratio - ratio_at_signal) / ratio_at_signal * expected_direction
        
        # 更新最大盈利和回撤
        if daily_return > max_profit:
            max_profit = daily_return
        if daily_return < max_drawdown:
            max_drawdown = daily_return
        
        # 更新最高/最低比值
        if current_ratio > max_ratio:
            max_ratio = current_ratio
        if current_ratio < min_ratio:
            min_ratio = current_ratio
        
        # 记录每日表现
        followup_data.append({
            "date": current_date,
            "ratio": current_ratio,
            "return": daily_return * 100  # 转为百分比
        })
    
    # 最终收益率
    final_return = followup_data[-1]["return"] / 100 if followup_data else 0
    
    # 确定信号是否盈利
    was_profitable = final_return > 0
    
    # 更新记录
    target_record["was_profitable"] = bool(was_profitable)
    target_record["actual_return"] = float(final_return)
    target_record["max_drawdown"] = float(abs(max_drawdown))
    target_record["validation_completed"] = True
    target_record["followup_data"] = followup_data
    target_record["days_tracked"] = int(len(followup_data))
    
    # 保存更新后的记录
    save_signal_records(records)
    
    return target_record


def get_signal_performance_stats(code_a: str = None, code_b: str = None) -> Dict[str, Any]:
    """
    获取信号表现统计数据
    
    参数:
        code_a: 可选，股票A代码，用于筛选特定股票对
        code_b: 可选，股票B代码，用于筛选特定股票对
    
    返回:
        信号表现统计报告
    """
    records = load_signal_records()
    
    # 如果指定了股票代码，则筛选相关记录
    if code_a and code_b:
        records = [r for r in records if r["code_a"] == code_a and r["code_b"] == code_b]
    
    # 过滤已完成验证的记录
    validated_records = [r for r in records if r["validation_completed"]]
    
    if not validated_records:
        return {
            "total_signals": 0,
            "validated_signals": 0,
            "success_rate": 0.000,
            "avg_return": 0.000,
            "avg_risk": 0.000,
            "signal_types": {},
            "confidence_levels": {}
        }
    
    # 计算整体成功率
    successful_signals = [r for r in validated_records if r["was_profitable"]]
    success_rate = float(len(successful_signals) / len(validated_records))
    
    # 计算平均收益和风险
    avg_return = float(sum(r["actual_return"] for r in validated_records) / len(validated_records))
    avg_risk = float(sum(r["max_drawdown"] for r in validated_records) / len(validated_records))
    
    # 按信号类型统计
    signal_types = {}
    for sig_type in ["positive", "negative"]:
        type_records = [r for r in validated_records if r["signal_type"] == sig_type]
        if type_records:
            type_success = [r for r in type_records if r["was_profitable"]]
            type_success_rate = float(len(type_success) / len(type_records))
            type_avg_return = float(sum(r["actual_return"] for r in type_records) / len(type_records))
            signal_types[sig_type] = {
                "count": int(len(type_records)),
                "success_rate": round(type_success_rate, 3),
                "avg_return": round(type_avg_return, 3)
            }
    
    # 按信号可信度级别统计
    confidence_levels = {}
    for level in ["low", "medium", "high"]:
        level_records = [r for r in validated_records if r["confidence_level"] == level]
        if level_records:
            level_success = [r for r in level_records if r["was_profitable"]]
            level_success_rate = float(len(level_success) / len(level_records))
            level_avg_return = float(sum(r["actual_return"] for r in level_records) / len(level_records))
            confidence_levels[level] = {
                "count": int(len(level_records)),
                "success_rate": round(level_success_rate, 3),
                "avg_return": round(level_avg_return, 3)
            }
    
    return {
        "total_signals": int(len(records)),
        "validated_signals": int(len(validated_records)),
        "success_rate": round(success_rate, 3),
        "avg_return": round(avg_return, 3),
        "avg_risk": round(avg_risk, 3),
        "signal_types": signal_types,
        "confidence_levels": confidence_levels
    }


def get_signal_history(
    code_a: str = None,
    code_b: str = None,
    limit: int = 50,
    include_followup: bool = False
) -> List[Dict[str, Any]]:
    """
    获取信号历史记录
    
    参数:
        code_a: 股票A代码(可选)
        code_b: 股票B代码(可选)
        limit: 返回记录数量限制
        include_followup: 是否包含详细跟踪数据
        
    返回:
        信号历史记录列表
    """
    records = load_signal_records()
    
    # 应用过滤器
    if code_a or code_b:
        filtered_records = []
        for record in records:
            if (code_a is None or record["code_a"] == code_a) and \
               (code_b is None or record["code_b"] == code_b):
                filtered_records.append(record)
        records = filtered_records
    
    # 按日期排序
    records.sort(key=lambda x: x["signal_date"], reverse=True)
    
    # 限制返回数量
    records = records[:limit]
    
    # 如果不需要跟踪数据，移除以减少数据量
    if not include_followup:
        for record in records:
            if "followup_data" in record:
                record["followup_data"] = []
    
    # 确保所有NumPy类型转换为Python原生类型
    for record in records:
        if "was_profitable" in record and record["was_profitable"] is not None:
            record["was_profitable"] = bool(record["was_profitable"])
        if "actual_return" in record and record["actual_return"] is not None:
            record["actual_return"] = float(record["actual_return"])
        if "max_drawdown" in record and record["max_drawdown"] is not None:
            record["max_drawdown"] = float(record["max_drawdown"])
        if "days_tracked" in record and record["days_tracked"] is not None:
            record["days_tracked"] = int(record["days_tracked"])
        if "quality_score" in record and record["quality_score"] is not None:
            record["quality_score"] = float(record["quality_score"])
        if "expected_return" in record and record["expected_return"] is not None:
            record["expected_return"] = float(record["expected_return"])
        if "risk_ratio" in record and record["risk_ratio"] is not None:
            record["risk_ratio"] = float(record["risk_ratio"])
        if "risk_return_ratio" in record and record["risk_return_ratio"] is not None:
            record["risk_return_ratio"] = float(record["risk_return_ratio"])
        if "z_score" in record and record["z_score"] is not None:
            record["z_score"] = float(record["z_score"])
        
        # 处理followup_data中的字段
        if "followup_data" in record and record["followup_data"]:
            for data_point in record["followup_data"]:
                if "ratio" in data_point:
                    data_point["ratio"] = float(data_point["ratio"])
                if "return" in data_point:
                    data_point["return"] = float(data_point["return"])
    
    return records 