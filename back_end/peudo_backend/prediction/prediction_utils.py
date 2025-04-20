"""
预测模型辅助工具函数
包含数据预处理、后处理和模型评估功能
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import os


def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    将时间序列数据转换为监督学习格式的序列
    
    参数:
        data: 输入的时间序列数据
        seq_length: 序列长度（lookback窗口）
        
    返回:
        (X, y): X为输入序列，y为目标值
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def preprocess_data(ratio_data: List[float], dates: List[str], seq_length: int = 10, 
                    prediction_days: int = 30, test_size: float = 0.2) -> Dict[str, Any]:
    """
    预处理时间序列数据用于LSTM模型训练
    
    参数:
        ratio_data: 价格比值数据
        dates: 对应的日期
        seq_length: 序列长度（lookback窗口）
        prediction_days: 预测天数
        test_size: 测试集比例
        
    返回:
        预处理后的数据字典，包含训练集、测试集和标准化参数
    """
    # 转换为NumPy数组
    data = np.array(ratio_data).reshape(-1, 1)
    
    # 数据标准化
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:  # 避免除以零
        std = 1e-8
    scaled_data = (data - mean) / std
    
    # 创建序列
    X, y = create_sequences(scaled_data, seq_length)
    
    # 分割训练集和测试集
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # 生成预测所需的最后一个序列
    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    
    # 转换日期
    last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
    future_dates = []
    for i in range(1, prediction_days + 1):
        future_dates.append((last_date + timedelta(days=i)).strftime("%Y-%m-%d"))
    
    # 仅保留最近30天历史数据用于图表展示
    recent_dates = dates[-30:] if len(dates) > 30 else dates
    recent_values = ratio_data[-30:] if len(ratio_data) > 30 else ratio_data
    
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "last_sequence": last_sequence,
        "mean": mean,
        "std": std,
        "future_dates": future_dates,
        "historical_dates": recent_dates,
        "historical_values": recent_values,
        "full_data": ratio_data,
        "original_dates": dates
    }


def postprocess_results(predictions: np.ndarray, preprocessed_data: Dict[str, Any], 
                        confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    对预测结果进行后处理，包括反标准化、计算置信区间等
    
    参数:
        predictions: 模型预测值
        preprocessed_data: 预处理后的数据字典
        confidence_level: 置信水平
        
    返回:
        后处理后的预测结果字典
    """
    # 获取预处理数据中的统计值
    mean = preprocessed_data["mean"]
    std = preprocessed_data["std"]
    
    # 反标准化预测值
    predictions_rescaled = predictions * std + mean
    predictions_list = predictions_rescaled.flatten().tolist()
    
    # 计算置信区间
    z_score = 1.96  # 默认使用95%置信区间的Z分数
    if confidence_level == 0.9:
        z_score = 1.645
    elif confidence_level == 0.99:
        z_score = 2.576
    
    # 使用历史数据标准差来估计预测误差
    prediction_std = std * 0.8  # 假设预测误差略小于历史标准差
    
    margin = z_score * prediction_std
    upper_bound = [float(val + margin) for val in predictions_list]
    lower_bound = [float(val - margin) for val in predictions_list]
    
    # 确定预测趋势方向
    if len(predictions_list) >= 2:
        first_val = predictions_list[0]
        last_val = predictions_list[-1]
        change_rate = (last_val - first_val) / first_val if first_val != 0 else 0
        
        if abs(change_rate) < 0.01:
            forecast_trend = "stable"
        else:
            forecast_trend = "up" if change_rate > 0 else "down"
    else:
        forecast_trend = "stable"
    
    # 确定风险级别
    avg_value = sum(predictions_list) / len(predictions_list) if predictions_list else 0
    avg_interval_width = sum(u - l for u, l in zip(upper_bound, lower_bound)) / len(predictions_list) if predictions_list else 0
    relative_width = avg_interval_width / avg_value if avg_value != 0 else 0
    
    if relative_width > 0.2:
        risk_level = "high"
    elif relative_width > 0.1:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # 返回后处理结果
    return {
        "dates": preprocessed_data["future_dates"],
        "values": [float(val) for val in predictions_list],
        "upper_bound": upper_bound,
        "lower_bound": lower_bound,
        "historical_dates": preprocessed_data["historical_dates"],
        "historical_values": preprocessed_data["historical_values"],
        "risk_level": risk_level,
        "forecast_trend": forecast_trend
    }


def evaluate_model_performance(model, X_test: np.ndarray, y_test: np.ndarray, 
                               preprocessed_data: Dict[str, Any]) -> Dict[str, float]:
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        X_test: 测试集输入
        y_test: 测试集目标值
        preprocessed_data: 预处理数据字典
        
    返回:
        包含性能指标的字典
    """
    # 获取预处理数据中的统计值
    mean = preprocessed_data["mean"]
    std = preprocessed_data["std"]
    
    # 模型预测
    y_pred = model.predict(X_test)
    
    # 反标准化
    y_test_rescaled = y_test * std + mean
    y_pred_rescaled = y_pred * std + mean
    
    # 计算性能指标
    mse = np.mean((y_test_rescaled - y_pred_rescaled) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))
    
    # 计算R方值
    y_mean = np.mean(y_test_rescaled)
    ss_total = np.sum((y_test_rescaled - y_mean) ** 2)
    ss_residual = np.sum((y_test_rescaled - y_pred_rescaled) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }


def save_cached_model(model_path: str, model_data: Dict[str, Any]):
    """
    保存模型缓存数据
    
    参数:
        model_path: 模型保存路径
        model_data: 模型数据字典
    """
    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型信息
    with open(f"{model_path}_info.json", "w") as f:
        # 将不可JSON序列化的数据转换为字符串
        info = {
            "timestamp": model_data["timestamp"],
            "codes": model_data["codes"],
            "params": model_data["params"]
        }
        json.dump(info, f)
    
    # 模型对象由TensorFlow或PyTorch的save函数处理
    # model_data["model"].save(model_path)


def load_cached_model(model_path: str) -> Dict[str, Any]:
    """
    加载缓存的模型数据
    
    参数:
        model_path: 模型路径
        
    返回:
        加载的模型数据字典，如果无法加载则返回None
    """
    try:
        # 加载模型信息
        with open(f"{model_path}_info.json", "r") as f:
            info = json.load(f)
        
        # 加载模型对象
        # from tensorflow.keras.models import load_model
        # model = load_model(model_path)
        
        return {
            "timestamp": info["timestamp"],
            "codes": info["codes"],
            "params": info["params"],
            # "model": model
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return None 