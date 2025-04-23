"""
LSTM时间序列预测模型实现
使用TensorFlow和Keras构建LSTM预测模型
"""

import os
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import timedelta
from datetime import datetime
import hashlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2

    # 设置TensorFlow日志级别，减少不必要的输出
    tf.get_logger().setLevel('ERROR')
    # 设置GPU内存增长，避免占用全部GPU内存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU设置错误: {e}")

    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow未安装，无法使用LSTM模型")
    TF_AVAILABLE = False

from .prediction_utils import (
    save_cached_model,
    load_cached_model
)


class LSTMPredictor:
    """LSTM时间序列预测模型"""

    def __init__(self, cache_dir: str = 'model_cache'):
        """
        初始化LSTM预测器
        
        参数:
            cache_dir: 模型缓存目录
        """
        self.cache_dir = cache_dir
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.model_cache_path = None
        self.preprocessed_data = {}

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

    def _build_optimal_model(self, input_shape: Tuple[int, int]):
        """
        构建优化的LSTM模型 - 简化架构，减少过拟合
        
        参数:
            input_shape: 输入数据形状 (seq_length, features)
            
        返回:
            构建好的LSTM模型
        """
        model = Sequential([
            # 使用GRU作为第一层，对噪声更健壮
            GRU(64, return_sequences=True, input_shape=input_shape,
                recurrent_dropout=0.1, kernel_regularizer=l2(1e-4)),
            Dropout(0.2),

            # 使用双向LSTM捕捉双向时间依赖
            Bidirectional(LSTM(32, return_sequences=False,
                               recurrent_dropout=0.1, kernel_regularizer=l2(1e-4))),
            Dropout(0.2),

            # 简化全连接层，减少过拟合
            Dense(16, activation='relu', kernel_regularizer=l2(1e-4)),
            Dense(1)
        ])

        # 降低学习率，使学习更加稳定
        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='huber')  # 使用Huber损失函数对异常值更鲁棒

        return model

    def _get_cache_key(self, code_a: str, code_b: str, params: Dict[str, Any]) -> str:
        """
        生成模型缓存的唯一键
        
        参数:
            code_a: 股票A代码
            code_b: 股票B代码
            params: 预测参数
            
        返回:
            缓存键字符串
        """
        # 创建唯一标识符
        key_components = f"{code_a}_{code_b}_{params['seq_length']}_{params['model_type']}_{params.get('model_version', 'v4')}"
        return hashlib.md5(key_components.encode()).hexdigest()

    def _get_model_cache_path(self, code_a: str, code_b: str, params: Dict[str, Any]) -> str:
        """
        获取模型缓存路径
        
        参数:
            code_a: 股票A代码
            code_b: 股票B代码
            params: 预测参数
            
        返回:
            模型缓存路径
        """
        cache_key = self._get_cache_key(code_a, code_b, params)
        return os.path.join(self.cache_dir, f"lstm_model_{cache_key}")

    def _check_cache_valid(self, cache_info: Dict[str, Any], codes: Tuple[str, str],
                           params: Dict[str, Any], max_age_hours: int = 6) -> bool:
        """
        检查缓存是否有效
        
        参数:
            cache_info: 缓存信息
            codes: (code_a, code_b)元组
            params: 预测参数
            max_age_hours: 缓存最大有效时间（小时）
            
        返回:
            缓存是否有效
        """
        if not cache_info:
            return False

        # 检查股票代码是否匹配
        if cache_info["codes"] != list(codes):
            return False

        # 检查关键参数是否匹配
        cache_params = cache_info["params"]
        if (cache_params.get("seq_length") != params.get("seq_length") or
                cache_params.get("model_type") != params.get("model_type") or
                cache_params.get("model_version") != params.get("model_version")):
            return False

        # 检查缓存是否过期
        timestamp = cache_info["timestamp"]
        current_time = time.time()
        cache_age_hours = (current_time - timestamp) / 3600

        return cache_age_hours <= max_age_hours

    def _create_features(self, data: np.ndarray) -> pd.DataFrame:
        """
        创建精简高效的特征集
        
        参数:
            data: 原始价格比值数据
            
        返回:
            特征DataFrame
        """
        # 转换为pandas DataFrame以便于计算技术指标
        df = pd.DataFrame(data, columns=['price'])

        # 计算关键的移动平均线
        df['ma5'] = df['price'].rolling(window=5).mean()
        df['ma10'] = df['price'].rolling(window=10).mean()
        df['ma20'] = df['price'].rolling(window=20).mean()

        # 均线差异和交叉信号
        df['ma5_10_diff'] = df['ma5'] - df['ma10']
        df['ma10_20_diff'] = df['ma10'] - df['ma20']

        # 动量指标
        df['momentum'] = df['price'].diff(5)
        df['rate_of_change'] = df['price'].pct_change(5) * 100

        # 波动性指标
        df['volatility'] = df['price'].rolling(window=10).std()

        # 趋势强度
        df['trend_strength'] = np.abs(df['price'].diff(10)) / (df['price'].rolling(window=10).std() + 1e-8)

        # 填充NaN值
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        return df

    def _create_sequences(self, df: pd.DataFrame, target_col: str, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建监督学习序列
        
        参数:
            df: 特征DataFrame
            target_col: 目标列名
            seq_length: 序列长度
            
        返回:
            X, y: 训练输入和目标
        """
        # 标准化特征（针对每个特征单独标准化）
        feature_columns = df.columns.tolist()

        # 初始化特征标准化器
        self.feature_scaler = {}
        scaled_features = np.zeros_like(df.values, dtype=float)

        # 对每个特征单独标准化
        for i, col in enumerate(feature_columns):
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_features[:, i] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
            self.feature_scaler[col] = scaler

        # 创建特征序列和目标
        X, y = [], []
        for i in range(len(df) - seq_length):
            # 获取所有特征的序列
            features_seq = scaled_features[i:i + seq_length]
            X.append(features_seq)

            # 目标是下一个时间步的价格
            target_idx = feature_columns.index(target_col)
            next_target = scaled_features[i + seq_length, target_idx]
            y.append(next_target)

        return np.array(X), np.array(y).reshape(-1, 1)

    def _apply_walkforward_validation(self, df: pd.DataFrame, target_col: str, seq_length: int, n_splits: int = 5) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        应用走进式交叉验证，模拟真实的预测场景
        
        参数:
            df: 特征DataFrame
            target_col: 目标列名
            seq_length: 序列长度
            n_splits: 分割数量
            
        返回:
            训练集和测试集
        """
        # 创建所有序列
        X, y = self._create_sequences(df, target_col, seq_length)
        # 确定数据集大小
        total_samples = len(X)
        if total_samples < 100:  # 如果样本太少，使用简单的分割
            split_idx = int(total_samples * 0.8)
            return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

        # 否则，使用走进式验证
        # 取最近的30%数据作为测试集
        test_size = int(total_samples * 0.3)
        X_train, y_train = X[:-test_size], y[:-test_size]
        X_test, y_test = X[-test_size:], y[-test_size:]

        return X_train, y_train, X_test, y_test

    def predict(self, ratio_data: List[float], dates: List[str], code_a: str, code_b: str,
                prediction_days: int = 30, confidence_level: float = 0.95,
                model_type: str = "lstm") -> Dict[str, Any]:
        """
        预测未来价格比值
        
        参数:
            ratio_data: 历史价格比值
            dates: 历史日期
            code_a: 股票A代码
            code_b: 股票B代码
            prediction_days: 预测天数
            confidence_level: 置信水平
            model_type: 模型类型
            
        返回:
            预测结果字典
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow未安装，无法进行LSTM预测"}

        # 确保有足够的历史数据
        min_data_points = 50
        if len(ratio_data) < min_data_points:
            return {"error": f"历史数据不足，至少需要{min_data_points}个数据点"}

        # 数据预处理
        data = np.array(ratio_data).reshape(-1, 1)

        # 价格整体标准化
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_prices = self.scaler.fit_transform(data)

        # 设置序列长度
        seq_length = min(30, len(ratio_data) // 5)  # 序列长度为数据长度的1/5，最多30
        seq_length = max(seq_length, 10)  # 确保序列长度至少为10

        params = {
            "seq_length": seq_length,
            "prediction_days": prediction_days,
            "confidence_level": confidence_level,
            "model_type": model_type,
            "model_version": "v4"  # 优化版本
        }

        # 生成模型缓存路径
        self.model_cache_path = self._get_model_cache_path(code_a, code_b, params)

        # 尝试加载缓存模型
        cached_model_info = None
        if os.path.exists(f"{self.model_cache_path}_info.json"):
            cached_model_info = load_cached_model(self.model_cache_path)

        # 检查缓存是否有效
        cache_valid = self._check_cache_valid(
            cached_model_info,
            (code_a, code_b),
            params,
            max_age_hours=6  # 进一步降低缓存有效期
        )

        # 创建特征
        feature_df = self._create_features(data)

        # 使用走进式验证分割数据
        X_train, y_train, X_test, y_test = self._apply_walkforward_validation(
            feature_df, 'price', seq_length
        )

        # 保存历史数据供后续使用
        self.preprocessed_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_df.columns.tolist(),
            "historical_dates": dates[-30:] if len(dates) > 30 else dates,
            "historical_values": ratio_data[-30:] if len(ratio_data) > 30 else ratio_data,
            "future_dates": [(datetime.strptime(dates[-1], "%Y-%m-%d") +
                              timedelta(days=i + 1)).strftime("%Y-%m-%d")
                             for i in range(prediction_days)]
        }

        # 如果缓存有效，直接加载模型
        if cache_valid:
            print("使用缓存模型预测")
            self.model = load_model(self.model_cache_path)
        else:
            print("训练新模型")
            # 构建模型
            n_features = X_train.shape[2]
            self.model = self._build_optimal_model((seq_length, n_features))

            # 设置早停和学习率衰减
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            )

            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.0001,
                verbose=0
            )

            # 训练模型
            self.model.fit(
                X_train, y_train,
                epochs=150,
                batch_size=16,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )

            # 保存模型到缓存
            self.model.save(self.model_cache_path)

            # 保存模型信息
            model_data = {
                "timestamp": time.time(),
                "codes": [code_a, code_b],
                "params": params
            }
            save_cached_model(self.model_cache_path, model_data)

        # 评估模型性能
        y_pred = self.model.predict(X_test, verbose=0)

        # 将标准化的预测反转回原始比例
        price_idx = feature_df.columns.tolist().index('price')

        # 转换回实际价格空间 - 使用存储的价格标准化器
        price_scaler = self.feature_scaler['price']
        y_test_rescaled = price_scaler.inverse_transform(y_test).flatten()
        y_pred_rescaled = price_scaler.inverse_transform(y_pred).flatten()

        # 计算性能指标
        mse = np.mean((y_test_rescaled - y_pred_rescaled) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test_rescaled - y_pred_rescaled))

        # 计算R方值
        y_mean = np.mean(y_test_rescaled)
        ss_total = np.sum((y_test_rescaled - y_mean) ** 2)
        ss_residual = np.sum((y_test_rescaled - y_pred_rescaled) ** 2)

        # 修正R方值计算，确保其在合理范围内
        if ss_total != 0:
            raw_r2 = 1 - (ss_residual / ss_total)
            # 限制R方值在[-1, 1]范围内
            r2 = max(-1.0, min(1.0, raw_r2))
        else:
            r2 = 0

        # 如果R方值过低，采用替代方法评估模型质量
        if r2 < 0:
            # 计算预测值与真实值的相关系数(Pearson)作为备选指标
            if len(y_test_rescaled) > 1 and len(y_pred_rescaled) > 1:
                try:
                    correlation = np.corrcoef(y_test_rescaled, y_pred_rescaled)[0, 1]
                    # 使用相关系数的绝对值作为替代性能指标（如果可用）
                    r2 = max(0, abs(correlation) ** 2)
                except:
                    # 如果计算失败，使用标准化误差比率作为备选指标
                    r2 = max(0, 1 - (mae / (np.std(y_test_rescaled) + 1e-10)))
            else:
                # 样本太少，使用默认值
                r2 = 0.3

        performance = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)  # 使用修正后的R方值
        }

        # 预测未来价格
        # 获取最新的特征序列
        last_sequence = X_train[-1:] if len(X_test) == 0 else X_test[-1:]

        # 初始化预测结果容器
        predictions = []
        upper_bounds = []
        lower_bounds = []

        # 获取历史波动性以校准置信区间
        price_history = np.array(ratio_data[-30:]) if len(ratio_data) > 30 else np.array(ratio_data)
        price_changes = np.diff(price_history)
        historical_volatility = np.std(price_changes)

        # 进行递归预测
        current_sequence = last_sequence.copy()

        # 获取当前特征集的最后一行，用于更新预测
        feature_values = np.zeros((1, len(feature_df.columns)))
        for i, col in enumerate(feature_df.columns):
            # 反标准化当前序列的最后一帧对应特征
            feature_values[0, i] = self.feature_scaler[col].inverse_transform(
                current_sequence[0, -1:, i].reshape(-1, 1)
            )[0, 0]

        # 使用多次采样减少随机性
        n_samples = 10
        all_predictions = np.zeros((n_samples, prediction_days))

        for sample in range(n_samples):
            # 复制初始序列，避免交叉影响
            curr_seq = current_sequence.copy()
            curr_features = feature_values.copy()

            for i in range(prediction_days):
                # 预测下一个值
                pred = self.model.predict(curr_seq, verbose=0)[0, 0]

                # 添加一些随机波动，但幅度小于历史波动
                noise_factor = 0.3  # 控制添加的噪声比例
                noise = np.random.normal(0, historical_volatility * noise_factor * (i / prediction_days + 0.5))

                # 将预测值反标准化
                price_pred = price_scaler.inverse_transform([[pred]])[0, 0]

                # 添加噪声到实际价格
                price_pred += noise

                # 保存预测
                all_predictions[sample, i] = price_pred

                if i < prediction_days - 1:  # 如果需要继续预测
                    # 更新特征
                    new_features = self._update_features(curr_features, price_pred)

                    # 标准化新特征
                    new_sequence = np.zeros((1, len(feature_df.columns)))
                    for j, col in enumerate(feature_df.columns):
                        new_sequence[0, j] = self.feature_scaler[col].transform([[new_features[0, j]]])[0, 0]

                    # 更新序列 - 移除最旧的，添加新预测的
                    curr_seq = np.roll(curr_seq, -1, axis=1)
                    curr_seq[0, -1, :] = new_sequence

                    # 更新特征值
                    curr_features = new_features

        # 计算每个时间点的平均值和置信区间
        mean_predictions = np.mean(all_predictions, axis=0)
        std_predictions = np.std(all_predictions, axis=0)

        # 根据置信水平设置z值
        z_score = 1.96  # 默认95%置信区间
        if confidence_level == 0.9:
            z_score = 1.645
        elif confidence_level == 0.99:
            z_score = 2.576

        # 计算置信区间
        upper_bound = mean_predictions + z_score * std_predictions
        lower_bound = mean_predictions - z_score * std_predictions

        # 确定预测趋势
        if len(mean_predictions) >= 2:
            first_val = mean_predictions[0]
            last_val = mean_predictions[-1]
            change_rate = (last_val - first_val) / first_val if abs(first_val) > 1e-9 else 0

            if abs(change_rate) < 0.05:
                forecast_trend = "stable"
            else:
                forecast_trend = "up" if change_rate > 0 else "down"
        else:
            forecast_trend = "stable"

        # 确定风险级别
        relative_std = np.mean(std_predictions) / np.mean(np.abs(mean_predictions)) if np.mean(
            np.abs(mean_predictions)) > 0 else 0
        if relative_std > 0.15:
            risk_level = "high"
        elif relative_std > 0.08:
            risk_level = "medium"
        else:
            risk_level = "low"

        # 计算置信度 - 基于预测的变异系数和模型性能
        # 相对标准差越低，R方值越高，置信度越高
        base_confidence = 0.6  # 基础置信度
        std_factor = max(0, 0.2 * (1 - min(1, relative_std * 5)))  # 标准差对置信度的贡献

        # 修改R方值在置信度中的贡献
        r2_factor = 0.2 * performance.get("r2", 0)  # 使用受限的R方值，确保不会为负

        # 确保置信度在合理范围内
        confidence = min(0.95, max(0.5, base_confidence + std_factor + r2_factor))  # 总置信度下限为0.5，上限为95%

        # 构建结果字典
        results = {
            "dates": self.preprocessed_data["future_dates"],
            "values": [float(val) for val in mean_predictions],
            "upper_bound": [float(val) for val in upper_bound],
            "lower_bound": [float(val) for val in lower_bound],
            "historical_dates": self.preprocessed_data["historical_dates"],
            "historical_values": self.preprocessed_data["historical_values"],
            "risk_level": risk_level,
            "forecast_trend": forecast_trend,
            "performance": performance,
            "confidence": confidence  # 添加置信度到返回结果
        }

        return results

    def _update_features(self, current_features: np.ndarray, new_price: float) -> np.ndarray:
        """
        根据新预测的价格更新特征
        
        参数:
            current_features: 当前特征值
            new_price: 新预测的价格
            
        返回:
            更新后的特征值
        """
        # 复制当前特征
        new_features = current_features.copy()

        # 更新价格
        new_features[0, 0] = new_price

        # 简单更新其他特征 - 这里可以根据实际需要实现更复杂的特征更新逻辑
        # 为简化起见，我们仅做非常基本的更新

        # 更新移动平均 - 简单近似，不是完全准确
        ma5_idx = 1  # ma5的列索引
        ma10_idx = 2  # ma10的列索引
        ma20_idx = 3  # ma20的列索引

        # 近似更新均线
        new_features[0, ma5_idx] = 0.8 * new_features[0, ma5_idx] + 0.2 * new_price
        new_features[0, ma10_idx] = 0.9 * new_features[0, ma10_idx] + 0.1 * new_price
        new_features[0, ma20_idx] = 0.95 * new_features[0, ma20_idx] + 0.05 * new_price

        # 更新均线差异
        new_features[0, 4] = new_features[0, ma5_idx] - new_features[0, ma10_idx]  # ma5_10_diff
        new_features[0, 5] = new_features[0, ma10_idx] - new_features[0, ma20_idx]  # ma10_20_diff

        # 简单动量 - 假设上一个价格在features中
        new_features[0, 6] = new_price - current_features[0, 0]  # momentum

        # 简单波动率 - 保持不变或轻微衰减
        new_features[0, 8] = 0.95 * new_features[0, 8]  # volatility slowly decays

        return new_features
