"""
增强版LSTM时间序列预测模型
基于原LSTM预测器，添加了多项优化和改进的特性
"""

import os
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import timedelta
from datetime import datetime
import hashlib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import logging

# 抑制警告信息
warnings.filterwarnings("ignore")
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0:默认, 1:屏蔽info, 2:屏蔽info和warning, 3:屏蔽所有

try:
    import tensorflow as tf
    # 抑制TensorFlow特定的警告
    tf.get_logger().setLevel(logging.ERROR)
    
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional, LeakyReLU, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2, l1_l2
    from tensorflow.keras.initializers import GlorotNormal

    # 禁用eager执行可以减少一些警告
    # tf.compat.v1.disable_eager_execution()
    
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
    print("TensorFlow未安装，无法使用增强版LSTM模型")
    TF_AVAILABLE = False

from .prediction_utils import (
    save_cached_model,
    load_cached_model
)


class EnhancedLSTMPredictor:
    """增强版LSTM时间序列预测模型"""

    def __init__(self, cache_dir: str = 'model_cache'):
        """
        初始化增强版LSTM预测器
        
        参数:
            cache_dir: 模型缓存目录
        """
        self.cache_dir = cache_dir
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.model_cache_path = None
        self.preprocessed_data = {}
        self.model_version = "v5"  # 版本标识，用于缓存区分

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

    def _build_advanced_model(self, input_shape: Tuple[int, int]):
        """
        构建增强版LSTM模型架构
        
        参数:
            input_shape: 输入数据形状 (seq_length, features)
            
        返回:
            构建好的LSTM模型
        """
        initializer = GlorotNormal(seed=42)

        model = Sequential([
            # 第一层：双向GRU层
            Bidirectional(
                GRU(64, return_sequences=True,
                    kernel_initializer=initializer,
                    recurrent_initializer=initializer,
                    kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4),
                    recurrent_regularizer=l2(1e-4),
                    recurrent_dropout=0.0),  # 移除dropout，使用BatchNorm代替
                input_shape=input_shape
            ),
            BatchNormalization(),  # 添加批归一化，稳定训练

            # 第二层：LSTM层
            LSTM(48, return_sequences=True,
                 kernel_initializer=initializer,
                 recurrent_initializer=initializer,
                 kernel_regularizer=l2(1e-4),
                 recurrent_dropout=0.0),
            BatchNormalization(),
            Dropout(0.3),  # 在BatchNorm后应用dropout

            # 第三层：GRU层
            GRU(32, return_sequences=False,
                kernel_initializer=initializer,
                recurrent_initializer=initializer,
                kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            Dropout(0.2),

            # 全连接层网络
            Dense(24, kernel_initializer=initializer,
                  kernel_regularizer=l2(1e-4)),
            LeakyReLU(alpha=0.1),  # 使用LeakyReLU激活函数
            BatchNormalization(),
            Dropout(0.2),

            # 输出层
            Dense(1, activation='linear')
        ])

        # 使用Adam优化器，添加学习率衰减
        # 修复：使用legacy版本的优化器以支持decay参数
        try:
            # 尝试使用legacy版本的Adam优化器
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=1e-5)
        except:
            # 如果不支持legacy版本，使用标准优化器不带decay参数
            optimizer = Adam(learning_rate=0.001)
            print("注意：使用标准Adam优化器，不支持decay参数")

        # 使用Huber损失函数，对异常值更鲁棒
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae']
        )

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
        key_components = f"{code_a}_{code_b}_{params['seq_length']}_{params['model_type']}_{self.model_version}"
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
        return os.path.join(self.cache_dir, f"enhanced_lstm_{cache_key}")

    def _check_cache_valid(self, cache_info: Dict[str, Any], codes: Tuple[str, str],
                           params: Dict[str, Any], max_age_hours: int = 3) -> bool:
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
                cache_params.get("model_version") != self.model_version):
            return False

        # 检查缓存是否过期
        timestamp = cache_info["timestamp"]
        current_time = time.time()
        cache_age_hours = (current_time - timestamp) / 3600

        return cache_age_hours <= max_age_hours

    def _create_enhanced_features(self, data: np.ndarray) -> pd.DataFrame:
        """
        创建精细特征集，优化信号质量
        
        参数:
            data: 原始价格比值数据
            
        返回:
            特征DataFrame
        """
        # 转换为pandas DataFrame以便于计算技术指标
        df = pd.DataFrame(data, columns=['price'])

        # 1. 移动平均线
        for window in [5, 10, 20, 30]:
            df[f'ma{window}'] = df['price'].rolling(window=window).mean()

        # 2. 价格相对于均线的位置
        for window in [5, 10, 20]:
            df[f'price_ma{window}_ratio'] = df['price'] / df[f'ma{window}']

        # 3. 均线差异和交叉信号
        df['ma5_10_diff'] = df['ma5'] - df['ma10']
        df['ma10_20_diff'] = df['ma10'] - df['ma20']

        # 计算均线交叉情况
        df['ma5_10_cross'] = ((df['ma5'].shift(1) < df['ma10'].shift(1)) &
                              (df['ma5'] > df['ma10'])).astype(float) - \
                             ((df['ma5'].shift(1) > df['ma10'].shift(1)) &
                              (df['ma5'] < df['ma10'])).astype(float)

        # 4. 动量指标
        for period in [3, 5, 10]:
            # 变动量
            df[f'momentum_{period}'] = df['price'].diff(period)
            # 变动率
            df[f'roc_{period}'] = df['price'].pct_change(period) * 100

        # 5. 波动性指标
        for window in [10, 20]:
            df[f'volatility_{window}'] = df['price'].rolling(window=window).std()
            # 相对于移动平均的波动率
            df[f'rel_volatility_{window}'] = df[f'volatility_{window}'] / df[f'ma{window}']

        # 6. 趋势指标
        # 计算斜率 (价格变化的线性回归系数)
        for window in [5, 10, 15]:
            x = np.arange(window)
            for i in range(window, len(df) + 1):
                y = df['price'].iloc[i - window:i].values
                if len(y) == window:  # 确保数据长度满足要求
                    slope, _ = np.polyfit(x, y, 1)
                    df.loc[df.index[i - 1], f'slope_{window}'] = slope

        # 7. 周期性指标
        # 计算周期特征 - 5日、10日高低点
        for window in [5, 10]:
            df[f'high_{window}'] = df['price'].rolling(window=window).max()
            df[f'low_{window}'] = df['price'].rolling(window=window).min()
            df[f'high_low_range_{window}'] = df[f'high_{window}'] - df[f'low_{window}']
            # 当前价格在区间中的位置
            df[f'price_position_{window}'] = (df['price'] - df[f'low_{window}']) / df[f'high_low_range_{window}']

        # 填充NaN值 - 使用更合理的前向和后向填充组合
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 移除极端值，将其替换为边界值
        for col in df.columns:
            if col != 'price':  # 保留原始价格
                # 计算 99% 分位数作为上下界
                upper_bound = df[col].quantile(0.99)
                lower_bound = df[col].quantile(0.01)
                # 替换极端值
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        return df

    def _create_sequences(self, df: pd.DataFrame, target_col: str, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建监督学习序列，使用增强的标准化方法
        
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

        # 对每个特征单独标准化，使用RobustScaler处理异常值
        for i, col in enumerate(feature_columns):
            if col == target_col:
                # 价格使用MinMaxScaler保持一定的范围
                scaler = MinMaxScaler(feature_range=(-1, 1))
            else:
                # 其他特征使用RobustScaler减少异常值影响
                scaler = RobustScaler()

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

    def _apply_walkforward_validation(self, df: pd.DataFrame, target_col: str, seq_length: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        应用优化的时间序列交叉验证，确保测试集是最近的连续数据
        
        参数:
            df: 特征DataFrame
            target_col: 目标列名
            seq_length: 序列长度
            
        返回:
            训练集和测试集
        """
        # 创建所有序列
        X, y = self._create_sequences(df, target_col, seq_length)

        # 确定数据集大小
        total_samples = len(X)

        # 如果样本太少，使用简单的分割
        if total_samples < 100:
            split_idx = int(total_samples * 0.8)
            return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

        # 取最近的25%数据作为测试集
        test_size = int(total_samples * 0.25)
        train_size = total_samples - test_size

        X_train, y_train = X[:train_size], y[:train_size]
        X_test, y_test = X[train_size:], y[train_size:]

        return X_train, y_train, X_test, y_test

    def predict(self, ratio_data: List[float], dates: List[str], code_a: str, code_b: str,
                prediction_days: int = 30, confidence_level: float = 0.95,
                model_type: str = "enhanced_lstm") -> Dict[str, Any]:
        """
        预测未来价格比值，使用增强的LSTM模型
        
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
        min_data_points = 60  # 增加最小数据点要求
        if len(ratio_data) < min_data_points:
            return {"error": f"历史数据不足，至少需要{min_data_points}个数据点"}

        # 数据预处理
        data = np.array(ratio_data).reshape(-1, 1)

        # 价格整体标准化
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_prices = self.scaler.fit_transform(data)

        # 设置序列长度 - 自适应调整
        seq_length = min(int(len(ratio_data) * 0.2), 45)  # 最多使用20%的数据作为序列长度，上限45
        seq_length = max(seq_length, 15)  # 确保序列长度至少为15

        params = {
            "seq_length": seq_length,
            "prediction_days": prediction_days,
            "confidence_level": confidence_level,
            "model_type": model_type,
            "model_version": self.model_version
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
            max_age_hours=3  # 缩短缓存有效期，确保模型更新
        )

        # 创建增强特征
        feature_df = self._create_enhanced_features(data)

        # 使用优化的走进式验证分割数据
        X_train, y_train, X_test, y_test = self._apply_walkforward_validation(
            feature_df, 'price', seq_length
        )

        # 保存历史数据供后续使用
        self.preprocessed_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": list(feature_df.columns),  # 确保转换为列表
            "historical_dates": dates[-30:] if len(dates) > 30 else dates,
            "historical_values": ratio_data[-30:] if len(ratio_data) > 30 else ratio_data,
            "future_dates": [(datetime.strptime(dates[-1], "%Y-%m-%d") +
                              timedelta(days=i + 1)).strftime("%Y-%m-%d")
                             for i in range(prediction_days)]
        }

        # 如果缓存有效，直接加载模型
        if cache_valid:
            print("使用缓存模型预测")
            try:
                self.model = load_model(self.model_cache_path)
            except Exception as e:
                print(f"加载缓存模型失败：{e}，将训练新模型")
                cache_valid = False

        # 如果缓存无效，训练新模型
        if not cache_valid:
            print("训练新模型")
            # 构建模型
            n_features = X_train.shape[2]
            self.model = self._build_advanced_model((seq_length, n_features))

            # 设置回调函数
            callbacks = [
                # 早停
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=0
                ),
                # 学习率调整
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=0.0001,
                    verbose=0
                ),
                # 添加指数衰减学习率调度器作为替代decay参数的方案
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch, lr: lr * (1 / (1 + 1e-5 * epoch)),
                    verbose=0
                ),
                # 模型检查点
                ModelCheckpoint(
                    self.model_cache_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=0
                )
            ]

            # 训练模型 - 使用更大的批量和更多的轮次
            self.model.fit(
                X_train, y_train,
                epochs=200,  # 增加最大轮次
                batch_size=32,  # 增加批量大小提高稳定性
                validation_split=0.2,
                callbacks=callbacks,
                verbose=0
            )

            # 保存模型信息
            model_data = {
                "timestamp": time.time(),
                "codes": [code_a, code_b],
                "params": params
            }
            save_cached_model(self.model_cache_path, model_data)

        # 评估模型性能
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test, verbose=0)

            # 将标准化的预测反转回原始比例
            price_idx = feature_df.columns.tolist().index('price')

            # 转换回实际价格空间
            price_scaler = self.feature_scaler['price']
            y_test_rescaled = price_scaler.inverse_transform(y_test).flatten()
            y_pred_rescaled = price_scaler.inverse_transform(y_pred).flatten()

            # 计算性能指标
            mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

            # 计算R方值 - 使用sklearn的实现增加稳定性
            r2 = r2_score(y_test_rescaled, y_pred_rescaled)
            r2 = max(-1.0, min(1.0, r2))  # 限制范围

            performance = {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            }
        else:
            # 如果没有测试集，使用估计的性能指标
            performance = {
                "mse": 0.01,
                "rmse": 0.1,
                "mae": 0.08,
                "r2": 0.5,
                "estimated": True
            }

        # 预测未来价格 - 使用集成预测减少方差
        # 获取最新的特征序列
        last_sequence = X_test[-1:] if len(X_test) > 0 else X_train[-1:]

        # 获取当前特征集的最后一行，用于更新预测
        feature_values = np.zeros((1, len(feature_df.columns)))
        for i, col in enumerate(feature_df.columns):
            # 反标准化当前序列的最后一帧对应特征
            feature_values[0, i] = self.feature_scaler[col].inverse_transform(
                last_sequence[0, -1:, i].reshape(-1, 1)
            )[0, 0]

        # 计算历史波动性，用于调整预测区间和添加噪声
        price_history = np.array(ratio_data[-60:]) if len(ratio_data) > 60 else np.array(ratio_data)
        price_changes = np.diff(price_history)
        historical_volatility = np.std(price_changes)

        # 进行多次采样预测并取平均值 - 使用蒙特卡洛模拟
        n_samples = 15  # 增加样本数量提高稳定性
        all_predictions = np.zeros((n_samples, prediction_days))

        # 启用批量预测提高性能
        for sample in range(n_samples):
            # 复制初始序列，避免交叉影响
            curr_seq = last_sequence.copy()
            curr_features = feature_values.copy()

            # 在一个批次中进行所有预测
            for i in range(prediction_days):
                # 预测下一个值
                pred = self.model.predict(curr_seq, verbose=0)[0, 0]

                # 添加一些随机波动，波动程度随预测天数增加而增加
                noise_scale = historical_volatility * 0.25 * (1 + i / prediction_days)
                noise = np.random.normal(0, noise_scale)

                # 将预测值反标准化
                price_pred = self.feature_scaler['price'].inverse_transform([[pred]])[0, 0]

                # 添加噪声到实际价格
                price_pred += noise

                # 保存预测
                all_predictions[sample, i] = price_pred

                if i < prediction_days - 1:  # 如果需要继续预测
                    # 更新特征和序列
                    curr_features = self._update_features(curr_features, price_pred, feature_df.columns)

                    # 标准化新特征
                    new_sequence = np.zeros((1, len(feature_df.columns)))
                    for j, col in enumerate(feature_df.columns):
                        new_sequence[0, j] = self.feature_scaler[col].transform([[curr_features[0, j]]])[0, 0]

                    # 更新序列 - 移除最旧的，添加新预测的
                    curr_seq = np.roll(curr_seq, -1, axis=1)
                    curr_seq[0, -1, :] = new_sequence

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

        # 计算预测置信度
        # 相对标准差越低，R方值越高，置信度越高
        base_confidence = 0.6  # 基础置信度
        std_factor = max(0, 0.2 * (1 - min(1, relative_std * 5)))  # 标准差对置信度的贡献
        r2_factor = 0.2 * performance.get("r2", 0)  # R方值对置信度的贡献

        # 确保置信度在合理范围内
        confidence = min(0.95, max(0.5, base_confidence + std_factor + r2_factor))

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
            "confidence": confidence
        }

        return results

    def _update_features(self, current_features: np.ndarray, new_price: float,
                         feature_columns: List[str]) -> np.ndarray:
        """
        根据新预测的价格更新特征，同时保持特征间的关系一致性
        
        参数:
            current_features: 当前特征值
            new_price: 新预测的价格
            feature_columns: 特征列名列表
            
        返回:
            更新后的特征值
        """
        # 复制当前特征
        new_features = current_features.copy()
        
        # 更新价格 - 修复可能的索引错误
        try:
            # 尝试直接使用列表的index方法
            price_idx = feature_columns.index('price')
        except:
            # 如果feature_columns是pandas的Index对象，使用不同的方法
            if hasattr(feature_columns, 'get_loc'):
                price_idx = feature_columns.get_loc('price')
            else:
                # 最后的备选方案，手动搜索
                price_idx = -1
                for i, col in enumerate(feature_columns):
                    if col == 'price':
                        price_idx = i
                        break
                if price_idx == -1:  # 如果仍然找不到
                    print("警告: 找不到'price'列，使用第一列作为价格列")
                    price_idx = 0
        
        new_features[0, price_idx] = new_price
        
        # 定义特征更新逻辑
        
        # 1. 更新移动平均线 - 对所有ma特征进行加权更新
        for i, col in enumerate(feature_columns):
            if col.startswith('ma'):
                # 判断是否是纯粹的移动平均线特征 (如 ma5, ma10)，而不是其衍生特征 (如 ma5_10_diff)
                if col[2:].isdigit():  # 确保后面是纯数字
                    window = int(col[2:])  # 提取窗口大小
                    weight = 1.0 / window
                    new_features[0, i] = (1 - weight) * new_features[0, i] + weight * new_price
        
        # 2. 更新价格相对于均线的位置
        for i, col in enumerate(feature_columns):
            if col.startswith('price_ma') and col.endswith('_ratio'):
                ma_col = col.replace('price_', '').replace('_ratio', '')
                # 安全地获取索引
                try:
                    ma_idx = feature_columns.index(ma_col) if isinstance(feature_columns, list) else feature_columns.get_loc(ma_col)
                    if new_features[0, ma_idx] != 0:
                        new_features[0, i] = new_price / new_features[0, ma_idx]
                except (ValueError, KeyError):
                    # 如果找不到对应的MA列，跳过
                    continue
        
        # 3. 更新均线差异 - 安全地获取所有索引
        def safe_get_index(col_name):
            """安全地获取列索引，即使列不存在也不会报错"""
            try:
                if isinstance(feature_columns, list):
                    return feature_columns.index(col_name)
                else:
                    return feature_columns.get_loc(col_name)
            except (ValueError, KeyError):
                return -1  # 返回-1表示未找到
        
        ma5_idx = safe_get_index('ma5')
        ma10_idx = safe_get_index('ma10')
        ma20_idx = safe_get_index('ma20')
        ma5_10_diff_idx = safe_get_index('ma5_10_diff')
        ma10_20_diff_idx = safe_get_index('ma10_20_diff')
        
        if ma5_idx >= 0 and ma10_idx >= 0 and ma5_10_diff_idx >= 0:
            new_features[0, ma5_10_diff_idx] = new_features[0, ma5_idx] - new_features[0, ma10_idx]
            
        if ma10_idx >= 0 and ma20_idx >= 0 and ma10_20_diff_idx >= 0:
            new_features[0, ma10_20_diff_idx] = new_features[0, ma10_idx] - new_features[0, ma20_idx]
        
        # 4. 更新动量指标
        for i, col in enumerate(feature_columns):
            if col.startswith('momentum_'):
                new_features[0, i] = new_price - current_features[0, price_idx]
            elif col.startswith('roc_'):
                if current_features[0, price_idx] != 0:
                    new_features[0, i] = (new_price / current_features[0, price_idx] - 1) * 100
        
        # 5. 更新高低点范围信息 - 使用安全索引获取方法
        for window in [5, 10]:
            high_col = f'high_{window}'
            low_col = f'low_{window}'
            range_col = f'high_low_range_{window}'
            pos_col = f'price_position_{window}'
            
            high_idx = safe_get_index(high_col)
            low_idx = safe_get_index(low_col)
            range_idx = safe_get_index(range_col)
            pos_idx = safe_get_index(pos_col)
            
            if high_idx >= 0:
                new_features[0, high_idx] = max(new_features[0, high_idx], new_price)
            
            if low_idx >= 0:
                new_features[0, low_idx] = min(new_features[0, low_idx], new_price)
            
            if high_idx >= 0 and low_idx >= 0 and range_idx >= 0:
                new_features[0, range_idx] = new_features[0, high_idx] - new_features[0, low_idx]
            
            if low_idx >= 0 and range_idx >= 0 and pos_idx >= 0 and new_features[0, range_idx] != 0:
                new_features[0, pos_idx] = (new_price - new_features[0, low_idx]) / new_features[0, range_idx]
        
        return new_features
