"""
股票价格比值预测模型预测器
用于加载预训练模型并提供快速预测功能
"""

import os
import numpy as np
import pandas as pd
import json
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
import traceback

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入数据库类
from back_end.peudo_backend.get_stock_data.stock_data_base import StockKlineDatabase

# 尝试导入TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.error(f"GPU设置错误: {e}")

    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow未安装，将使用替代模型")
    TF_AVAILABLE = False


class StockRatioPredictor:
    """股票价格比值预测模型预测器"""

    def __init__(self,
                 model_dir: str = 'trained_models',
                 default_model: str = None):
        """
        初始化预测器
        
        参数:
            model_dir: 模型目录
            default_model: 默认模型名称，如果为None则自动加载最新模型
        """
        self.model_dir = Path(model_dir)
        self.default_model = default_model

        # 初始化变量
        self.model = None
        self.scalers = None
        self.metadata = None
        self.feature_cols = None
        self.seq_length = 0
        self.forecast_horizon = 0
        self.model_loaded = False
        self.last_pair = (None, None)

        # 初始化数据库连接
        self.db = StockKlineDatabase()

        # 加载默认模型
        if default_model:
            self.load_model(default_model)
        else:
            self._load_latest_model()

    def _load_latest_model(self) -> bool:
        """
        加载最新模型
        
        返回:
            是否成功加载
        """
        if not self.model_dir.exists():
            logger.error(f"模型目录不存在: {self.model_dir}")
            return False

        # 查找所有模型目录
        model_dirs = [d for d in self.model_dir.glob("*") if d.is_dir()]

        if not model_dirs:
            logger.error(f"没有找到模型: {self.model_dir}")
            return False

        # 按修改时间排序
        model_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # 加载最新模型
        latest_model = model_dirs[0].name
        return self.load_model(latest_model)

    def load_model(self, model_name: str) -> bool:
        """
        加载指定模型
        
        参数:
            model_name: 模型名称
            
        返回:
            是否成功加载
        """
        model_path = self.model_dir / model_name

        if not model_path.exists():
            logger.error(f"模型路径不存在: {model_path}")
            return False

        try:
            # 加载TensorFlow模型
            model_file = model_path / "model.h5"
            if model_file.exists():
                self.model = load_model(str(model_file))
                logger.info(f"模型已加载: {model_file}")
            else:
                logger.error(f"模型文件不存在: {model_file}")
                return False

            # 加载缩放器
            scaler_file = model_path / "scalers.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"缩放器已加载: {scaler_file}")
            else:
                logger.warning(f"缩放器文件不存在: {scaler_file}")

            # 加载元数据
            metadata_file = model_path / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"元数据已加载: {metadata_file}")

                # 从元数据中提取序列长度和预测时间跨度
                # 这里需要根据模型配置提取，不同模型配置方式可能不同
                # 简化处理：从模型输入形状推断序列长度
                try:
                    input_shape = self.model.input_shape
                    if input_shape is not None and len(input_shape) == 3:
                        self.seq_length = input_shape[1]

                    output_shape = self.model.output_shape
                    if output_shape is not None and len(output_shape) == 2:
                        self.forecast_horizon = output_shape[1]

                    logger.info(f"序列长度: {self.seq_length}, 预测时间跨度: {self.forecast_horizon}")
                except Exception as e:
                    logger.warning(f"从模型中提取参数失败: {e}")
                    self.seq_length = 30  # 默认值
                    self.forecast_horizon = 5  # 默认值

            # 标记模型已加载
            self.model_loaded = True
            logger.info(f"模型 {model_name} 加载完成")
            return True

        except Exception as e:
            logger.error(f"加载模型出错: {e}")
            logger.error(traceback.format_exc())
            return False

    def fetch_stock_data(self,
                         stock_code: str,
                         days: int = 60,
                         end_date: str = None) -> pd.DataFrame:
        """
        获取最近N天的股票数据
        
        参数:
            stock_code: 股票代码
            days: 获取天数
            end_date: 结束日期，默认为当前日期
            
        返回:
            股票数据DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        # 计算开始日期，多获取一些数据以确保有足够的有效记录
        start_date_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
        start_date = start_date_dt.strftime('%Y-%m-%d')

        try:
            # 从数据库获取数据
            data = self.db.query_kline(stock_code, start_date, end_date)

            # 转换为DataFrame
            columns = ['stock_code', 'date', 'open', 'close', 'high', 'low',
                       'volume', 'amount', 'amplitude', 'change_pct', 'change_amt', 'turnover']
            df = pd.DataFrame(data, columns=columns)

            # 转换日期列为日期类型
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)

            # 确保数值类型正确
            for col in df.columns:
                if col != 'stock_code':
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # 取最近N天的数据
            if len(df) > days:
                df = df.iloc[-days:]

            logger.info(f"获取股票 {stock_code} 数据: {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            raise

    def prepare_prediction_data(self,
                                stock_code_a: str,
                                stock_code_b: str,
                                days: int = None,
                                end_date: str = None) -> pd.DataFrame:
        """
        准备预测所需的数据
        
        参数:
            stock_code_a: 股票A代码
            stock_code_b: 股票B代码
            days: 获取天数，默认使用序列长度的2倍
            end_date: 结束日期
            
        返回:
            处理后的特征DataFrame
        """
        # 如果未指定天数，使用序列长度的2倍
        if days is None:
            days = max(60, self.seq_length * 2)

        # 获取两只股票的数据
        df_a = self.fetch_stock_data(stock_code_a, days, end_date)
        df_b = self.fetch_stock_data(stock_code_b, days, end_date)

        # 重命名列以区分两只股票
        df_a_renamed = df_a.drop(columns=['stock_code']).add_prefix('A_')
        df_b_renamed = df_b.drop(columns=['stock_code']).add_prefix('B_')

        # 合并数据集
        merged_df = pd.concat([df_a_renamed, df_b_renamed], axis=1)

        # 填充缺失值
        merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

        # 计算价格比值
        merged_df['ratio'] = merged_df['A_close'] / merged_df['B_close']

        # 移除仍然包含NaN的行
        merged_df = merged_df.dropna()

        # 记录当前股票对
        self.last_pair = (stock_code_a, stock_code_b)

        # 调用特征工程函数
        feature_df = self.engineer_features(merged_df)

        logger.info(f"准备预测数据: {stock_code_a}/{stock_code_b}, 共 {len(feature_df)} 条有效记录")

        return feature_df

    def engineer_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20, 30]) -> pd.DataFrame:
        """
        为价格比值数据创建特征
        
        参数:
            df: 输入DataFrame，包含基本价格数据
            window_sizes: 用于计算技术指标的窗口大小列表
            
        返回:
            添加了特征的DataFrame
        """
        # 创建新的DataFrame以保存特征
        feature_df = df.copy()

        # 计算价格比值变化率
        feature_df['ratio_change'] = feature_df['ratio'].pct_change()

        # 添加时间特征
        feature_df['day_of_week'] = feature_df.index.dayofweek
        feature_df['month'] = feature_df.index.month
        feature_df['quarter'] = feature_df.index.quarter

        # 为A和B股票分别计算技术指标
        for prefix in ['A', 'B']:
            # 价格变化率
            feature_df[f'{prefix}_return'] = feature_df[f'{prefix}_close'].pct_change()

            # 波动性指标
            feature_df[f'{prefix}_volatility'] = feature_df[f'{prefix}_return'].rolling(window=20).std()

            # 移动平均线
            for window in window_sizes:
                feature_df[f'{prefix}_ma{window}'] = feature_df[f'{prefix}_close'].rolling(window=window).mean()
                feature_df[f'{prefix}_ma{window}_ratio'] = feature_df[f'{prefix}_close'] / feature_df[
                    f'{prefix}_ma{window}']

            # 相对强弱指标(RSI)
            delta = feature_df[f'{prefix}_close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            feature_df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))

            # 交易量变化
            feature_df[f'{prefix}_volume_change'] = feature_df[f'{prefix}_volume'].pct_change()

            # 布林带
            for window in [20]:
                feature_df[f'{prefix}_bb_middle'] = feature_df[f'{prefix}_close'].rolling(window=window).mean()
                feature_df[f'{prefix}_bb_std'] = feature_df[f'{prefix}_close'].rolling(window=window).std()
                feature_df[f'{prefix}_bb_upper'] = feature_df[f'{prefix}_bb_middle'] + 2 * feature_df[
                    f'{prefix}_bb_std']
                feature_df[f'{prefix}_bb_lower'] = feature_df[f'{prefix}_bb_middle'] - 2 * feature_df[
                    f'{prefix}_bb_std']
                feature_df[f'{prefix}_bb_width'] = (feature_df[f'{prefix}_bb_upper'] - feature_df[
                    f'{prefix}_bb_lower']) / feature_df[f'{prefix}_bb_middle']

            # 价格位置指标
            feature_df[f'{prefix}_close_to_high'] = feature_df[f'{prefix}_close'] / feature_df[f'{prefix}_high']
            feature_df[f'{prefix}_close_to_low'] = feature_df[f'{prefix}_close'] / feature_df[f'{prefix}_low']

        # 计算两股票之间的相对指标
        for window in window_sizes:
            feature_df[f'ma{window}_ratio'] = feature_df['ratio'].rolling(window=window).mean()
            feature_df[f'ratio_to_ma{window}'] = feature_df['ratio'] / feature_df[f'ma{window}_ratio']

        # 为价格比值计算专门的技术指标

        # 比值RSI
        delta_ratio = feature_df['ratio'].diff()
        gain_ratio = delta_ratio.where(delta_ratio > 0, 0).rolling(window=14).mean()
        loss_ratio = -delta_ratio.where(delta_ratio < 0, 0).rolling(window=14).mean()
        rs_ratio = gain_ratio / loss_ratio
        feature_df['ratio_rsi'] = 100 - (100 / (1 + rs_ratio))

        # 比值动量
        for window in [5, 10, 20]:
            feature_df[f'ratio_momentum_{window}'] = feature_df['ratio'] - feature_df['ratio'].shift(window)

        # 比值波动率
        feature_df['ratio_volatility'] = feature_df['ratio_change'].rolling(window=20).std()

        # 比值MACD
        feature_df['ratio_ema12'] = feature_df['ratio'].ewm(span=12, adjust=False).mean()
        feature_df['ratio_ema26'] = feature_df['ratio'].ewm(span=26, adjust=False).mean()
        feature_df['ratio_macd'] = feature_df['ratio_ema12'] - feature_df['ratio_ema26']
        feature_df['ratio_macd_signal'] = feature_df['ratio_macd'].ewm(span=9, adjust=False).mean()
        feature_df['ratio_macd_hist'] = feature_df['ratio_macd'] - feature_df['ratio_macd_signal']

        # 比值布林带
        feature_df['ratio_bb_middle'] = feature_df['ratio'].rolling(window=20).mean()
        feature_df['ratio_bb_std'] = feature_df['ratio'].rolling(window=20).std()
        feature_df['ratio_bb_upper'] = feature_df['ratio_bb_middle'] + 2 * feature_df['ratio_bb_std']
        feature_df['ratio_bb_lower'] = feature_df['ratio_bb_middle'] - 2 * feature_df['ratio_bb_std']
        feature_df['ratio_bb_width'] = (feature_df['ratio_bb_upper'] - feature_df['ratio_bb_lower']) / feature_df[
            'ratio_bb_middle']
        feature_df['ratio_bb_pct'] = (feature_df['ratio'] - feature_df['ratio_bb_lower']) / (
                feature_df['ratio_bb_upper'] - feature_df['ratio_bb_lower'])

        # Z-score
        feature_df['ratio_zscore'] = (feature_df['ratio'] - feature_df['ratio'].rolling(window=20).mean()) / feature_df[
            'ratio'].rolling(window=20).std()

        # 相对强度指标 (两股票的相对强度)
        feature_df['relative_strength'] = feature_df['A_close'].pct_change(20) - feature_df['B_close'].pct_change(20)

        # 删除包含NaN的行
        feature_df = feature_df.dropna()

        # 保存特征列名
        self.feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'ratio' in self.feature_cols:
            self.feature_cols.remove('ratio')

        return feature_df

    def create_sequence(self,
                        df: pd.DataFrame,
                        feature_cols: List[str] = None) -> np.ndarray:
        """
        为预测创建输入序列
        
        参数:
            df: 特征DataFrame
            feature_cols: 特征列，如果为None则使用所有数值列
            
        返回:
            模型输入序列
        """
        if feature_cols is None:
            if self.feature_cols is not None:
                feature_cols = self.feature_cols
            else:
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'ratio' in feature_cols:
                    feature_cols.remove('ratio')

        # 确保有足够的数据
        if len(df) < self.seq_length:
            raise ValueError(f"数据不足，需要至少 {self.seq_length} 条记录，但只有 {len(df)} 条")

        # 获取最近的序列
        data = df[feature_cols].values
        input_seq = data[-self.seq_length:].reshape(1, self.seq_length, len(feature_cols))

        return input_seq

    def predict(self,
                stock_code_a: str,
                stock_code_b: str,
                days_to_predict: int = None,
                end_date: str = None) -> Dict[str, Any]:
        """
        预测两只股票的价格比值
        
        参数:
            stock_code_a: 股票A代码
            stock_code_b: 股票B代码
            days_to_predict: 预测天数，默认使用模型预测时间跨度
            end_date: 结束日期，即预测的起始日期
            
        返回:
            预测结果字典
        """
        if not self.model_loaded:
            logger.error("模型未加载")
            return {"error": "模型未加载"}

        if days_to_predict is None:
            days_to_predict = self.forecast_horizon

        try:
            start_time = time.time()

            # 准备数据
            feature_df = self.prepare_prediction_data(stock_code_a, stock_code_b, days=None, end_date=end_date)

            # 提取日期
            dates = feature_df.index.tolist()
            last_date = dates[-1]

            # 创建预测序列
            input_seq = self.create_sequence(feature_df)

            # 缩放输入数据
            if 'feature_scaler' in self.scalers:
                n_features = input_seq.shape[2]
                input_seq_reshaped = input_seq.reshape(-1, n_features)
                input_seq_scaled = self.scalers['feature_scaler'].transform(input_seq_reshaped)
                input_seq_scaled = input_seq_scaled.reshape(1, self.seq_length, n_features)
            else:
                input_seq_scaled = input_seq

            # 执行预测
            prediction_scaled = self.model.predict(input_seq_scaled)

            # 反向缩放
            if 'target_scaler' in self.scalers:
                prediction = self.scalers['target_scaler'].inverse_transform(prediction_scaled)
            else:
                prediction = prediction_scaled

            # 生成未来日期
            future_dates = []
            current_date = last_date
            for i in range(1, days_to_predict + 1):
                next_date = current_date + timedelta(days=1)
                # 跳过周末
                while next_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                    next_date += timedelta(days=1)
                future_dates.append(next_date)
                current_date = next_date

            # 构建结果
            historical_values = feature_df['ratio'].values.tolist()[-30:]  # 最近30天数据
            historical_dates = [d.strftime('%Y-%m-%d') for d in dates[-30:]]

            prediction_values = prediction.flatten().tolist()
            future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]

            # 为结果添加置信区间
            # 这里使用基本的方法，实际应该基于模型不确定性估计
            std_dev = np.std(historical_values)
            confidence_factor = 1.96  # 95% 置信区间
            upper_bound = [float(val + std_dev * confidence_factor) for val in prediction_values]
            lower_bound = [float(val - std_dev * confidence_factor) for val in prediction_values]

            # 计算预测趋势
            if len(prediction_values) >= 2:
                first_val = prediction_values[0]
                last_val = prediction_values[-1]
                change_rate = (last_val - first_val) / first_val if first_val != 0 else 0

                if abs(change_rate) < 0.01:
                    forecast_trend = "stable"
                else:
                    forecast_trend = "up" if change_rate > 0 else "down"
            else:
                forecast_trend = "stable"

            # 计算风险级别
            relative_width = np.mean([u - l for u, l in zip(upper_bound, lower_bound)]) / np.mean(
                prediction_values) if prediction_values else 0
            if relative_width > 0.2:
                risk_level = "high"
            elif relative_width > 0.1:
                risk_level = "medium"
            else:
                risk_level = "low"

            # 性能评估（基于历史数据）
            performance = {
                "mean": float(np.mean(historical_values)),
                "std": float(np.std(historical_values)),
                "min": float(np.min(historical_values)),
                "max": float(np.max(historical_values)),
                "current": float(historical_values[-1])
            }

            result = {
                "dates": future_dates_str,
                "values": [float(val) for val in prediction_values],
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "historical_dates": historical_dates,
                "historical_values": [float(val) for val in historical_values],
                "risk_level": risk_level,
                "forecast_trend": forecast_trend,
                "performance": performance,
                "prediction_time": float(time.time() - start_time)
            }

            logger.info(f"预测完成，耗时: {result['prediction_time']:.2f}秒")
            return result

        except Exception as e:
            logger.error(f"预测失败: {e}")
            logger.error(traceback.format_exc())
            return {"error": str(e)}


# 全局预测器实例
_PREDICTOR_INSTANCE = None


def get_predictor(model_dir: str = 'trained_models', default_model: str = None) -> StockRatioPredictor:
    """
    获取全局预测器实例
    
    参数:
        model_dir: 模型目录
        default_model: 默认模型名称
        
    返回:
        预测器实例
    """
    global _PREDICTOR_INSTANCE

    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = StockRatioPredictor(model_dir, default_model)

    return _PREDICTOR_INSTANCE


if __name__ == "__main__":
    # 测试预测功能
    predictor = StockRatioPredictor()

    # 加载模型（如果没有自动加载）
    if not predictor.model_loaded:
        print("手动加载最新模型")
        predictor._load_latest_model()

    # 执行预测
    result = predictor.predict('399001', '399107', days_to_predict=30)

    if "error" in result:
        print(f"预测失败: {result['error']}")
    else:
        print("\n========== 预测结果摘要 ==========")
        print(f"预测日期: {result['dates']}")
        print(f"预测值: {result['values']}")
        print(f"风险级别: {result['risk_level']}")
        print(f"预测趋势: {result['forecast_trend']}")
        print(f"预测耗时: {result['prediction_time']:.2f}秒")
        print("===================================\n")
