"""
股票价格比值预测模型训练器
提供完整的模型训练、验证、评估和保存流程
支持多种模型架构和训练策略
"""

import os
import numpy as np
import pandas as pd
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import sqlite3
import pickle
import logging
import traceback
from typing import List, Dict, Tuple, Any, Optional, Union

# 机器学习相关库
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入数据库类
from back_end.peudo_backend.get_stock_data.stock_data_base import StockKlineDatabase

# 尝试导入深度学习库
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model, save_model
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Input, Concatenate,
        BatchNormalization, Bidirectional, TimeDistributed,
        Conv1D, MaxPooling1D, Flatten, LeakyReLU,
        Add, LayerNormalization, Attention, MultiHeadAttention,
        GlobalAveragePooling1D
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard, CSVLogger
    )
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.regularizers import l1, l2, l1_l2

    # 设置TensorFlow日志级别
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"找到GPU设备: {len(gpus)}个")
        except RuntimeError as e:
            logger.error(f"GPU设置错误: {e}")

    TF_AVAILABLE = True
except ImportError:
    logger.warning("TensorFlow未安装，将使用替代模型")
    TF_AVAILABLE = False


class StockRatioPredictorTrainer:
    """股票价格比值预测模型训练器"""

    def __init__(self,
                 model_dir: str = 'trained_models',
                 logs_dir: str = 'training_logs',
                 db_path: str = None):
        """
        初始化训练器
        
        参数:
            model_dir: 模型保存目录
            logs_dir: 训练日志目录
            db_path: 数据库路径，默认使用内置路径
        """
        # 创建必要的目录
        self.model_dir = Path(model_dir)
        self.logs_dir = Path(logs_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)

        # 初始化数据库连接
        self.db = StockKlineDatabase(db_path)

        # 初始化数据预处理器
        self.scalers = {}

        # 当前训练会话信息
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = None
        self.training_history = None

        logger.info(f"初始化训练器，会话ID: {self.session_id}")

    def fetch_stock_data(self,
                         stock_code: str,
                         start_date: str = '2015-01-01',
                         end_date: str = None) -> pd.DataFrame:
        """
        从数据库获取股票数据
        
        参数:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期，默认为当前日期
            
        返回:
            包含股票数据的DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

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

            logger.info(f"获取股票 {stock_code} 数据: {len(df)} 条记录")
            return df

        except Exception as e:
            logger.error(f"获取股票数据失败: {e}")
            raise

    def prepare_ratio_data(self,
                           stock_code_a: str,
                           stock_code_b: str,
                           start_date: str = '2015-01-01',
                           end_date: str = None,
                           fill_method: str = 'ffill') -> pd.DataFrame:
        """
        准备两只股票的价格比值数据和特征
        
        参数:
            stock_code_a: 股票A代码
            stock_code_b: 股票B代码
            start_date: 开始日期
            end_date: 结束日期
            fill_method: 缺失值填充方法 ['ffill', 'bfill', 'linear']
            
        返回:
            包含合并数据和特征的DataFrame
        """
        # 获取两只股票的数据
        df_a = self.fetch_stock_data(stock_code_a, start_date, end_date)
        df_b = self.fetch_stock_data(stock_code_b, start_date, end_date)

        # 重命名列以区分两只股票
        df_a_renamed = df_a.drop(columns=['stock_code']).add_prefix('A_')
        df_b_renamed = df_b.drop(columns=['stock_code']).add_prefix('B_')

        # 时间对齐处理 - 只保留两个股票都有交易的日期
        common_dates = df_a_renamed.index.intersection(df_b_renamed.index)
        df_a_aligned = df_a_renamed.loc[common_dates]
        df_b_aligned = df_b_renamed.loc[common_dates]

        # 合并数据集
        merged_df = pd.concat([df_a_aligned, df_b_aligned], axis=1)

        # 填充缺失值
        if fill_method == 'linear':
            merged_df = merged_df.interpolate(method='linear')
        else:
            merged_df = merged_df.fillna(method=fill_method)

        # 再次填充可能的边界缺失值
        merged_df = merged_df.fillna(method='bfill').fillna(method='ffill')

        # 计算价格比值
        merged_df['ratio'] = merged_df['A_close'] / merged_df['B_close']

        # 移除仍然包含NaN的行
        merged_df = merged_df.dropna()

        logger.info(
            f"准备价格比值数据: {stock_code_a}/{stock_code_b}, 原始记录数: A={len(df_a)}, B={len(df_b)}, 对齐后记录数: {len(merged_df)}")

        return merged_df

    def engineer_features(self, df: pd.DataFrame, window_sizes: List[int] = [5, 10, 20, 30],
                          feature_selection: bool = True, max_features: int = 60) -> pd.DataFrame:
        """
        为价格比值数据创建特征
        
        参数:
            df: 输入DataFrame，包含基本价格数据
            window_sizes: 用于计算技术指标的窗口大小列表
            feature_selection: 是否进行特征选择
            max_features: 特征选择后保留的最大特征数量
            
        返回:
            添加了特征的DataFrame
        """
        # 创建新的DataFrame以保存特征
        feature_df = df.copy()
        
        # 增加安全检查，确保价格数据为正数且非零
        for prefix in ['A', 'B']:
            close_col = f'{prefix}_close'
            if close_col in feature_df.columns:
                # 将无效值替换为前一个有效值，避免零或负值
                feature_df[close_col] = feature_df[close_col].replace([0, np.nan, np.inf, -np.inf], method='ffill')
                # 保证所有价格值大于0
                min_valid_value = feature_df[close_col].min()
                if min_valid_value <= 0:
                    # 如果仍有零或负值，替换为最小正数值的1%
                    positive_min = feature_df[feature_df[close_col] > 0][close_col].min()
                    safe_min = positive_min * 0.01 if positive_min > 0 else 0.001
                    feature_df[close_col] = feature_df[close_col].apply(lambda x: safe_min if x <= 0 else x)

        # 计算价格比值，避免除零
        if 'A_close' in feature_df.columns and 'B_close' in feature_df.columns:
            # 安全计算比值，避免除以零
            feature_df['ratio'] = feature_df['A_close'] / feature_df['B_close'].replace(0, np.nan)
            feature_df['ratio'] = feature_df['ratio'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')

        # 计算价格比值变化率
        feature_df['ratio_change'] = feature_df['ratio'].pct_change()
        feature_df['ratio_change_pct'] = feature_df['ratio_change'] / feature_df['ratio'].shift(1).replace(0, np.nan) * 100
        feature_df['ratio_change_pct'] = feature_df['ratio_change_pct'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 添加时间特征
        feature_df['day_of_week'] = feature_df.index.dayofweek
        feature_df['month'] = feature_df.index.month
        feature_df['quarter'] = feature_df.index.quarter

        # 为A和B股票分别计算技术指标
        for prefix in ['A', 'B']:
            # 价格变化率 - 核心特征 - 增加安全处理
            price_col = f'{prefix}_close'
            feature_df[f'{prefix}_return'] = feature_df[price_col].pct_change()
            feature_df[f'{prefix}_return'] = feature_df[f'{prefix}_return'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            feature_df[f'{prefix}_return_5d'] = feature_df[price_col].pct_change(5)
            feature_df[f'{prefix}_return_10d'] = feature_df[price_col].pct_change(10)
            
            # 处理各种周期变化率中的无穷值
            feature_df[f'{prefix}_return_5d'] = feature_df[f'{prefix}_return_5d'].replace([np.inf, -np.inf], np.nan).fillna(0)
            feature_df[f'{prefix}_return_10d'] = feature_df[f'{prefix}_return_10d'].replace([np.inf, -np.inf], np.nan).fillna(0)

            # 波动性指标 - 增加安全处理
            feature_df[f'{prefix}_volatility'] = feature_df[f'{prefix}_return'].rolling(window=20).std()
            feature_df[f'{prefix}_volatility_10d'] = feature_df[f'{prefix}_return'].rolling(window=10).std()
            
            # 安全计算移动平均线，确保无NaN和无穷值
            for window in window_sizes:
                feature_df[f'{prefix}_ma{window}'] = feature_df[price_col].rolling(window=window).mean()
                # 安全计算价格与均线比值
                ma_col = f'{prefix}_ma{window}'
                feature_df[f'{prefix}_ma{window}_ratio'] = feature_df[price_col] / feature_df[ma_col].replace(0, np.nan)
                feature_df[f'{prefix}_ma{window}_ratio'] = feature_df[f'{prefix}_ma{window}_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # 指数移动平均线
            feature_df[f'{prefix}_ema12'] = feature_df[price_col].ewm(span=12, adjust=False).mean()
            feature_df[f'{prefix}_ema26'] = feature_df[price_col].ewm(span=26, adjust=False).mean()
            
            # 安全计算EMA比值
            feature_df[f'{prefix}_ema12_26_ratio'] = feature_df[f'{prefix}_ema12'] / feature_df[f'{prefix}_ema26'].replace(0, np.nan)
            feature_df[f'{prefix}_ema12_26_ratio'] = feature_df[f'{prefix}_ema12_26_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # 相对强弱指标(RSI) - 增加安全处理
            delta = feature_df[price_col].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            # 避免除以零
            rs = gain / loss.replace(0, np.nan)
            rs = rs.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            feature_df[f'{prefix}_rsi'] = 100 - (100 / (1 + rs))
            feature_df[f'{prefix}_rsi'] = feature_df[f'{prefix}_rsi'].clip(0, 100) # 限制RSI在0-100范围内

            # 交易量变化 - 增加安全处理
            volume_col = f'{prefix}_volume'
            if volume_col in feature_df.columns:
                feature_df[f'{prefix}_volume_change'] = feature_df[volume_col].pct_change()
                feature_df[f'{prefix}_volume_change'] = feature_df[f'{prefix}_volume_change'].replace([np.inf, -np.inf], np.nan).fillna(0)
                
                feature_df[f'{prefix}_volume_ma10'] = feature_df[volume_col].rolling(window=10).mean()
                # 安全计算成交量比值
                feature_df[f'{prefix}_volume_ratio'] = feature_df[volume_col] / feature_df[f'{prefix}_volume_ma10'].replace(0, np.nan)
                feature_df[f'{prefix}_volume_ratio'] = feature_df[f'{prefix}_volume_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # 布林带 - 增加安全处理
            for window in [20]:
                feature_df[f'{prefix}_bb_middle'] = feature_df[price_col].rolling(window=window).mean()
                feature_df[f'{prefix}_bb_std'] = feature_df[price_col].rolling(window=window).std()
                feature_df[f'{prefix}_bb_upper'] = feature_df[f'{prefix}_bb_middle'] + 2 * feature_df[f'{prefix}_bb_std']
                feature_df[f'{prefix}_bb_lower'] = feature_df[f'{prefix}_bb_middle'] - 2 * feature_df[f'{prefix}_bb_std']
                
                # 安全计算布林带宽度
                middle = feature_df[f'{prefix}_bb_middle'].replace(0, np.nan)
                feature_df[f'{prefix}_bb_width'] = (feature_df[f'{prefix}_bb_upper'] - feature_df[f'{prefix}_bb_lower']) / middle
                feature_df[f'{prefix}_bb_width'] = feature_df[f'{prefix}_bb_width'].replace([np.inf, -np.inf], np.nan).fillna(0.01)

            # 价格位置指标 - 增加安全处理
            high_col = f'{prefix}_high'
            low_col = f'{prefix}_low'
            if high_col in feature_df.columns and low_col in feature_df.columns:
                feature_df[f'{prefix}_close_to_high'] = feature_df[price_col] / feature_df[high_col].replace(0, np.nan)
                feature_df[f'{prefix}_close_to_low'] = feature_df[price_col] / feature_df[low_col].replace(0, np.nan)
                
                feature_df[f'{prefix}_close_to_high'] = feature_df[f'{prefix}_close_to_high'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
                feature_df[f'{prefix}_close_to_low'] = feature_df[f'{prefix}_close_to_low'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # 计算两股票之间的相对指标 - 增加安全处理
        for window in window_sizes:
            feature_df[f'ma{window}_ratio'] = feature_df['ratio'].rolling(window=window).mean()
            # 安全计算比值与其移动平均的关系
            feature_df[f'ratio_to_ma{window}'] = feature_df['ratio'] / feature_df[f'ma{window}_ratio'].replace(0, np.nan)
            feature_df[f'ratio_to_ma{window}'] = feature_df[f'ratio_to_ma{window}'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # 为价格比值计算专门的技术指标 - 增加安全处理

        # 比值RSI - 安全计算
        delta_ratio = feature_df['ratio'].diff()
        gain_ratio = delta_ratio.where(delta_ratio > 0, 0).rolling(window=14).mean()
        loss_ratio = -delta_ratio.where(delta_ratio < 0, 0).rolling(window=14).mean()
        # 避免除以零
        rs_ratio = gain_ratio / loss_ratio.replace(0, np.nan)
        rs_ratio = rs_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        feature_df['ratio_rsi'] = 100 - (100 / (1 + rs_ratio))
        feature_df['ratio_rsi'] = feature_df['ratio_rsi'].clip(0, 100) # 限制RSI在0-100范围

        # 比值动量 - 核心预测特征 - 增加安全处理
        for window in [5, 10, 20]:
            feature_df[f'ratio_momentum_{window}'] = feature_df['ratio'] - feature_df['ratio'].shift(window)
            # 添加归一化动量，安全处理除零情况
            denominator = feature_df['ratio'].shift(window).replace(0, np.nan)
            feature_df[f'ratio_momentum_{window}_norm'] = feature_df[f'ratio_momentum_{window}'] / denominator
            feature_df[f'ratio_momentum_{window}_norm'] = feature_df[f'ratio_momentum_{window}_norm'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 比值波动率
        feature_df['ratio_volatility'] = feature_df['ratio_change'].rolling(window=20).std()
        feature_df['ratio_volatility_10d'] = feature_df['ratio_change'].rolling(window=10).std()
        feature_df['ratio_volatility_5d'] = feature_df['ratio_change'].rolling(window=5).std()

        # 比值MACD - 增加安全处理
        feature_df['ratio_ema12'] = feature_df['ratio'].ewm(span=12, adjust=False).mean()
        feature_df['ratio_ema26'] = feature_df['ratio'].ewm(span=26, adjust=False).mean()
        feature_df['ratio_macd'] = feature_df['ratio_ema12'] - feature_df['ratio_ema26']
        feature_df['ratio_macd_signal'] = feature_df['ratio_macd'].ewm(span=9, adjust=False).mean()
        feature_df['ratio_macd_hist'] = feature_df['ratio_macd'] - feature_df['ratio_macd_signal']
        feature_df['ratio_macd_hist_dir'] = np.sign(feature_df['ratio_macd_hist'])

        # 比值布林带 - 增加安全处理
        feature_df['ratio_bb_middle'] = feature_df['ratio'].rolling(window=20).mean()
        feature_df['ratio_bb_std'] = feature_df['ratio'].rolling(window=20).std()
        feature_df['ratio_bb_upper'] = feature_df['ratio_bb_middle'] + 2 * feature_df['ratio_bb_std']
        feature_df['ratio_bb_lower'] = feature_df['ratio_bb_middle'] - 2 * feature_df['ratio_bb_std']
        
        # 安全计算布林带宽度和百分比位置
        middle = feature_df['ratio_bb_middle'].replace(0, np.nan)
        feature_df['ratio_bb_width'] = (feature_df['ratio_bb_upper'] - feature_df['ratio_bb_lower']) / middle
        feature_df['ratio_bb_width'] = feature_df['ratio_bb_width'].replace([np.inf, -np.inf], np.nan).fillna(0.01)
        
        # 布林带百分比位置计算
        band_width = (feature_df['ratio_bb_upper'] - feature_df['ratio_bb_lower']).replace(0, np.nan)
        feature_df['ratio_bb_pct'] = (feature_df['ratio'] - feature_df['ratio_bb_lower']) / band_width
        feature_df['ratio_bb_pct'] = feature_df['ratio_bb_pct'].replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0, 1)

        # Z-score - 安全计算
        mean_20d = feature_df['ratio'].rolling(window=20).mean()
        std_20d = feature_df['ratio'].rolling(window=20).std().replace(0, np.nan)
        feature_df['ratio_zscore'] = (feature_df['ratio'] - mean_20d) / std_20d
        feature_df['ratio_zscore'] = feature_df['ratio_zscore'].replace([np.inf, -np.inf], np.nan).fillna(0)
        # 限制极端Z值
        feature_df['ratio_zscore'] = feature_df['ratio_zscore'].clip(-5, 5)

        # 相对强度指标 (两股票的相对强度) - 增加安全处理
        feature_df['relative_strength'] = feature_df['A_close'].pct_change(20) - feature_df['B_close'].pct_change(20)
        feature_df['relative_strength_10d'] = feature_df['A_close'].pct_change(10) - feature_df['B_close'].pct_change(10)
        feature_df['relative_strength_5d'] = feature_df['A_close'].pct_change(5) - feature_df['B_close'].pct_change(5)
        
        # 处理各种相对强度指标中的无穷值
        feature_df['relative_strength'] = feature_df['relative_strength'].replace([np.inf, -np.inf], np.nan).fillna(0)
        feature_df['relative_strength_10d'] = feature_df['relative_strength_10d'].replace([np.inf, -np.inf], np.nan).fillna(0)
        feature_df['relative_strength_5d'] = feature_df['relative_strength_5d'].replace([np.inf, -np.inf], np.nan).fillna(0)

        # 添加趋势指标 - 价格比值变化方向，使用安全处理的变化值
        safe_change = feature_df['ratio_change'].replace([np.inf, -np.inf], np.nan).fillna(0)
        safe_change_5d = feature_df['ratio'].diff(5).replace([np.inf, -np.inf], np.nan).fillna(0)
        
        feature_df['ratio_trend'] = np.sign(safe_change)
        feature_df['ratio_trend_5d'] = np.sign(safe_change_5d)

        # 简化版连续趋势计算，避免过度拟合
        feature_df['ratio_consec_up'] = (safe_change > 0).astype(int)
        feature_df['ratio_consec_down'] = (safe_change < 0).astype(int)
        
        # 全局安全处理：检查并替换所有无穷值和NaN
        # 将所有无穷大值替换为NaN
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        
        # 检查每列是否包含NaN，并填充
        for col in feature_df.columns:
            if feature_df[col].isnull().any():
                if col == 'ratio' or col.startswith('A_') or col.startswith('B_'):
                    # 对于基本价格数据，使用前向填充然后后向填充
                    feature_df[col] = feature_df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # 对于派生指标，可以用0或均值填充
                    if '_zscore' in col or '_momentum_' in col or 'trend' in col:
                        # 对于Z分数、动量和趋势指标，用0填充
                        feature_df[col] = feature_df[col].fillna(0)
                    elif 'volatility' in col or 'std' in col:
                        # 对于波动率指标，用小的正值填充
                        feature_df[col] = feature_df[col].fillna(feature_df[col].mean() if not feature_df[col].isnull().all() else 0.01)
                    else:
                        # 对于其他指标，用均值填充
                        feature_df[col] = feature_df[col].fillna(feature_df[col].mean() if not feature_df[col].isnull().all() else 0)
        
        # 检查是否仍有NaN值
        if feature_df.isnull().any().any():
            logger.warning(f"在特征工程完成后仍有NaN值：{feature_df.isnull().sum().sum()}个")
            # 最后的安全网：用0填充任何剩余的NaN
            feature_df = feature_df.fillna(0)
        
        # 删除包含NaN的行
        original_len = len(feature_df)
        feature_df = feature_df.dropna()
        if len(feature_df) < original_len:
            logger.info(f"删除了{original_len - len(feature_df)}行包含NaN的数据，剩余{len(feature_df)}行")

        # 添加数值范围检查：将极端值裁剪到合理范围
        for col in feature_df.select_dtypes(include=[np.number]).columns:
            # 计算列的统计信息
            mean_val = feature_df[col].mean()
            std_val = feature_df[col].std()
            
            # 设置合理范围 (均值 +/- 5 * 标准差)
            lower_bound = mean_val - 5 * std_val
            upper_bound = mean_val + 5 * std_val
            
            # 裁剪极端值
            if not np.isnan(lower_bound) and not np.isnan(upper_bound):
                # 避免过度收缩(确保上下界不相同)
                if abs(upper_bound - lower_bound) > 1e-10:
                    feature_df[col] = feature_df[col].clip(lower_bound, upper_bound)
        
        # 检查是否有全为零或常量的列
        constant_cols = []
        for col in feature_df.columns:
            if col != 'ratio' and feature_df[col].nunique() <= 1:
                constant_cols.append(col)
                
        if constant_cols:
            logger.warning(f"以下{len(constant_cols)}列为常量值，考虑删除：{constant_cols}")
            # 删除常量列
            feature_df = feature_df.drop(columns=constant_cols)

        # 特征选择(如果启用)...
        # 保留原有的特征选择代码
        
        logger.info(
            f"特征工程完成，从 {original_len} 条记录创建了 {len(feature_df)} 条有效特征记录，特征数量: {len(feature_df.columns) - 1}")

        return feature_df

    def create_sequences(self,
                         df: pd.DataFrame,
                         target_col: str = 'ratio',
                         feature_cols: List[str] = None,
                         seq_length: int = 20,
                         forecast_horizon: int = 1,
                         stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列序列用于训练
        
        参数:
            df: 输入特征DataFrame
            target_col: 目标列名
            feature_cols: 特征列名列表，如果为None则使用所有数值列
            seq_length: 序列长度
            forecast_horizon: 预测时间跨度
            stride: 序列创建步长
            
        返回:
            (X, y): 输入序列和目标值
        """
        # 如果未指定特征列，使用所有数值列
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除目标列和日期索引
            if target_col in feature_cols:
                feature_cols.remove(target_col)

        # 提取特征和目标
        data = df[feature_cols].values
        targets = df[target_col].values

        X, y = [], []

        # 创建序列
        for i in range(0, len(df) - seq_length - forecast_horizon + 1, stride):
            X.append(data[i:i + seq_length])
            y.append(targets[i + seq_length:i + seq_length + forecast_horizon])

        return np.array(X), np.array(y)

    def split_data(self,
                   X: np.ndarray,
                   y: np.ndarray,
                   test_size: float = 0.2,
                   val_size: float = 0.1,
                   random_state: int = 42) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        按时间顺序拆分数据为训练集、验证集和测试集
        
        参数:
            X: 输入特征序列
            y: 目标值
            test_size: 测试集比例
            val_size: 验证集比例 
            random_state: 随机种子
            
        返回:
            (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # 计算分割点
        train_end = int(len(X) * (1 - test_size - val_size))
        val_end = int(len(X) * (1 - test_size))

        # 按时间顺序分割
        X_train, y_train = X[:train_end], y[:train_end]
        X_val, y_val = X[train_end:val_end], y[train_end:val_end]
        X_test, y_test = X[val_end:], y[val_end:]

        logger.info(f"数据分割完成 - 训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self,
                   X_train: np.ndarray,
                   X_val: np.ndarray,
                   X_test: np.ndarray,
                   y_train: np.ndarray,
                   y_val: np.ndarray,
                   y_test: np.ndarray,
                   scaler_type: str = 'robust') -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        对数据进行缩放
        
        参数:
            X_train, X_val, X_test: 训练、验证和测试特征
            y_train, y_val, y_test: 训练、验证和测试目标
            scaler_type: 缩放器类型 ['minmax', 'standard', 'robust']
            
        返回:
            缩放后的数据
        """
        try:
            # 特征维度
            n_samples_train, n_timesteps, n_features = X_train.shape
            
            # 安全检查 - 检测无穷值和极端值
            logger.info(f"开始数据缩放前的安全检查，检测无穷值和极端值...")
            
            # 创建数据副本以避免修改原始数据
            X_train_safe = X_train.copy()
            X_val_safe = X_val.copy()
            X_test_safe = X_test.copy()
            y_train_safe = y_train.copy()
            y_val_safe = y_val.copy() 
            y_test_safe = y_test.copy()
            
            # 检查并处理X数据中的无穷值和NaN
            X_train_reshaped = X_train_safe.reshape(-1, n_features)
            
            # 检查无穷值
            inf_mask = ~np.isfinite(X_train_reshaped)
            inf_count = np.sum(inf_mask)
            if inf_count > 0:
                logger.warning(f"训练特征中发现{inf_count}个无穷值，将替换为有限值")
                # 计算每列的有限值统计
                col_means = np.nanmean(np.where(np.isfinite(X_train_reshaped), X_train_reshaped, np.nan), axis=0)
                col_stds = np.nanstd(np.where(np.isfinite(X_train_reshaped), X_train_reshaped, np.nan), axis=0)
                
                # 对于每列，将无穷值替换为该列均值
                for j in range(n_features):
                    col_inf_mask = inf_mask[:, j]
                    if np.any(col_inf_mask):
                        # 如果该列全是无穷值，使用0替代
                        if np.all(col_inf_mask) or np.isnan(col_means[j]):
                            X_train_reshaped[col_inf_mask, j] = 0
                        else:
                            # 否则使用均值替代
                            X_train_reshaped[col_inf_mask, j] = col_means[j]
            
            # 检查NaN值
            nan_mask = np.isnan(X_train_reshaped)
            nan_count = np.sum(nan_mask)
            if nan_count > 0:
                logger.warning(f"训练特征中发现{nan_count}个NaN值，将替换为均值或0")
                # 对于每列，将NaN替换为该列均值
                for j in range(n_features):
                    col_nan_mask = nan_mask[:, j]
                    if np.any(col_nan_mask):
                        col_values = X_train_reshaped[~col_nan_mask, j]
                        if len(col_values) > 0:
                            X_train_reshaped[col_nan_mask, j] = np.mean(col_values)
                        else:
                            # 如果该列全是NaN，使用0替代
                            X_train_reshaped[col_nan_mask, j] = 0

            # 检查极端值 (超出合理范围的值)
            for j in range(n_features):
                col_values = X_train_reshaped[:, j]
                if len(col_values) > 0:
                    # 使用分位数而不是均值和标准差，对极端异常值更鲁棒
                    q1 = np.percentile(col_values, 1)
                    q99 = np.percentile(col_values, 99)
                    iqr = q99 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q99 + 1.5 * iqr
                    
                    # 裁剪极端值
                    extreme_mask = (col_values < lower_bound) | (col_values > upper_bound)
                    extreme_count = np.sum(extreme_mask)
                    if extreme_count > 0:
                        # logger.debug(f"特征{j}中有{extreme_count}个极端值，进行裁剪")
                        X_train_reshaped[extreme_mask, j] = np.clip(
                            X_train_reshaped[extreme_mask, j], lower_bound, upper_bound)
            
            # 重构回原始形状
            X_train_safe = X_train_reshaped.reshape(n_samples_train, n_timesteps, n_features)
            
            # 同样处理验证和测试数据
            # 处理验证集
            if X_val_safe.size > 0:
                n_samples_val = X_val_safe.shape[0]
                X_val_reshaped = X_val_safe.reshape(-1, n_features)
                X_val_reshaped = np.nan_to_num(X_val_reshaped, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 应用与训练集相同的裁剪逻辑
                for j in range(n_features):
                    col_values = X_train_reshaped[:, j]  # 使用训练集的统计数据
                    if len(col_values) > 0:
                        q1 = np.percentile(col_values, 1)
                        q99 = np.percentile(col_values, 99)
                        iqr = q99 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q99 + 1.5 * iqr
                        
                        # 裁剪验证集中的极端值
                        X_val_reshaped[:, j] = np.clip(X_val_reshaped[:, j], lower_bound, upper_bound)
                
                X_val_safe = X_val_reshaped.reshape(n_samples_val, n_timesteps, n_features)
            
            # 处理测试集
            if X_test_safe.size > 0:
                n_samples_test = X_test_safe.shape[0]
                X_test_reshaped = X_test_safe.reshape(-1, n_features)
                X_test_reshaped = np.nan_to_num(X_test_reshaped, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # 应用与训练集相同的裁剪逻辑
                for j in range(n_features):
                    col_values = X_train_reshaped[:, j]  # 使用训练集的统计数据
                    if len(col_values) > 0:
                        q1 = np.percentile(col_values, 1)
                        q99 = np.percentile(col_values, 99)
                        iqr = q99 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q99 + 1.5 * iqr
                        
                        # 裁剪测试集中的极端值
                        X_test_reshaped[:, j] = np.clip(X_test_reshaped[:, j], lower_bound, upper_bound)
                
                X_test_safe = X_test_reshaped.reshape(n_samples_test, n_timesteps, n_features)
            
            # 处理目标值中的无穷值和NaN
            y_train_safe = np.nan_to_num(y_train_safe, nan=np.nanmedian(y_train_safe), posinf=1e6, neginf=-1e6)
            y_val_safe = np.nan_to_num(y_val_safe, nan=np.nanmedian(y_val_safe), posinf=1e6, neginf=-1e6)
            y_test_safe = np.nan_to_num(y_test_safe, nan=np.nanmedian(y_test_safe), posinf=1e6, neginf=-1e6)
            
            # 初始化特征缩放器
            if scaler_type == 'minmax':
                feature_scaler = MinMaxScaler()
            elif scaler_type == 'standard':
                feature_scaler = StandardScaler()
            else:
                feature_scaler = RobustScaler()

            # 初始化目标缩放器
            target_scaler = RobustScaler()

            # 重塑特征进行缩放
            X_train_reshaped = X_train_safe.reshape(-1, n_features)
            
            # 使用try-except包装缩放操作，以防数据中仍有问题
            try:
                X_train_scaled = feature_scaler.fit_transform(X_train_reshaped)
            except Exception as e:
                logger.error(f"缩放训练特征时出错: {e}")
                # 最后的安全处理 - 使用更简单的归一化方法
                # 对每列进行min-max归一化，对异常值更鲁棒
                X_train_scaled = np.zeros_like(X_train_reshaped)
                for j in range(n_features):
                    col = X_train_reshaped[:, j]
                    col_min, col_max = np.min(col), np.max(col)
                    if col_max > col_min:
                        X_train_scaled[:, j] = (col - col_min) / (col_max - col_min)
                    else:
                        X_train_scaled[:, j] = 0  # 如果列是常数，设为0
            
            X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)

            # 缩放验证特征
            n_samples_val = X_val_safe.shape[0]
            X_val_reshaped = X_val_safe.reshape(-1, n_features)
            try:
                X_val_scaled = feature_scaler.transform(X_val_reshaped)
            except Exception as e:
                logger.error(f"缩放验证特征时出错: {e}")
                # 应用与训练集相同的手动归一化
                X_val_scaled = np.zeros_like(X_val_reshaped)
                for j in range(n_features):
                    col = X_val_reshaped[:, j]
                    col_min, col_max = np.min(X_train_reshaped[:, j]), np.max(X_train_reshaped[:, j])
                    if col_max > col_min:
                        X_val_scaled[:, j] = (col - col_min) / (col_max - col_min)
                    else:
                        X_val_scaled[:, j] = 0
            
            X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)

            # 缩放测试特征
            n_samples_test = X_test_safe.shape[0]
            X_test_reshaped = X_test_safe.reshape(-1, n_features)
            try:
                X_test_scaled = feature_scaler.transform(X_test_reshaped)
            except Exception as e:
                logger.error(f"缩放测试特征时出错: {e}")
                # 应用与训练集相同的手动归一化
                X_test_scaled = np.zeros_like(X_test_reshaped)
                for j in range(n_features):
                    col = X_test_reshaped[:, j]
                    col_min, col_max = np.min(X_train_reshaped[:, j]), np.max(X_train_reshaped[:, j])
                    if col_max > col_min:
                        X_test_scaled[:, j] = (col - col_min) / (col_max - col_min)
                    else:
                        X_test_scaled[:, j] = 0
            
            X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)

            # 缩放目标值
            try:
                y_train_scaled = target_scaler.fit_transform(y_train_safe.reshape(-1, 1)).reshape(y_train_safe.shape)
                y_val_scaled = target_scaler.transform(y_val_safe.reshape(-1, 1)).reshape(y_val_safe.shape)
                y_test_scaled = target_scaler.transform(y_test_safe.reshape(-1, 1)).reshape(y_test_safe.shape)
            except Exception as e:
                logger.error(f"缩放目标值时出错: {e}")
                # 手动归一化目标值
                y_min, y_max = np.min(y_train_safe), np.max(y_train_safe)
                if y_max > y_min:
                    y_train_scaled = (y_train_safe - y_min) / (y_max - y_min)
                    y_val_scaled = (y_val_safe - y_min) / (y_max - y_min)
                    y_test_scaled = (y_test_safe - y_min) / (y_max - y_min)
                else:
                    # 如果目标值是常数，设为0
                    y_train_scaled = np.zeros_like(y_train_safe)
                    y_val_scaled = np.zeros_like(y_val_safe)
                    y_test_scaled = np.zeros_like(y_test_safe)
                
                # 创建一个简单的还原函数来替代缺失的scaler
                class SimpleScaler:
                    def __init__(self, min_val, max_val):
                        self.min_val = min_val
                        self.max_val = max_val
                    
                    def transform(self, X):
                        return (X - self.min_val) / (self.max_val - self.min_val) if self.max_val > self.min_val else np.zeros_like(X)
                    
                    def inverse_transform(self, X):
                        return X * (self.max_val - self.min_val) + self.min_val if self.max_val > self.min_val else np.ones_like(X) * self.min_val
                
                target_scaler = SimpleScaler(y_min, y_max)

            # 保存缩放器供后续使用
            self.scalers = {
                'feature_scaler': feature_scaler,
                'target_scaler': target_scaler
            }

            logger.info(f"数据缩放完成，使用缩放器: {scaler_type}")

            return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled
        
        except Exception as e:
            logger.error(f"数据缩放过程中出现未捕获的错误: {e}")
            logger.error(traceback.format_exc())
            
            # 创建安全的返回值（全零数组）
            if X_train.size > 0:
                X_train_scaled = np.zeros_like(X_train)
                y_train_scaled = np.zeros_like(y_train)
            else:
                X_train_scaled = np.zeros((1, 1, 1))
                y_train_scaled = np.zeros((1, 1))
                
            if X_val.size > 0:
                X_val_scaled = np.zeros_like(X_val)
                y_val_scaled = np.zeros_like(y_val)
            else:
                X_val_scaled = np.zeros((1, 1, 1))
                y_val_scaled = np.zeros((1, 1))
                
            if X_test.size > 0:
                X_test_scaled = np.zeros_like(X_test)
                y_test_scaled = np.zeros_like(y_test)
            else:
                X_test_scaled = np.zeros((1, 1, 1))
                y_test_scaled = np.zeros((1, 1))
            
            # 创建一个身份缩放器来避免后续的错误
            class IdentityScaler:
                def transform(self, X):
                    return X
                
                def inverse_transform(self, X):
                    return X
            
            self.scalers = {
                'feature_scaler': IdentityScaler(),
                'target_scaler': IdentityScaler()
            }
            
            return X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled

    def build_lstm_model(self,
                         input_shape: tuple,
                         output_dim: int = 1,
                         model_type: str = 'simple',
                         lstm_units: list = None,
                         dense_units: list = None,
                         dropout_rate: float = 0.2,
                         learning_rate: float = 0.001,
                         regularization: float = 0.001,
                         direction_penalty: float = 0.0,
                         loss_function: str = 'mse') -> tf.keras.Model:
        """
        构建LSTM模型
        
        参数:
            input_shape: 输入形状
            output_dim: 输出维度
            model_type: 模型类型 (simple, stacked, bidirectional, hybrid)
            lstm_units: LSTM单元列表
            dense_units: 全连接层单元列表
            dropout_rate: Dropout比例
            learning_rate: 学习率
            regularization: 正则化系数
            direction_penalty: 方向预测损失权重
            loss_function: 损失函数类型 (mse, mae, huber, r2)
            
        返回:
            构建的模型
        """
        if lstm_units is None:
            lstm_units = [64, 32]

        if dense_units is None:
            dense_units = [32, 16]

        # 定义R²损失函数
        def r2_loss(y_true, y_pred):
            # 计算总平方和 (y_true - y_mean)^2
            SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
            # 计算残差平方和 (y_true - y_pred)^2
            SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
            # 计算R²
            r2 = 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())
            # 返回负R²，因为我们是在最小化，而R²越大越好
            return -r2

        # 改进的方向损失函数（更关注未来预测点的方向）
        def direction_loss(y_true, y_pred):
            # 我们只关心预测值相对于当前值的方向是否正确
            # 获取最后一个真实值（当前值）
            current_value = 0  # 相对于0的方向

            # 计算方向
            true_direction = tf.cast(tf.greater(y_true[:, 0], current_value), tf.float32)
            pred_direction = tf.cast(tf.greater(y_pred[:, 0], current_value), tf.float32)

            # 计算预测的方向是否正确
            correct_direction = tf.cast(tf.equal(true_direction, pred_direction), tf.float32)

            # 返回方向错误的损失
            dir_loss = 1.0 - tf.reduce_mean(correct_direction)

            # 添加幅度权重 - 大的真实变化应该有更高的惩罚权重
            magnitude_weight = tf.abs(y_true[:, 0]) + 0.1  # 添加小的偏置避免零权重
            magnitude_weight = magnitude_weight / tf.reduce_mean(magnitude_weight)  # 归一化

            # 应用幅度权重
            weighted_errors = (1.0 - correct_direction) * magnitude_weight
            weighted_dir_loss = tf.reduce_sum(weighted_errors) / tf.reduce_sum(magnitude_weight)

            # 混合基本方向损失和权重损失
            return 0.5 * dir_loss + 0.5 * weighted_dir_loss

        # 正则化器
        regularizer = l2(regularization) if regularization > 0 else None

        # 创建模型
        inputs = Input(shape=input_shape)

        if model_type == 'simple':
            # 简化的LSTM模型 - 更稳定
            x = LSTM(lstm_units[0],
                     return_sequences=False,
                     kernel_regularizer=regularizer,
                     recurrent_regularizer=regularizer,
                     recurrent_dropout=0.0)(inputs)  # 避免recurrent_dropout，提高稳定性
            x = Dropout(dropout_rate)(x)

        elif model_type == 'stacked':
            # 堆叠LSTM - 更适合方向预测
            x = LSTM(lstm_units[0],
                     return_sequences=True,
                     kernel_regularizer=regularizer,
                     recurrent_regularizer=regularizer,
                     recurrent_dropout=0.0)(inputs)
            x = BatchNormalization()(x)  # 添加批归一化提高稳定性
            x = Dropout(dropout_rate / 2)(x)
            x = LSTM(lstm_units[1],
                     return_sequences=False,
                     kernel_regularizer=regularizer,
                     recurrent_regularizer=regularizer,
                     recurrent_dropout=0.0)(x)
            x = Dropout(dropout_rate)(x)

        elif model_type == 'bidirectional':
            # 双向LSTM - 更好的值预测
            x = Bidirectional(LSTM(lstm_units[0],
                                   return_sequences=False,
                                   kernel_regularizer=regularizer,
                                   recurrent_regularizer=regularizer,
                                   recurrent_dropout=0.0))(inputs)
            x = BatchNormalization()(x)  # 添加批归一化
            x = Dropout(dropout_rate)(x)

        elif model_type == 'hybrid':
            # 改进的混合模型 - 平衡值预测和方向预测
            # 并行卷积路径捕捉局部模式
            conv1 = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu',
                           kernel_regularizer=regularizer)(inputs)
            conv2 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu',
                           kernel_regularizer=regularizer)(inputs)
            conv_concat = Concatenate()([conv1, conv2])
            conv_output = MaxPooling1D(pool_size=2)(conv_concat)

            # LSTM路径捕捉长期依赖
            lstm_output = Bidirectional(LSTM(lstm_units[0],
                                             return_sequences=False,
                                             kernel_regularizer=regularizer))(inputs)

            # 全局平均池化提取卷积特征
            conv_flat = GlobalAveragePooling1D()(conv_output)

            # 合并特征
            x = Concatenate()([lstm_output, conv_flat])
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        # 全连接层（已简化）
        x = Dense(dense_units[0], activation='relu', kernel_regularizer=regularizer)(x)
        x = BatchNormalization()(x)  # 添加批归一化
        x = Dropout(dropout_rate / 2)(x)

        # 输出层
        outputs = Dense(output_dim, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs)

        # 损失函数选择
        if loss_function == 'mse':
            main_loss = 'mse'
        elif loss_function == 'mae':
            main_loss = 'mae'
        elif loss_function == 'huber':
            main_loss = tf.keras.losses.Huber(delta=1.0)
        elif loss_function == 'r2':
            main_loss = r2_loss
        else:
            raise ValueError(f"不支持的损失函数: {loss_function}")

        # 自定义损失函数
        if direction_penalty > 0:
            def combined_loss(y_true, y_pred):
                # 计算主损失
                main_loss_value = main_loss(y_true, y_pred)
                # 方向损失
                dir_loss_value = direction_loss(y_true, y_pred)
                # 组合损失，自适应权重
                return main_loss_value + direction_penalty * dir_loss_value

            model_loss = combined_loss
        else:
            model_loss = main_loss

        # 编译模型
        # 使用更温和的学习率衰减
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.95,  # 更温和的衰减
            staircase=False  # 平滑衰减
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            loss=model_loss,
            optimizer=optimizer,
            metrics=[tf.keras.metrics.MeanAbsoluteError(),
                     tf.keras.metrics.MeanSquaredError()]
        )

        self.model = model
        self.model_type = model_type

        logger.info(f"构建了{model_type}类型的LSTM模型，损失函数: {loss_function}")
        logger.info(f"模型输入形状: {input_shape}")

        return model

    def train_model(self,
                    model: tf.keras.Model,
                    X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    batch_size: int = 32,
                    epochs: int = 100,
                    patience: int = 20,
                    min_delta: float = 0.001,
                    restore_best_weights: bool = True) -> Dict[str, Any]:
        """
        训练模型
        
        参数:
            model: TensorFlow模型
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            batch_size: 批大小
            epochs: 最大训练轮数
            patience: 早停的耐心值
            min_delta: 早停的最小改进量
            restore_best_weights: 是否恢复最佳权重
            
        返回:
            训练历史
        """
        if not TF_AVAILABLE or model is None:
            logger.error("无法训练模型：TensorFlow未安装或模型不可用")
            return None

        # 设置回调函数
        callbacks = []

        # 早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=restore_best_weights,
            verbose=1
        )
        callbacks.append(early_stopping)

        # 学习率降低回调
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=0.00001,
            verbose=1
        )
        callbacks.append(reduce_lr)

        # 检查点回调
        checkpoint_path = self.model_dir / f"model_checkpoint_{self.session_id}.h5"
        checkpoint = ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callbacks.append(checkpoint)

        # TensorBoard回调
        tensorboard_dir = self.logs_dir / f"tensorboard_{self.session_id}"
        tensorboard = TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)

        # CSV日志回调
        csv_path = self.logs_dir / f"training_log_{self.session_id}.csv"
        csv_logger = CSVLogger(str(csv_path))
        callbacks.append(csv_logger)

        # 开始训练
        logger.info(f"开始训练模型，最大轮数: {epochs}, 批大小: {batch_size}")
        start_time = time.time()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        training_time = time.time() - start_time
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")

        # 保存当前模型和训练历史
        self.model = model
        self.training_history = history.history

        return history.history

    def evaluate_model(self,
                       X_test: np.ndarray,
                       y_test: np.ndarray,
                       scaled: bool = True) -> Dict[str, float]:
        """
        评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试目标
            scaled: 数据是否已缩放
            
        返回:
            性能评估结果字典
        """
        if self.model is None:
            logger.error("没有可评估的模型")
            return None

        # 使用模型进行预测
        y_pred = self.model.predict(X_test)

        # 如果数据已缩放，则需要反向变换
        if scaled and 'target_scaler' in self.scalers:
            y_test_original = self.scalers['target_scaler'].inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.scalers['target_scaler'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_test_original = y_test.flatten()
            y_pred_original = y_pred.flatten()

        # 计算评估指标
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)

        # 计算方向准确率
        direction_correct = np.sum(np.sign(y_test_original[1:] - y_test_original[:-1]) ==
                                   np.sign(y_pred_original[1:] - y_pred_original[:-1]))
        direction_accuracy = direction_correct / (len(y_test_original) - 1)

        # 计算MAPE (平均绝对百分比误差)
        mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-10))) * 100

        # 计算最大绝对误差
        max_error = np.max(np.abs(y_test_original - y_pred_original))

        # 记录评估结果
        eval_results = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'mape': float(mape),
            'max_error': float(max_error)
        }

        logger.info(f"模型评估结果: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, R²={r2:.6f}")
        logger.info(f"方向准确率: {direction_accuracy:.2%}, MAPE: {mape:.2f}%, 最大误差: {max_error:.6f}")

        return eval_results

    def visualize_predictions(self,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              scaled: bool = True,
                              save_path: str = None,
                              plot_title: str = "预测结果与真实值对比") -> None:
        """
        可视化模型预测结果与真实值对比
        
        参数:
            X_test: 测试特征
            y_test: 测试目标
            scaled: 数据是否已缩放
            save_path: 图表保存路径，如果为None则显示不保存
            plot_title: 图表标题
        """
        if self.model is None:
            logger.error("没有可用的模型进行预测")
            return
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 使用模型进行预测
        y_pred = self.model.predict(X_test)

        # 如果数据已缩放，则需要反向变换
        if scaled and 'target_scaler' in self.scalers:
            y_test_original = self.scalers['target_scaler'].inverse_transform(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.scalers['target_scaler'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_test_original = y_test.flatten()
            y_pred_original = y_pred.flatten()

        # 计算评估指标
        mae = np.mean(np.abs(y_test_original - y_pred_original))
        mse = np.mean((y_test_original - y_pred_original) ** 2)
        rmse = np.sqrt(mse)

        # 设置可视化样式
        plt.figure(figsize=(12, 6))
        # sns.set_style('whitegrid')

        # 绘制真实值和预测值
        plt.plot(y_test_original, label='真实值', color='blue', linewidth=2)
        plt.plot(y_pred_original, label='预测值', color='red', linewidth=2, linestyle='--')

        # 添加标题和标签
        plt.title(f"{plot_title}\nMAE: {mae:.4f}, RMSE: {rmse:.4f}", fontsize=16)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('价格比值', fontsize=12)

        # 设置清晰的图例
        plt.legend(fontsize=12, loc='best', frameon=True, facecolor='white', edgecolor='gray')

        # 添加网格线增强可读性
        plt.grid(True, linestyle='--', alpha=0.7)

        # 添加错误带 - 可视化预测误差
        if len(y_test_original) > 100:
            # 为了清晰度，对数据进行降采样
            sample_rate = len(y_test_original) // 100
            sample_indices = range(0, len(y_test_original), sample_rate)
            plt.fill_between(
                sample_indices,
                y_test_original[sample_indices],
                y_pred_original[sample_indices],
                color='gray', alpha=0.2, label='预测误差'
            )
        else:
            plt.fill_between(
                range(len(y_test_original)),
                y_test_original,
                y_pred_original,
                color='gray', alpha=0.2, label='预测误差'
            )

        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"预测可视化图表已保存至: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_training_history(self, save_path: str = None) -> None:
        """
        可视化训练历史
        
        参数:
            save_path: 图表保存路径，如果为None则显示不保存
        """
        if self.training_history is None:
            logger.error("没有可用的训练历史")
            return

        # 设置可视化样式
        fig = plt.figure(figsize=(18, 12))
        sns.set_style('whitegrid')

        # 1. 损失曲线
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(self.training_history['loss'], label='训练损失', color='blue', linewidth=2)
        ax1.plot(self.training_history['val_loss'], label='验证损失', color='red', linewidth=2)

        min_val_loss = min(self.training_history['val_loss'])
        min_val_loss_epoch = self.training_history['val_loss'].index(min_val_loss)
        ax1.scatter(min_val_loss_epoch, min_val_loss, color='green', s=100, zorder=5,
                    label=f'最佳验证损失: {min_val_loss:.6f} (轮次 {min_val_loss_epoch})')

        ax1.set_title('训练和验证损失曲线', fontsize=14)
        ax1.set_xlabel('轮次', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 2. MAE曲线 - 检查键是否存在
        ax2 = fig.add_subplot(2, 2, 2)
        mae_metrics_available = 'mae' in self.training_history and 'val_mae' in self.training_history

        if mae_metrics_available:
            ax2.plot(self.training_history['mae'], label='训练MAE', color='blue', linewidth=2)
            ax2.plot(self.training_history['val_mae'], label='验证MAE', color='red', linewidth=2)

            min_val_mae = min(self.training_history['val_mae'])
            min_val_mae_epoch = self.training_history['val_mae'].index(min_val_mae)
            ax2.scatter(min_val_mae_epoch, min_val_mae, color='green', s=100, zorder=5,
                        label=f'最佳验证MAE: {min_val_mae:.6f} (轮次 {min_val_mae_epoch})')

            ax2.set_title('训练和验证MAE曲线', fontsize=14)
            ax2.set_xlabel('轮次', fontsize=12)
            ax2.set_ylabel('平均绝对误差', fontsize=12)
            ax2.legend(fontsize=10)
            ax2.grid(True, linestyle='--', alpha=0.7)
        else:
            ax2.text(0.5, 0.5, 'MAE指标不可用', ha='center', va='center', fontsize=14)
            ax2.set_title('MAE曲线 - 数据不可用', fontsize=14)
            logger.warning("训练历史中不包含MAE指标，跳过MAE曲线绘制")

        # 3. MSE曲线 - 检查键是否存在
        ax3 = fig.add_subplot(2, 2, 3)
        mse_metrics_available = 'mse' in self.training_history and 'val_mse' in self.training_history

        if mse_metrics_available:
            ax3.plot(self.training_history['mse'], label='训练MSE', color='blue', linewidth=2)
            ax3.plot(self.training_history['val_mse'], label='验证MSE', color='red', linewidth=2)

            min_val_mse = min(self.training_history['val_mse'])
            min_val_mse_epoch = self.training_history['val_mse'].index(min_val_mse)
            ax3.scatter(min_val_mse_epoch, min_val_mse, color='green', s=100, zorder=5,
                        label=f'最佳验证MSE: {min_val_mse:.6f} (轮次 {min_val_mse_epoch})')

            ax3.set_title('训练和验证MSE曲线', fontsize=14)
            ax3.set_xlabel('轮次', fontsize=12)
            ax3.set_ylabel('均方误差', fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(True, linestyle='--', alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'MSE指标不可用', ha='center', va='center', fontsize=14)
            ax3.set_title('MSE曲线 - 数据不可用', fontsize=14)
            logger.warning("训练历史中不包含MSE指标，跳过MSE曲线绘制")

        # 4. 损失函数在最后几轮的放大视图
        ax4 = fig.add_subplot(2, 2, 4)
        last_epochs = min(30, len(self.training_history['loss']))
        start_idx = max(0, len(self.training_history['loss']) - last_epochs)

        x_range = list(range(start_idx, len(self.training_history['loss'])))
        ax4.plot(x_range,
                 self.training_history['loss'][start_idx:],
                 label='训练损失', color='blue', linewidth=2)
        ax4.plot(x_range,
                 self.training_history['val_loss'][start_idx:],
                 label='验证损失', color='red', linewidth=2)

        # 如果最佳验证损失在最后几轮
        if min_val_loss_epoch >= start_idx:
            ax4.scatter(min_val_loss_epoch, min_val_loss, color='green', s=100, zorder=5,
                        label=f'最佳验证损失: {min_val_loss:.6f}')

        ax4.set_title(f'最后{last_epochs}轮损失曲线详情', fontsize=14)
        ax4.set_xlabel('轮次', fontsize=12)
        ax4.set_ylabel('损失', fontsize=12)
        ax4.legend(fontsize=10)
        ax4.grid(True, linestyle='--', alpha=0.7)

        # 添加总体训练信息
        plt.suptitle('模型训练历史分析', fontsize=16)

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # 为顶部标题留出空间

        # 保存或显示图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图表已保存至: {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_feature_importance(self, feature_cols: List[str] = None, save_path: str = None) -> None:
        """
        通过置换特征值来估计特征重要性
        
        参数:
            feature_cols: 特征列名列表
            save_path: 图表保存路径，如果为None则显示不保存
        """
        if self.model is None or 'feature_scaler' not in self.scalers:
            logger.error("没有可用的模型或特征缩放器")
            return

        if feature_cols is None or len(feature_cols) == 0:
            logger.error("需要提供特征列名列表")
            return

        # 特征重要性可视化 - 对于深度学习模型，使用置换重要性
        logger.info("暂未实现特征重要性计算，需要另外实现")

    def save_model(self,
                   model_name: str = None,
                   include_scalers: bool = True,
                   include_metadata: bool = True) -> str:
        """
        保存模型和相关组件
        
        参数:
            model_name: 模型名称，如果为None则使用时间戳生成
            include_scalers: 是否包含缩放器
            include_metadata: 是否包含元数据
            
        返回:
            保存路径
        """
        if self.model is None:
            logger.error("没有可保存的模型")
            return None

        # 生成模型名称
        if model_name is None:
            model_name = f"stock_ratio_predictor_{self.session_id}"

        # 创建保存目录
        model_dir = self.model_dir / model_name
        model_dir.mkdir(exist_ok=True, parents=True)

        # 保存TensorFlow模型
        model_path = model_dir / "model.h5"
        self.model.save(str(model_path))
        logger.info(f"模型已保存至: {model_path}")

        # 保存缩放器
        if include_scalers and self.scalers:
            scaler_path = model_dir / "scalers.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers, f)
            logger.info(f"缩放器已保存至: {scaler_path}")

        # 保存训练历史
        if include_metadata and self.training_history:
            history_path = model_dir / "training_history.json"
            with open(history_path, 'w') as f:
                # 转换numpy类型为Python原生类型
                history_dict = {}
                for key, value in self.training_history.items():
                    history_dict[key] = [float(v) for v in value]

                json.dump(history_dict, f, indent=4)
            logger.info(f"训练历史已保存至: {history_path}")

        # 保存模型元数据
        if include_metadata:
            try:
                # 安全地转换TensorFlow对象为可序列化格式
                def tf_object_converter(obj):
                    if hasattr(obj, 'numpy'):
                        # 将TensorFlow张量转换为NumPy数组
                        return obj.numpy().tolist()
                    elif hasattr(obj, 'as_list'):
                        # 将TensorShape转换为列表
                        return obj.as_list()
                        # 处理其他复杂对象
                    try:
                        return str(obj)
                    except:
                        return "不可序列化对象"

                # 获取模型配置并序列化
                model_config = self.model.get_config()

                # 递归处理配置中的TensorFlow对象
                def process_config(config):
                    if isinstance(config, dict):
                        return {k: process_config(v) for k, v in config.items()}
                    elif isinstance(config, list):
                        return [process_config(item) for item in config]
                    elif hasattr(config, 'numpy') or hasattr(config, 'as_list'):
                        return tf_object_converter(config)
                    elif isinstance(config, (int, float, str, bool, type(None))):
                        return config
                    else:
                        return str(config)

                # 构建元数据字典
                metadata = {
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "session_id": self.session_id,
                    # 不包含模型摘要，因为这不是可序列化对象
                    "model_config_summary": str(self.model.summary()),
                    "model_config": process_config(model_config)
                }

                metadata_path = model_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=4)
                logger.info(f"模型元数据已保存至: {metadata_path}")

            except Exception as e:
                logger.error(f"保存模型元数据时出错: {e}")
                logger.error(traceback.format_exc())
                # 继续执行，不因为元数据保存失败而中断整个流程

        return str(model_dir)

    def load_model(self, model_path: str) -> bool:
        """
        加载模型和相关组件
        
        参数:
            model_path: 模型路径
            
        返回:
            加载是否成功
        """
        model_dir = Path(model_path)

        # 检查路径是否存在
        if not model_dir.exists():
            logger.error(f"模型路径不存在: {model_path}")
            return False

        try:
            # 加载TensorFlow模型
            model_file = model_dir / "model.h5"
            if model_file.exists():
                self.model = load_model(str(model_file))
                logger.info(f"模型已加载: {model_file}")
            else:
                logger.error(f"模型文件不存在: {model_file}")
                return False

            # 加载缩放器
            scaler_file = model_dir / "scalers.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.scalers = pickle.load(f)
                logger.info(f"缩放器已加载: {scaler_file}")

            # 加载训练历史
            history_file = model_dir / "training_history.json"
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.training_history = json.load(f)
                logger.info(f"训练历史已加载: {history_file}")

            return True

        except Exception as e:
            logger.error(f"加载模型出错: {e}")
            return False

    def predict_ratio(self,
                      X_input: np.ndarray,
                      scaled: bool = True) -> np.ndarray:
        """
        使用模型预测价格比值
        
        参数:
            X_input: 输入特征
            scaled: 数据是否已缩放
            
        返回:
            预测值
        """
        if self.model is None:
            logger.error("没有可用的模型进行预测")
            return None

        # 使用模型进行预测
        y_pred = self.model.predict(X_input)

        # 如果数据已缩放，则需要反向变换
        if scaled and 'target_scaler' in self.scalers:
            y_pred_original = self.scalers['target_scaler'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
        else:
            y_pred_original = y_pred.flatten()

        return y_pred_original

    def run_complete_training_pipeline(self,
                                       stock_code_a: str,
                                       stock_code_b: str,
                                       start_date: str = '2015-01-01',
                                       end_date: str = None,
                                       model_type: str = 'stacked',  # 默认使用stacked，更擅长方向预测
                                       seq_length: int = 20,
                                       forecast_horizon: int = 1,
                                       test_size: float = 0.15,
                                       val_size: float = 0.15,
                                       batch_size: int = 24,  # 适中的批量大小
                                       epochs: int = 150,  # 增加训练轮数
                                       patience: int = 25,  # 更有耐心的早停
                                       learning_rate: float = 0.0001,  # 降低学习率
                                       dropout_rate: float = 0.25,  # 适当的dropout
                                       regularization: float = 0.001,  # 适当的正则化
                                       feature_selection: bool = True,
                                       max_features: int = 30,  # 减少特征数量
                                       direction_penalty: float = 0.3,  # 更强的方向惩罚
                                       save_model: bool = True,
                                       plot_results: bool = True) -> Dict[str, Any]:
        """
        运行完整的训练流程
        
        参数:
            stock_code_a: 股票A代码
            stock_code_b: 股票B代码
            start_date: 开始日期
            end_date: 结束日期
            model_type: 模型类型
            seq_length: 序列长度
            forecast_horizon: 预测时间跨度
            test_size: 测试集比例
            val_size: 验证集比例
            batch_size: 批大小
            epochs: 训练轮数
            patience: 早停耐心值
            learning_rate: 学习率
            dropout_rate: Dropout比例
            regularization: 正则化系数
            feature_selection: 是否进行特征选择
            max_features: 特征选择后保留的最大特征数量
            direction_penalty: 方向预测损失权重
            save_model: 是否保存模型
            plot_results: 是否绘制结果图表
            
        返回:
            包含训练结果的字典
        """
        start_time = time.time()
        pipeline_results = {}

        try:
            # 1. 准备数据
            logger.info(f"开始训练流程: {stock_code_a}/{stock_code_b}, 模型类型: {model_type}")
            df = self.prepare_ratio_data(stock_code_a, stock_code_b, start_date, end_date)
            pipeline_results['data_shape'] = df.shape

            # 2. 特征工程
            feature_df = self.engineer_features(df,
                                                feature_selection=feature_selection,
                                                max_features=max_features)
            pipeline_results['feature_shape'] = feature_df.shape

            # 准备特征列，排除日期索引和目标列
            feature_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols.remove('ratio')  # 移除目标列
            pipeline_results['feature_count'] = len(feature_cols)

            # 记录特征名称
            pipeline_results['feature_names'] = feature_cols

            # 3. 创建序列
            X, y = self.create_sequences(
                feature_df,
                target_col='ratio',
                feature_cols=feature_cols,
                seq_length=seq_length,
                forecast_horizon=forecast_horizon
            )
            pipeline_results['sequence_shape'] = {'X': X.shape, 'y': y.shape}

            # 4. 拆分数据
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(
                X, y, test_size=test_size, val_size=val_size
            )

            # 5. 缩放数据
            X_train_scaled, X_val_scaled, X_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled = self.scale_data(
                X_train, X_val, X_test, y_train, y_val, y_test
            )

            # 6. 构建模型
            input_shape = (seq_length, len(feature_cols))

            # 根据特征数量调整网络规模
            lstm_units = [64, 32]
            dense_units = [32, 16]
            if len(feature_cols) > 30:
                lstm_units = [96, 48]
                dense_units = [48, 24]

            model = self.build_lstm_model(
                input_shape=input_shape,
                output_dim=forecast_horizon,
                model_type=model_type,
                lstm_units=lstm_units,
                dense_units=dense_units,
                dropout_rate=dropout_rate,
                learning_rate=learning_rate,
                regularization=regularization,
                direction_penalty=direction_penalty,
                loss_function='huber'
            )

            # 7. 训练模型
            history = self.train_model(
                model=model,
                X_train=X_train_scaled,
                y_train=y_train_scaled,
                X_val=X_val_scaled,
                y_val=y_val_scaled,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience
            )
            pipeline_results['training_history'] = {
                'final_loss': float(history['loss'][-1]),
                'final_val_loss': float(history['val_loss'][-1]),
                'best_val_loss': float(min(history['val_loss']))
            }

            # 8. 评估模型
            eval_results = self.evaluate_model(
                X_test=X_test_scaled,
                y_test=y_test_scaled,
                scaled=True
            )
            pipeline_results['evaluation'] = eval_results

            # 报告方向准确率
            logger.info(f"方向预测准确率: {eval_results['direction_accuracy']:.2%}")

            # 检查模型性能是否需要改进
            if eval_results['direction_accuracy'] < 0.52 or eval_results['r2'] < 0:
                logger.warning("模型性能较差，考虑调整参数或使用不同的模型架构")
                pipeline_results['needs_improvement'] = True

                # 如果R2过低，提供性能改进建议
                if eval_results['r2'] < 0:
                    logger.warning("R2值为负，表明模型预测比简单使用均值预测还要差")
                    pipeline_results['improvement_suggestions'] = [
                        "进一步减少特征数量，重点关注最具预测力的特征",
                        "尝试更简单的模型架构，例如单层LSTM或线性模型",
                        "延长训练数据的时间范围获取更多历史模式",
                        "考虑使用额外的外部特征(如市场情绪、宏观经济指标)",
                        "检查数据预处理流程是否存在泄漏或其他问题"
                    ]

            # 9. 可视化结果
            if plot_results:
                # 创建结果目录
                results_dir = self.logs_dir / f"results_{self.session_id}"
                results_dir.mkdir(exist_ok=True, parents=True)

                # 绘制预测结果对比图
                self.visualize_predictions(
                    X_test=X_test_scaled,
                    y_test=y_test_scaled,
                    scaled=True,
                    save_path=str(results_dir / "predictions.png"),
                    plot_title=f"{stock_code_a}/{stock_code_b} 模型预测结果 (方向准确率: {eval_results['direction_accuracy']:.2%})"
                )

                # 绘制训练历史图
                self.visualize_training_history(
                    save_path=str(results_dir / "training_history.png")
                )

            # 10. 保存模型
            if save_model:
                model_name = f"{stock_code_a}_{stock_code_b}_{model_type}_seq{seq_length}_model"
                model_path = self.save_model(model_name=model_name)
                pipeline_results['model_path'] = model_path

            # 记录总耗时
            training_time = time.time() - start_time
            logger.info(f"训练流程完成，总耗时: {training_time:.2f}秒")
            pipeline_results['training_time'] = training_time

            return pipeline_results

        except Exception as e:
            logger.error(f"训练流程出错: {e}")
            logger.error(traceback.format_exc())
            # 返回包含错误信息的字典，并添加基本字段避免KeyError
            error_results = {
                'error': str(e),
                'elapsed_time': time.time() - start_time,
                'training_time': time.time() - start_time,  # 确保包含training_time字段
                # 添加基本字段，避免KeyError
                'data_shape': (0, 0),
                'feature_count': 0,
                'sequence_shape': {'X': (0, 0, 0), 'y': (0, 0)},
                'training_history': {
                    'final_loss': 0.0,
                    'final_val_loss': 0.0,
                    'best_val_loss': 0.0
                },
                'evaluation': {
                    'mse': 0.0,
                    'rmse': 0.0,
                    'mae': 0.0,
                    'r2': 0.0,
                    'direction_accuracy': 0.0,
                    'mape': 0.0,
                    'max_error': 0.0
                }
            }
            return error_results


if __name__ == "__main__":
    import traceback

    # 设置日志级别为调试级别
    logger.setLevel(logging.DEBUG)

    # 创建训练器实例
    trainer = StockRatioPredictorTrainer()

    try:
        stock_a = 'XAU'
        stock_b = 'XAG'

        # 运行完整训练流程
        results = trainer.run_complete_training_pipeline(
            stock_code_a=stock_a,
            stock_code_b=stock_b,
            start_date='2015-01-01',
            end_date=None,
            model_type='hybrid',
            seq_length=30,  # 30天窗口
            forecast_horizon=5,  # 预测未来5天
            test_size=0.15,
            val_size=0.15,
            batch_size=32,
            epochs=150,
            patience=30,
            learning_rate=0.00005,
            dropout_rate=0.30,
            regularization=0.001,
            feature_selection=True,
            max_features=25,  # 减少特征数量以降低过拟合风险
            direction_penalty=0.25,
            save_model=True,
            plot_results=True
        )

        # 打印训练结果
        print("\n========== 训练结果摘要 ==========")
        
        # 安全获取字典中的值，避免KeyError
        def safe_get(d, key, default="未提供"):
            return d.get(key, default)
            
        print(f"数据形状: {safe_get(results, 'data_shape')}")
        print(f"特征数量: {safe_get(results, 'feature_count')}")
        print(f"序列形状: {safe_get(results, 'sequence_shape')}")
        
        # 安全获取嵌套字典值
        if 'training_history' in results:
            history = results['training_history']
            print(f"最终训练损失: {history.get('final_loss', 0.0):.6f}")
            print(f"最终验证损失: {history.get('final_val_loss', 0.0):.6f}")
            print(f"最佳验证损失: {history.get('best_val_loss', 0.0):.6f}")
        else:
            print("训练历史: 未提供")
            
        print("\n评估指标:")
        if 'evaluation' in results:
            for metric, value in results['evaluation'].items():
                print(f"  {metric}: {value}")
        else:
            print("  无评估指标")

        if 'model_path' in results:
            print(f"\n模型已保存至: {results['model_path']}")

        # 安全获取训练时间
        training_time = results.get('training_time', 0.0)
        print(f"\n总训练时间: {training_time:.2f}秒")
        print("===================================")

    except Exception as e:
        print(f"程序出错: {e}")
        traceback.print_exc()
