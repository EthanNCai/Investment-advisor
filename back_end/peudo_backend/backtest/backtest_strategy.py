import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import math
import json

# 从现有模块导入函数
from k_chart_fetcher import get_stock_data_pair, date_alignment, durations
from indicators.investment_signals import analyze_current_position, generate_investment_signals
from indicators.technical_indicators import calculate_price_ratio_anomaly
from get_stock_data.stock_trends_base import StockTrendsDatabase

# 在类定义外添加辅助函数
def convert_numpy_types(obj):
    """将NumPy数据类型转换为标准Python类型，使其可JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class BacktestEngine:
    def __init__(self):
        """初始化回测引擎"""
        pass

    def run_price_ratio_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行价差交易回测 - 基于异常点检测

        参数:
            params: 回测参数字典，包含：
                - start_date: 回测开始日期
                - end_date: 回测结束日期
                - code_a: 资产A代码
                - code_b: 资产B代码
                - initial_capital: 初始资金
                - position_size_type: 仓位计算方式 ('fixed'/'percent'/'kelly')
                - position_size: 仓位大小
                - stop_loss: 止损比例(%)
                - take_profit: 止盈比例(%)
                - max_positions: 最大持仓数量
                - trading_fee: 交易费率
                - trailing_stop: 追踪止损(%) 可选
                - time_stop: 最长持仓时间(天) 可选
                - hedge_mode: 对冲模式 ('single'/'pair') 单边或双边交易
                - polynomial_degree: 拟合曲线的多项式次数
                - anomaly_threshold: 异常检测阈值

        返回:
            回测结果字典
        """
        try:
            # 获取基本参数
            start_date = datetime.strptime(params.get('start_date'), '%Y-%m-%d')
            end_date = datetime.strptime(params.get('end_date'), '%Y-%m-%d')
            code_a = params.get('code_a')
            code_b = params.get('code_b')
            initial_capital = float(params.get('initial_capital', 100000))
            position_size_type = params.get('position_size_type', 'percent')
            position_size = float(params.get('position_size', 10))
            stop_loss = float(params.get('stop_loss', 5)) / 100  # 转换为小数
            take_profit = float(params.get('take_profit', 10)) / 100  # 转换为小数
            max_positions = int(params.get('max_positions', 5))
            trading_fee = float(params.get('trading_fee', 0.0003))

            # 新增参数
            trailing_stop = float(params.get('trailing_stop', 0)) / 100  # 追踪止损百分比
            time_stop = int(params.get('time_stop', 0))  # 最长持仓时间（天）
            hedge_mode = params.get('hedge_mode', 'single')  # 对冲模式

            # 异常检测相关参数
            polynomial_degree = int(params.get('polynomial_degree', 3))  # 多项式拟合次数
            anomaly_threshold = float(params.get('anomaly_threshold', 2.0))  # 异常检测阈值

            # 新增出场策略参数
            mean_reversion_exit = params.get('mean_reversion_exit', True)  # 回归均值出场
            mean_reversion_threshold = float(params.get('mean_reversion_threshold', 0.5))  # 回归均值阈值
            reverse_anomaly_exit = params.get('reverse_anomaly_exit', True)  # 反向异常点出场

            # 获取历史K线数据
            close_a_, close_b_, dates_a_, dates_b_ = get_stock_data_pair(code_a, code_b)
            close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)

            # 筛选时间范围内的数据
            dates_dt = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
            mask = [(d >= start_date) and (d <= end_date) for d in dates_dt]

            dates_filtered = [dates[i] for i in range(len(dates)) if mask[i]]
            close_a_filtered = [close_a[i] for i in range(len(close_a)) if mask[i]]
            close_b_filtered = [close_b[i] for i in range(len(close_b)) if mask[i]]
            if len(dates_filtered) < 20:
                return {"error": f"时间范围内数据不足，无法进行回测。至少需要20个交易日的数据。"}

            # 创建DataFrame
            df = pd.DataFrame({
                'date': dates_filtered,
                'close_a': close_a_filtered,
                'close_b': close_b_filtered
            })

            # 添加日期列的datetime版本便于计算时间差
            df['date_dt'] = pd.to_datetime(df['date'])

            # 计算价格比值
            df['ratio'] = df['close_a'] / df['close_b']

            # 计算每日回报率，用于后续性能指标计算
            df['return_a'] = df['close_a'].pct_change()
            df['return_b'] = df['close_b'].pct_change()
            df['ratio_return'] = df['ratio'].pct_change()

            # 生成比值拟合曲线
            x = np.arange(len(df))
            y = df['ratio'].values
            # 使用多项式拟合
            poly_model = np.polyfit(x, y, polynomial_degree)
            poly_func = np.poly1d(poly_model)
            # 计算拟合值
            df['fitting_line'] = poly_func(x)
            # 计算与拟合线的差值
            df['delta'] = df['ratio'] - df['fitting_line']
            # 计算整个时间范围的标准差
            delta_array = np.array(df['delta'].values)
            std_value = np.std(delta_array)

            # 保留原有的移动平均线，可用于其他分析
            df['ratio_ma'] = df['ratio'].rolling(window=60).mean().fillna(df['ratio'])

            # 将标准差值存储在DataFrame中方便后续使用
            df['ratio_std'] = std_value  # 使用相同的标准差值填充整列

            # 进行异常检测
            ratio_data = df['ratio'].tolist()
            delta_data = df['delta'].tolist()
            # 使用计算出的全局标准差

            # 执行异常检测
            anomaly_result = calculate_price_ratio_anomaly(ratio_data, delta_data, anomaly_threshold, std_value)

            # 创建异常点标记列，并初始化为0
            df['anomaly_signal'] = 0

            # 将异常点信息整合到DataFrame中
            for anomaly in anomaly_result['anomalies']:
                idx = anomaly['index']
                if idx < len(df):
                    # 根据异常点相对于拟合线的位置确定方向
                    # 如果比值高于拟合线，做空资产A做多资产B
                    # 如果比值低于拟合线，做多资产A做空资产B
                    if df['delta'].iloc[idx] > 0:
                        df.loc[df.index[idx], 'anomaly_signal'] = 1  # 做空A做多B
                    else:
                        df.loc[df.index[idx], 'anomaly_signal'] = -1  # 做多A做空B

                    # 存储异常点的z_score和deviation，用于信号强度判断
                    df.loc[df.index[idx], 'z_score'] = anomaly['z_score']
                    df.loc[df.index[idx], 'deviation'] = anomaly['deviation']

            # 去除NaN值
            df = df.dropna().reset_index(drop=True)

            if len(df) < 10:
                return {"error": f"处理后的有效数据不足，无法进行回测。至少需要10个交易日的数据。"}

            # 初始化回测结果
            equity_curve = []  # 存储每个交易日的资金曲线
            trades = []  # 每笔交易记录
            current_positions = []  # 当前持仓
            cash = initial_capital
            equity = initial_capital
            trade_id = 1
            max_equity = initial_capital

            # 跟踪交易统计
            profitable_trades = 0
            losing_trades = 0
            total_profit = 0
            total_loss = 0

            # 为了计算夏普比率等指标
            daily_returns = []

            # 跟踪最大持仓时间统计
            position_days = []

            # 设置保护性参数，以防万一
            max_drawdown = 0
            max_drawdown_duration = 0
            drawdown_start = None
            last_equity_peak = initial_capital

            # 开始回测
            for i in range(len(df)):
                current_date = df['date'].iloc[i]
                current_date_dt = df['date_dt'].iloc[i]
                current_anomaly_signal = df['anomaly_signal'].iloc[i]
                current_price_a = df['close_a'].iloc[i]
                current_price_b = df['close_b'].iloc[i]
                current_ratio = df['ratio'].iloc[i]
                current_fitting = df['fitting_line'].iloc[i] if 'fitting_line' in df.columns else current_ratio
                current_delta = df['delta'].iloc[i] if 'delta' in df.columns else 0

                # 更新现有持仓信息
                positions_to_close = []

                # 更新当前持仓的价值
                for j, pos in enumerate(current_positions):
                    if pos['asset'] == code_a:
                        pos['current_price'] = current_price_a
                        if pos['direction'] == 'long':
                            pos['pnl'] = (current_price_a - pos['entry_price']) / pos['entry_price'] * pos[
                                'position_size']
                            pos['pnl_percent'] = (current_price_a - pos['entry_price']) / pos['entry_price']
                        else:  # short
                            pos['pnl'] = (pos['entry_price'] - current_price_a) / pos['entry_price'] * pos[
                                'position_size']
                            pos['pnl_percent'] = (pos['entry_price'] - current_price_a) / pos['entry_price']
                    else:  # code_b
                        pos['current_price'] = current_price_b
                        if pos['direction'] == 'long':
                            pos['pnl'] = (current_price_b - pos['entry_price']) / pos['entry_price'] * pos[
                                'position_size']
                            pos['pnl_percent'] = (current_price_b - pos['entry_price']) / pos['entry_price']
                        else:  # short
                            pos['pnl'] = (pos['entry_price'] - current_price_b) / pos['entry_price'] * pos[
                                'position_size']
                            pos['pnl_percent'] = (pos['entry_price'] - current_price_b) / pos['entry_price']

                    # 更新最高盈利点，用于追踪止损
                    if trailing_stop > 0:
                        if pos['pnl_percent'] > pos.get('highest_profit_percent', 0):
                            pos['highest_profit_percent'] = pos['pnl_percent']

                    # 计算持仓天数
                    pos_entry_date = datetime.strptime(pos['entry_date'], '%Y-%m-%d')
                    pos['holding_days'] = (current_date_dt - pos_entry_date).days

                    # 更新与拟合线的关系
                    pos['current_delta'] = current_delta

                # 检查平仓条件
                for j, pos in enumerate(current_positions):
                    close_reason = None

                    # 1. 止损检查
                    if pos['pnl_percent'] <= -stop_loss:
                        close_reason = 'stop_loss'

                    # 2. 止盈检查
                    elif pos['pnl_percent'] >= take_profit:
                        close_reason = 'take_profit'

                    # 3. 追踪止损检查
                    elif trailing_stop > 0 and 'highest_profit_percent' in pos:
                        trailing_stop_level = pos['highest_profit_percent'] - trailing_stop
                        if trailing_stop_level > 0 and pos['pnl_percent'] < trailing_stop_level:
                            close_reason = 'trailing_stop'

                    # 4. 时间止损检查
                    elif 0 < time_stop <= pos['holding_days']:
                        close_reason = 'time_stop'

                    # 5. 回归均值出场策略 - 当价格回归到拟合线附近
                    elif mean_reversion_exit and abs(current_delta) <= mean_reversion_threshold * std_value:
                        # 检查持仓方向是否与初始异常点信号一致
                        if (pos['entry_signal'] > 0 > current_delta) or (
                                pos['entry_signal'] < 0 < current_delta):
                            close_reason = 'mean_reversion'

                    # 6. 反向异常点出场 - 当出现反向的异常信号
                    elif reverse_anomaly_exit and current_anomaly_signal != 0:
                        # 检查当前异常信号是否与持仓方向相反
                        if (pos['entry_signal'] > 0 > current_anomaly_signal) or (
                                pos['entry_signal'] < 0 < current_anomaly_signal):
                            close_reason = 'reverse_anomaly'

                    # 如果满足任何平仓条件，将该仓位添加到平仓列表
                    if close_reason:
                        positions_to_close.append((j, close_reason))

                # 从后往前平仓，以避免索引问题
                for j, reason in reversed(positions_to_close):
                    pos = current_positions[j]

                    # 计算交易费用
                    fee = pos['position_size'] * trading_fee

                    # 更新资金
                    cash += pos['position_size'] + pos['pnl'] - fee

                    # 记录持仓时间
                    position_days.append(pos['holding_days'])

                    # 记录盈亏情况
                    if pos['pnl'] > 0:
                        profitable_trades += 1
                        total_profit += pos['pnl']
                    else:
                        losing_trades += 1
                        total_loss += abs(pos['pnl'])

                    # 记录交易
                    trades.append({
                        'id': trade_id,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'holding_days': pos['holding_days'],
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['current_price'],
                        'position_type': f"{pos['direction']}_{pos['asset']}",
                        'trade_direction': 'long_A_short_B' if pos['entry_signal_type'] == 'low_ratio' else 'short_A_long_B',
                        'position_size': pos['position_size'],
                        'entry_signal_type': pos['entry_signal_type'],
                        'entry_signal_strength': pos['entry_signal_strength'],
                        'pnl': pos['pnl'] - fee,
                        'pnl_percent': pos['pnl_percent'] - (fee / pos['position_size']),
                        'status': 'closed',
                        'exit_reason': reason
                    })
                    trade_id += 1

                    # 移除平仓的头寸
                    current_positions.pop(j)

                # 检查开仓信号 - 使用异常点作为入场信号
                if len(current_positions) < max_positions and current_anomaly_signal != 0:
                    # 计算每个头寸的大小
                    if position_size_type == 'fixed':
                        position_amount = min(position_size, cash)
                    elif position_size_type == 'kelly':
                        # 基本Kelly公式
                        win_rate = profitable_trades / max(1, profitable_trades + losing_trades)
                        avg_win = total_profit / max(1, profitable_trades)
                        avg_loss = total_loss / max(1, losing_trades)

                        if avg_loss > 0:
                            kelly_f = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                            kelly_f = max(0, min(0.5, kelly_f))  # 限制在0-50%范围
                            position_amount = min(cash * kelly_f, cash)
                        else:
                            position_amount = min(cash * 0.1, cash)  # 默认使用10%
                    else:  # percent
                        position_amount = min(cash * (position_size / 100), cash)

                    # 根据异常信号强度调整仓位
                    signal_strength = df['z_score'].iloc[i] if 'z_score' in df.columns and i < len(df) else 2.0
                    # 信号越强，仓位越大，但不超过原定仓位的1.5倍
                    position_multiplier = min(1.5, 1.0 + (signal_strength - 2.0) / 3.0)
                    position_amount = min(position_amount * position_multiplier, cash)

                    # 如果有足够资金，执行开仓
                    if 0 < position_amount <= cash:
                        # 根据异常信号决定交易方向
                        if current_anomaly_signal > 0:  # 比值高于拟合线，做空A做多B
                            trade_signal = 'short'
                            signal_type = 'high_ratio'
                        else:  # 比值低于拟合线，做多A做空B
                            trade_signal = 'long'
                            signal_type = 'low_ratio'

                        # 扣除费用
                        fee = position_amount * trading_fee
                        actual_position = position_amount - fee

                        # 确定交易资产和方向
                        if hedge_mode == 'pair':
                            # 在对模式下，同时开两个相反方向的仓位
                            if trade_signal == 'short':
                                # 做空A，做多B
                                current_positions.append({
                                    'asset': code_a,
                                    'direction': 'short',
                                    'entry_date': current_date,
                                    'entry_price': current_price_a,
                                    'current_price': current_price_a,
                                    'position_size': actual_position / 2,  # 平分资金
                                    'pnl': -fee / 2,
                                    'pnl_percent': -fee / (actual_position / 2),
                                    'entry_signal': current_anomaly_signal,
                                    'entry_signal_type': signal_type,
                                    'entry_signal_strength': signal_strength,
                                    'holding_days': 0
                                })

                                current_positions.append({
                                    'asset': code_b,
                                    'direction': 'long',
                                    'entry_date': current_date,
                                    'entry_price': current_price_b,
                                    'current_price': current_price_b,
                                    'position_size': actual_position / 2,  # 平分资金
                                    'pnl': -fee / 2,
                                    'pnl_percent': -fee / (actual_position / 2),
                                    'entry_signal': current_anomaly_signal,
                                    'entry_signal_type': signal_type,
                                    'entry_signal_strength': signal_strength,
                                    'holding_days': 0
                                })
                            else:
                                # 做多A，做空B
                                current_positions.append({
                                    'asset': code_a,
                                    'direction': 'long',
                                    'entry_date': current_date,
                                    'entry_price': current_price_a,
                                    'current_price': current_price_a,
                                    'position_size': actual_position / 2,  # 平分资金
                                    'pnl': -fee / 2,
                                    'pnl_percent': -fee / (actual_position / 2),
                                    'entry_signal': current_anomaly_signal,
                                    'entry_signal_type': signal_type,
                                    'entry_signal_strength': signal_strength,
                                    'holding_days': 0
                                })

                                current_positions.append({
                                    'asset': code_b,
                                    'direction': 'short',
                                    'entry_date': current_date,
                                    'entry_price': current_price_b,
                                    'current_price': current_price_b,
                                    'position_size': actual_position / 2,  # 平分资金
                                    'pnl': -fee / 2,
                                    'pnl_percent': -fee / (actual_position / 2),
                                    'entry_signal': current_anomaly_signal,
                                    'entry_signal_type': signal_type,
                                    'entry_signal_strength': signal_strength,
                                    'holding_days': 0
                                })
                        else:
                            # 单边模式，只交易资产A
                            current_positions.append({
                                'asset': code_a,
                                'direction': trade_signal,
                                'entry_date': current_date,
                                'entry_price': current_price_a,
                                'current_price': current_price_a,
                                'position_size': actual_position,
                                'pnl': -fee,
                                'pnl_percent': -fee / actual_position,
                                'entry_signal': current_anomaly_signal,
                                'entry_signal_type': signal_type,
                                'entry_signal_strength': signal_strength,
                                'holding_days': 0
                            })

                        # 更新现金
                        cash -= position_amount

                # 计算总权益
                positions_value = sum(pos['position_size'] + pos['pnl'] for pos in current_positions)
                equity = cash + positions_value

                # 计算日收益率
                daily_return = 0
                if i > 0:
                    previous_equity = equity_curve[-1]['equity']
                    if previous_equity > 0:
                        daily_return = (equity - previous_equity) / previous_equity
                daily_returns.append(daily_return)

                # 更新最大权益（用于计算回撤）
                if equity > last_equity_peak:
                    last_equity_peak = equity
                    drawdown_start = None

                # 计算回撤相关指标
                current_drawdown = (last_equity_peak - equity) / last_equity_peak if last_equity_peak > 0 else 0

                if current_drawdown > 0:
                    if drawdown_start is None:
                        drawdown_start = i
                    drawdown_duration = i - drawdown_start

                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                        max_drawdown_duration = drawdown_duration
                else:
                    drawdown_start = None

                # 记录权益曲线
                equity_curve.append({
                    'date': current_date,
                    'equity': equity,
                    'drawdown': -current_drawdown,  # 负值表示回撤
                    'cash': cash,
                    'positions_value': positions_value,
                    'anomaly_signal': current_anomaly_signal,
                    'ratio': current_ratio,
                    'delta': current_delta
                })

            # 回测结束，平掉所有未平仓的头寸
            for pos in current_positions:
                # 计算交易费用
                fee = pos['position_size'] * trading_fee

                # 更新交易统计
                if pos['pnl'] > 0:
                    profitable_trades += 1
                    total_profit += pos['pnl']
                else:
                    losing_trades += 1
                    total_loss += abs(pos['pnl'])

                # 记录持仓时间
                position_days.append(pos['holding_days'])

                # 记录交易
                trades.append({
                    'id': trade_id,
                    'entry_date': pos['entry_date'],
                    'exit_date': df['date'].iloc[-1],  # 使用最后一个交易日
                    'holding_days': pos['holding_days'],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['current_price'],
                    'position_type': f"{pos['direction']}_{pos['asset']}",
                    'trade_direction': 'long_A_short_B' if pos.get('entry_signal_type') == 'low_ratio' else 'short_A_long_B',
                    'position_size': pos['position_size'],
                    'entry_signal_type': pos.get('entry_signal_type', 'unknown'),
                    'entry_signal_strength': pos.get('entry_signal_strength', 0),
                    'pnl': pos['pnl'] - fee,
                    'pnl_percent': pos['pnl_percent'] - (fee / pos['position_size']),
                    'status': 'closed',
                    'exit_reason': 'end_of_backtest'
                })
                trade_id += 1

                # 更新现金
                cash += pos['position_size'] + pos['pnl'] - fee

            # 清空当前持仓
            current_positions = []

            # 计算最终权益
            final_equity = cash

            # 计算回测性能指标
            total_trades = profitable_trades + losing_trades
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            # 计算平均收益和亏损
            avg_profit = total_profit / profitable_trades if profitable_trades > 0 else 0
            avg_loss = total_loss / losing_trades if losing_trades > 0 else 0

            # 计算利润因子
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # 计算年化收益率
            first_date = datetime.strptime(df['date'].iloc[0], '%Y-%m-%d')
            last_date = datetime.strptime(df['date'].iloc[-1], '%Y-%m-%d')
            years = (last_date - first_date).days / 365.25
            total_return = (final_equity - initial_capital) / initial_capital
            annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if years > 0 else 0

            # 计算夏普比率
            if len(daily_returns) > 0:
                avg_daily_return = np.mean(daily_returns)
                daily_std = np.std(daily_returns)
                risk_free_rate = 0.02 / 252  # 假设年化无风险利率为2%
                sharpe_ratio = (avg_daily_return - risk_free_rate) / daily_std * np.sqrt(252) if daily_std > 0 else 0
            else:
                sharpe_ratio = 0

            # 计算索提诺比率（使用最大回撤作为风险度量）
            sortino_ratio = 0
            if len(daily_returns) > 0:
                negative_returns = [r for r in daily_returns if r < 0]
                if negative_returns:
                    downside_std = np.std(negative_returns)
                    if downside_std > 0:
                        sortino_ratio = (avg_daily_return - risk_free_rate) / downside_std * np.sqrt(252)

            # 计算最大回撤恢复期
            recovery_period = 0
            max_dd_idx = 0
            for i, point in enumerate(equity_curve):
                if point['drawdown'] == -max_drawdown:
                    max_dd_idx = i
                    break

            # 从最大回撤点向后查找第一个恢复到回撤前水平的点
            peak_equity = equity_curve[max_dd_idx]['equity'] / (1 - max_drawdown)
            for i in range(max_dd_idx, len(equity_curve)):
                if equity_curve[i]['equity'] >= peak_equity:
                    recovery_period = i - max_dd_idx
                    break

            # 计算平均持仓时间
            avg_holding_period = np.mean(position_days) if position_days else 0

            # 计算卡尔马比率（年化收益/最大回撤）
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

            # 异常点信号统计
            anomaly_signals_count = sum(1 for i in range(len(df)) if df['anomaly_signal'].iloc[i] != 0)
            long_signals_count = sum(1 for i in range(len(df)) if df['anomaly_signal'].iloc[i] < 0)
            short_signals_count = sum(1 for i in range(len(df)) if df['anomaly_signal'].iloc[i] > 0)

            # 构建回测结果
            backtest_result = {
                'equity_curve': equity_curve,
                'trades': trades,
                'initial_capital': initial_capital,
                'final_equity': final_equity,
                'total_return': total_return * 100,  # 转为百分比
                'annual_return': annual_return * 100,  # 转为百分比
                'max_drawdown': max_drawdown * 100,  # 转为百分比
                'max_drawdown_duration': max_drawdown_duration,
                'recovery_period': recovery_period,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate * 100,  # 转为百分比
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_holding_period': avg_holding_period,
                'strategy_parameters': {
                    'anomaly_threshold': anomaly_threshold,
                    'polynomial_degree': polynomial_degree,
                    'stop_loss': stop_loss * 100,  # 转为百分比
                    'take_profit': take_profit * 100,  # 转为百分比
                    'trailing_stop': trailing_stop * 100,  # 转为百分比
                    'time_stop': time_stop,
                    'hedge_mode': hedge_mode,
                    'position_size_type': position_size_type,
                    'position_size': position_size,
                    'mean_reversion_exit': mean_reversion_exit,
                    'mean_reversion_threshold': mean_reversion_threshold,
                    'reverse_anomaly_exit': reverse_anomaly_exit
                },
                'anomaly_signal_stats': {
                    'total_signals': anomaly_signals_count,
                    'long_signals': long_signals_count,
                    'short_signals': short_signals_count
                }
            }

            # 转换NumPy类型为JSON可序列化的类型
            return convert_numpy_types(backtest_result)

        except Exception as e:
            import traceback
            return {
                "error": f"回测执行过程中发生错误: {str(e)}",
                "traceback": traceback.format_exc()
            }

    def calculate_optimal_threshold(self, prices_a: List[float], prices_b: List[float],
                                    dates: List[str], lookback: int = 60) -> Dict[str, Any]:
        """
        计算最优入场和出场阈值

        参数:
            prices_a: 资产A价格列表
            prices_b: 资产B价格列表
            dates: 对应日期列表
            lookback: 向后看的天数，用于计算

        返回:
            包含最优阈值的字典
        """
        try:
            # 计算价格比率
            ratios = [a / b for a, b in zip(prices_a, prices_b)]

            if len(ratios) < lookback * 2:
                return {"error": "数据量不足，无法计算最优阈值"}

            # 创建DataFrame便于分析
            df = pd.DataFrame({
                'date': dates,
                'ratio': ratios,
                'price_a': prices_a,
                'price_b': prices_b
            })

            # 计算基础指标
            df['returns'] = df['ratio'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

            # 保存所有可能的阈值组合的结果
            results = []

            # 计算趋势指标
            df['sma20'] = df['ratio'].rolling(window=20).mean()
            df['sma50'] = df['ratio'].rolling(window=50).mean()
            df['trend'] = np.where(df['sma20'] > df['sma50'], 1, -1)  # 1=上升趋势，-1=下降趋势

            # 确定市场环境类型
            df['vol_percentile'] = df['volatility'].rolling(window=60).apply(
                lambda x: percentile_rank(x[-1], x), raw=True).fillna(0.5)

            # 定义市场类型：低波动、正常、高波动
            df['market_type'] = pd.cut(
                df['vol_percentile'],
                bins=[0, 0.33, 0.67, 1],
                labels=['low_vol', 'normal', 'high_vol']
            )

            # 测试不同的入场/出场阈值组合 - 针对不同市场环境
            for market_type in ['low_vol', 'normal', 'high_vol']:
                market_df = df[df['market_type'] == market_type].copy() if len(df) > 60 else df

                if len(market_df) < 30:  # 确保有足够的数据
                    continue

                # 为不同市场类型调整测试范围
                if market_type == 'low_vol':
                    entry_range = np.arange(1.0, 2.1, 0.1)  # 低波动市场用较低阈值
                    exit_range = np.arange(0.1, 0.7, 0.1)
                elif market_type == 'high_vol':
                    entry_range = np.arange(2.0, 3.6, 0.2)  # 高波动市场用较高阈值
                    exit_range = np.arange(0.5, 1.1, 0.1)
                else:  # normal
                    entry_range = np.arange(1.5, 3.1, 0.1)  # 正常市场用中等阈值
                    exit_range = np.arange(0.3, 0.9, 0.1)

                for entry_t in entry_range:
                    for exit_t in exit_range:
                        if exit_t >= entry_t:
                            continue  # 出场阈值应小于入场阈值

                        # 模拟使用这些阈值的交易
                        trades = []
                        in_position = False
                        entry_price = 0
                        max_drawdown = 0  # 跟踪最大回撤
                        current_drawdown = 0
                        peak_equity = 1.0  # 假设初始权益为1
                        equity = 1.0

                        for i in range(lookback, len(market_df)):
                            # 计算移动窗口的均值和标准差
                            window = market_df['ratio'].iloc[i - lookback:i]
                            mean = np.mean(window)
                            std = np.std(window)

                            # 计算当前Z-score
                            z_score = (market_df['ratio'].iloc[i] - mean) / std if std > 0 else 0

                            # 获取当前趋势
                            trend = market_df['trend'].iloc[i] if 'trend' in market_df.columns else 1

                            if not in_position:
                                # 寻找入场点 - 考虑趋势方向
                                if (abs(z_score) > entry_t and
                                        ((z_score > 0 and trend < 0) or  # 比值高且趋势向下
                                         (z_score < 0 and trend > 0))):  # 比值低且趋势向上
                                    in_position = True
                                    entry_price = market_df['ratio'].iloc[i]
                                    entry_date = market_df['date'].iloc[i]
                                    entry_zscore = z_score
                                    trade_equity = equity  # 记录开仓时的权益
                            else:
                                # 寻找出场点
                                if abs(z_score) < exit_t:
                                    in_position = False
                                    exit_price = market_df['ratio'].iloc[i]
                                    # 计算这笔交易的收益
                                    if entry_zscore > 0:  # 做空入场
                                        profit = entry_price - exit_price
                                    else:  # 做多入场
                                        profit = exit_price - entry_price

                                    profit_pct = profit / entry_price

                                    # 更新权益
                                    equity *= (1 + profit_pct)

                                    # 更新峰值权益和回撤
                                    if equity > peak_equity:
                                        peak_equity = equity
                                    current_drawdown = (peak_equity - equity) / peak_equity
                                    max_drawdown = max(max_drawdown, current_drawdown)

                                    trades.append({
                                        'entry_date': entry_date,
                                        'exit_date': market_df['date'].iloc[i],
                                        'entry_zscore': entry_zscore,
                                        'exit_zscore': z_score,
                                        'profit': profit_pct,
                                        'equity': equity,
                                        'drawdown': current_drawdown
                                    })

                                # 检查强制止损 (15% 止损)
                                elif ((entry_zscore > 0 and (
                                        exit_price - entry_price) / entry_price > 0.15) or  # 做空但上涨超过15%
                                      (entry_zscore < 0 and (
                                              entry_price - exit_price) / entry_price > 0.15)):  # 做多但下跌超过15%
                                    in_position = False
                                    exit_price = market_df['ratio'].iloc[i]

                                    # 计算亏损
                                    if entry_zscore > 0:  # 做空入场
                                        profit = entry_price - exit_price
                                    else:  # 做多入场
                                        profit = exit_price - entry_price

                                    profit_pct = profit / entry_price

                                    # 更新权益
                                    equity *= (1 + profit_pct)

                                    # 更新回撤
                                    current_drawdown = (peak_equity - equity) / peak_equity
                                    max_drawdown = max(max_drawdown, current_drawdown)

                                    trades.append({
                                        'entry_date': entry_date,
                                        'exit_date': market_df['date'].iloc[i],
                                        'entry_zscore': entry_zscore,
                                        'exit_zscore': z_score,
                                        'profit': profit_pct,
                                        'equity': equity,
                                        'drawdown': current_drawdown,
                                        'reason': 'stop_loss'
                                    })

                        # 计算策略绩效
                        if trades:
                            # 收益相关指标
                            profits = [t['profit'] for t in trades]
                            cum_profit = np.prod([1 + p for p in profits]) - 1
                            avg_profit = np.mean(profits)
                            win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades)

                            # 计算盈亏比
                            gains = [p for p in profits if p > 0]
                            losses = [abs(p) for p in profits if p < 0]
                            profit_factor = sum(gains) / abs(sum(losses)) if losses and sum(losses) != 0 else float(
                                'inf')

                            # 考虑回撤
                            max_dd = max_drawdown

                            # 计算卡尔马比率 (年化收益/最大回撤)
                            calmar = cum_profit / max_dd if max_dd > 0 else float('inf')

                            # 综合得分 - 权衡回报和风险
                            # 优化目标：高胜率 + 高盈亏比 + 低回撤 + 适中交易频率
                            score = (win_rate * 0.3 + profit_factor * 0.3 + calmar * 0.3 +
                                     min(1, len(trades) / 50) * 0.1)  # 交易次数奖励，但最多占10%权重

                            results.append({
                                'market_type': market_type,
                                'entry_threshold': float(entry_t),
                                'exit_threshold': float(exit_t),
                                'avg_profit': float(avg_profit),
                                'win_rate': float(win_rate),
                                'profit_factor': float(profit_factor),
                                'trade_count': int(len(trades)),
                                'max_drawdown': float(max_dd),
                                'calmar_ratio': float(calmar),
                                'cumulative_return': float(cum_profit),
                                'score': float(score)
                            })

            # 如果没有结果，返回默认值
            if not results:
                return {
                    "entry_threshold": 2.0,
                    "exit_threshold": 0.5,
                    "score": 0,
                    "trade_count": 0,
                    "error": "无法找到最优阈值，返回默认值"
                }

            # 根据得分排序
            results.sort(key=lambda x: x['score'], reverse=True)

            # 获取综合最优值
            best_overall = results[0]

            # 根据市场类型获取最优值
            best_by_market = {}
            for market_type in ['low_vol', 'normal', 'high_vol']:
                market_results = [r for r in results if r['market_type'] == market_type]
                if market_results:
                    market_results.sort(key=lambda x: x['score'], reverse=True)
                    best_by_market[market_type] = market_results[0]

            # 将结果转换为JSON可序列化类型
            return convert_numpy_types({
                "entry_threshold": best_overall['entry_threshold'],
                "exit_threshold": best_overall['exit_threshold'],
                "score": best_overall['score'],
                "win_rate": best_overall['win_rate'],
                "profit_factor": best_overall['profit_factor'],
                "max_drawdown": best_overall['max_drawdown'],
                "calmar_ratio": best_overall['calmar_ratio'],
                "trade_count": best_overall['trade_count'],
                "market_type": best_overall['market_type'],
                "best_by_market": best_by_market,
                "top_results": results[:5]  # 返回前5个最佳结果
            })

        except Exception as e:
            import traceback
            return {
                "error": f"计算最优阈值时发生错误: {str(e)}",
                "traceback": traceback.format_exc(),
                "entry_threshold": 2.0,  # 返回默认值
                "exit_threshold": 0.5
            }

    def run_similar_signals_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行基于投资信号相似度的回测策略

        该策略仅对当前比值位置与历史上最相似的几个投资信号进行回测，
        以验证当前投资机会的潜在表现。

        参数:
            params: 回测参数字典，包含：
                - code_a: 资产A代码
                - code_b: 资产B代码
                - initial_capital: 初始资金
                - position_size: 仓位大小(百分比)
                - stop_loss: 止损比例(%)
                - take_profit: 止盈比例(%)
                - trading_fee: 交易费率
                - polynomial_degree: 多项式拟合的次数
                - threshold_multiplier: 信号生成的阈值乘数
                - duration: 时间跨度 (1m, 3m, 1y, 2y, 5y, maximum)

        返回:
            回测结果字典，包含交易记录、资金曲线和性能指标
        """
        try:
            # 1. 获取基本参数
            code_a = params.get('code_a')
            code_b = params.get('code_b')
            initial_capital = float(params.get('initial_capital', 100000))
            position_size = float(params.get('position_size', 30))  # 默认30%仓位
            stop_loss = float(params.get('stop_loss', 5)) / 100  # 转换为小数
            take_profit = float(params.get('take_profit', 10)) / 100  # 转换为小数
            trading_fee = float(params.get('trading_fee', 0.0003))
            polynomial_degree = int(params.get('polynomial_degree', 3))
            threshold_multiplier = float(params.get('threshold_multiplier', 1.5))
            duration = params.get('duration', '1y')  # 默认使用1年的时间跨度

            # 2. 获取历史数据
            close_a_, close_b_, dates_a_, dates_b_ = get_stock_data_pair(code_a, code_b)
            close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)

            if len(dates) < 60:
                return {"error": "历史数据不足，无法进行回测。至少需要60个交易日的数据。"}

            # 根据时间跨度截取数据
            duration_days = durations.get(duration, -1)
            if duration_days != -1 and duration_days < len(close_a):
                close_a = close_a[-duration_days:]
                close_b = close_b[-duration_days:]
                dates = dates[-duration_days:]

            # 3. 生成历史投资信号
            signals = generate_investment_signals(
                close_a=close_a,
                close_b=close_b,
                dates=dates,
                degree=polynomial_degree,
                threshold_multiplier=threshold_multiplier
            )

            if not signals:
                return {"error": "未能生成有效的投资信号，无法进行回测。"}
            # 获取当前最新价格趋势
            db = StockTrendsDatabase()
            trends_a = db.query_trends(stock_code=code_a)
            trends_b = db.query_trends(stock_code=code_b)
            # 4. 分析当前价格比值位置，找出最相似的信号
            current_analysis = analyze_current_position(
                code_a=code_a,
                code_b=code_b,
                close_a=close_a,
                close_b=close_b,
                dates=dates,
                signals=signals,
                trends_a=trends_a,
                trends_b=trends_b,
                degree=polynomial_degree
            )

            similar_signals = current_analysis.get("nearest_signals", [])
            if not similar_signals:
                return {"error": "未找到与当前价格比值相似的历史信号，无法进行回测。"}

            # 5. 初始化回测结果
            trades = []  # 交易记录
            equity_curve = []  # 资金曲线
            cash = initial_capital
            equity = initial_capital
            trade_id = 1

            # 6. 针对每个相似信号进行回测
            for signal in similar_signals:
                try:
                    signal_id = signal["id"]
                    signal_date = signal["date"]
                    similarity = signal["similarity"]
                    signal_type = signal["type"]
                    signal_strength = signal["strength"]

                    # 查找原始信号的详细信息
                    original_signal = next((s for s in signals if s["id"] == signal_id), None)
                    if not original_signal:
                        continue

                    # 找到信号在历史数据中的位置
                    try:
                        signal_index = dates.index(signal_date)
                    except ValueError:
                        continue

                    # 确保有足够的后续数据进行回测（至少30天）
                    if signal_index + 30 >= len(dates):
                        continue

                    # 确定交易方向
                    direction = "short" if signal_type == "positive" else "long"

                    # 确定仓位大小（根据信号强度和相似度调整）
                    strength_factor = {"weak": 0.7, "medium": 1.0, "strong": 1.3}.get(signal_strength, 1.0)
                    similarity_factor = min(1.0, similarity * 1.1)  # 相似度越高，仓位越大
                    position_percent = position_size * strength_factor * similarity_factor / 100
                    position_amount = min(cash * position_percent, cash)

                    # 计算交易费用
                    fee = position_amount * trading_fee
                    actual_position = position_amount - fee

                    # 入场价格
                    entry_price_a = close_a[signal_index]
                    entry_price_b = close_b[signal_index]

                    # 记录入场交易
                    entry_date = dates[signal_index]
                    entry_ratio = entry_price_a / entry_price_b

                    # 更新资金
                    cash -= position_amount

                    # 追踪价格变化和盈亏
                    max_profit = 0
                    max_loss = 0
                    exit_index = None
                    exit_reason = None

                    # 模拟后续交易日
                    for i in range(signal_index + 1, min(signal_index + 60, len(dates))):
                        current_price_a = close_a[i]
                        current_price_b = close_b[i]
                        current_ratio = current_price_a / current_price_b

                        # 计算盈亏
                        if direction == "long":  # 做多比值（买A卖B）
                            pnl_percent = (current_ratio - entry_ratio) / entry_ratio
                        else:  # 做空比值（卖A买B）
                            pnl_percent = (entry_ratio - current_ratio) / entry_ratio

                        # 追踪最大盈亏
                        if pnl_percent > max_profit:
                            max_profit = pnl_percent
                        if pnl_percent < max_loss:
                            max_loss = pnl_percent

                        # 检查平仓条件
                        # 1. 止损
                        if pnl_percent <= -stop_loss:
                            exit_index = i
                            exit_reason = "止损"
                            break

                        # 2. 止盈
                        if pnl_percent >= take_profit:
                            exit_index = i
                            exit_reason = "止盈"
                            break

                        # 3. 自适应跟踪止损 - 高收益时保护利润
                        if max_profit >= take_profit * 0.7 and pnl_percent <= max_profit * 0.6:
                            exit_index = i
                            exit_reason = "跟踪止损"
                            break

                        # 4. 最长持仓时间（30天）
                        if i - signal_index >= 30:
                            exit_index = i
                            exit_reason = "时间到期"
                            break

                    # 如果没有触发任何平仓条件，使用最后一个可用数据点
                    if exit_index is None:
                        exit_index = min(signal_index + 30, len(dates) - 1)
                        exit_reason = "时间到期"

                    # 计算最终盈亏
                    exit_price_a = close_a[exit_index]
                    exit_price_b = close_b[exit_index]
                    exit_ratio = exit_price_a / exit_price_b
                    exit_date = dates[exit_index]

                    if direction == "long":
                        final_pnl_percent = (exit_ratio - entry_ratio) / entry_ratio
                    else:
                        final_pnl_percent = (entry_ratio - exit_ratio) / entry_ratio

                    final_pnl = actual_position * final_pnl_percent

                    # 平仓交易费
                    exit_fee = (actual_position + final_pnl) * trading_fee
                    final_pnl -= exit_fee

                    # 更新资金
                    cash += actual_position + final_pnl

                    # 记录交易
                    trades.append({
                        'id': trade_id,
                        'entry_date': entry_date,
                        'exit_date': exit_date,
                        'holding_days': exit_index - signal_index,
                        'entry_ratio': entry_ratio,
                        'exit_ratio': exit_ratio,
                        'direction': direction,
                        'position_size': actual_position,
                        'pnl': final_pnl,
                        'pnl_percent': final_pnl_percent * 100,  # 转为百分比显示
                        'exit_reason': exit_reason,
                        'similar_signal_id': signal_id,
                        'similarity': similarity
                    })

                    trade_id += 1

                    # 更新权益
                    equity = cash

                except Exception as e:
                    print(f"处理信号{signal_id}时发生错误: {str(e)}")
                    continue

            # 7. 计算回测性能指标
            final_equity = cash
            total_return = (final_equity - initial_capital) / initial_capital * 100

            # 统计盈亏
            profitable_trades = sum(1 for t in trades if t['pnl'] > 0)
            total_trades = len(trades)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0

            # 计算平均盈亏
            avg_profit = sum(
                t['pnl'] for t in trades if t['pnl'] > 0) / profitable_trades if profitable_trades > 0 else 0
            avg_loss = sum(abs(t['pnl']) for t in trades if t['pnl'] <= 0) / (
                    total_trades - profitable_trades) if total_trades - profitable_trades > 0 else 0
            profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0

            # 计算平均持仓时间
            avg_holding_days = sum(t['holding_days'] for t in trades) / total_trades if total_trades > 0 else 0

            # 生成资金曲线数据
            portfolio_value_history = []

            # 按日期排序交易记录
            sorted_trades = sorted(trades, key=lambda x: x['entry_date'])

            # 添加初始资金点
            portfolio_value_history.append({
                'date': sorted_trades[0]['entry_date'] if sorted_trades else dates[-1],
                'portfolio_value': initial_capital
            })

            # 累积每笔交易后的资金变化
            current_value = initial_capital
            for trade in sorted_trades:
                current_value += trade['pnl']
                portfolio_value_history.append({
                    'date': trade['exit_date'],
                    'portfolio_value': current_value
                })

            # 返回回测结果
            return convert_numpy_types({
                "trades": trades,
                "initial_capital": initial_capital,
                "final_equity": final_equity,
                "total_return": total_return,
                "total_trades": total_trades,
                "profitable_trades": profitable_trades,
                "win_rate": win_rate * 100,  # 转为百分比显示
                "profit_loss_ratio": profit_loss_ratio,
                "avg_holding_days": avg_holding_days,
                "current_analysis": current_analysis,
                "similar_signals": similar_signals,
                "portfolio_value_history": portfolio_value_history  # 添加资金曲线数据
            })

        except Exception as e:
            return {"error": f"执行回测时发生错误: {str(e)}"}


def percentile_rank(x, values):
    """计算x在values中的百分位排名 (0-1)"""
    if len(values) == 0:
        return 0.5
    return sum(1 for val in values if val < x) / len(values)
