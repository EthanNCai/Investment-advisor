import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import math

# 从现有模块导入函数
from k_chart_fetcher import get_stock_data_pair, date_alignment


class BacktestEngine:
    def __init__(self):
        """初始化回测引擎"""
        pass

    def run_price_ratio_backtest(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        运行价差交易回测

        参数:
            params: 回测参数字典，包含：
                - start_date: 回测开始日期
                - end_date: 回测结束日期
                - code_a: 资产A代码
                - code_b: 资产B代码
                - initial_capital: 初始资金
                - position_size_type: 仓位计算方式 ('fixed'/'percent'/'kelly')
                - position_size: 仓位大小
                - entry_threshold: 入场阈值
                - exit_threshold: 出场阈值
                - stop_loss: 止损比例(%)
                - take_profit: 止盈比例(%)
                - max_positions: 最大持仓数量
                - trading_fee: 交易费率
                - trailing_stop: 追踪止损(%) 可选
                - time_stop: 最长持仓时间(天) 可选
                - strategy_type: 策略类型 ('zscore'/'percent'/'volatility'/'trend')
                - hedge_mode: 对冲模式 ('single'/'pair') 单边或双边交易
                - risk_reward_ratio: 风险收益比 可选
                - breakeven_stop: 是否使用保本止损 可选

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
            entry_threshold = float(params.get('entry_threshold', 2.0))
            exit_threshold = float(params.get('exit_threshold', 0.5))
            stop_loss = float(params.get('stop_loss', 5)) / 100  # 转换为小数
            take_profit = float(params.get('take_profit', 10)) / 100  # 转换为小数
            max_positions = int(params.get('max_positions', 5))
            trading_fee = float(params.get('trading_fee', 0.0003))

            # 新增参数
            trailing_stop = float(params.get('trailing_stop', 0)) / 100  # 追踪止损百分比
            time_stop = int(params.get('time_stop', 0))  # 最长持仓时间（天）
            strategy_type = params.get('strategy_type', 'zscore')  # 策略类型
            hedge_mode = params.get('hedge_mode', 'single')  # 对冲模式
            risk_reward_ratio = float(params.get('risk_reward_ratio', 2.0))  # 风险收益比
            breakeven_stop = params.get('breakeven_stop', False)  # 是否使用保本止损

            # 设置波动率窗口
            volatility_window = int(params.get('volatility_window', 20))
            # 设置趋势判断参数
            trend_window = int(params.get('trend_window', 50))

            # 设置交易相关的阈值参数（对于不同策略类型有不同的解释）
            secondary_threshold = float(params.get('secondary_threshold', 1.0))

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

            # 基于不同策略类型进行相应的计算
            if strategy_type == 'zscore':
                # Z分数策略 - 使用滚动均值和标准差
                window = 60
                df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
                df['ratio_std'] = df['ratio'].rolling(window=window).std()
                df['zscore'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']
                df['signal'] = df['zscore']

            elif strategy_type == 'percent':
                # 百分比偏离策略 - 相对于移动平均的偏离百分比
                window = 60
                df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
                df['percent_deviation'] = (df['ratio'] - df['ratio_ma']) / df['ratio_ma'] * 100
                df['signal'] = df['percent_deviation'] / secondary_threshold  # 标准化信号

            elif strategy_type == 'volatility':
                # 波动率调整策略 - 考虑市场波动率的Z分数
                window = 60
                vol_window = volatility_window
                df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
                df['ratio_std'] = df['ratio'].rolling(window=window).std()
                # 计算动态波动率
                df['volatility'] = df['ratio_return'].rolling(window=vol_window).std() * np.sqrt(252)
                # 使用相对波动率调整Z分数
                df['vol_adj_factor'] = df['volatility'].rolling(window=vol_window).mean() / df['volatility']
                df['vol_adj_factor'] = df['vol_adj_factor'].fillna(1)
                # 波动率调整后的Z分数
                df['zscore'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']
                df['vol_adj_zscore'] = df['zscore'] * df['vol_adj_factor']
                df['signal'] = df['vol_adj_zscore']

            elif strategy_type == 'trend':
                # 趋势跟踪策略 - 考虑价格比率的趋势方向
                window = 60
                short_window = 20
                long_window = trend_window
                # 计算基本指标
                df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
                df['ratio_std'] = df['ratio'].rolling(window=window).std()
                df['zscore'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']
                # 计算趋势指标
                df['short_ma'] = df['ratio'].rolling(window=short_window).mean()
                df['long_ma'] = df['ratio'].rolling(window=long_window).mean()
                df['trend'] = np.where(df['short_ma'] > df['long_ma'], 1, -1)
                # 结合Z分数和趋势
                df['signal'] = df['zscore'] * df['trend'] * secondary_threshold

            else:
                # 默认使用Z分数策略
                window = 60
                df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
                df['ratio_std'] = df['ratio'].rolling(window=window).std()
                df['zscore'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']
                df['signal'] = df['zscore']

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
                current_signal = df['signal'].iloc[i]
                current_price_a = df['close_a'].iloc[i]
                current_price_b = df['close_b'].iloc[i]
                current_ratio = df['ratio'].iloc[i]

                # 更新现有持仓信息
                positions_to_close = []

                # 更新当前持仓的价值
                for j, pos in enumerate(current_positions):
                    if pos['asset'] == code_a:
                        print('asset a')
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
                    elif time_stop > 0 and pos['holding_days'] >= time_stop:
                        close_reason = 'time_stop'

                    # 5. 保本止损检查
                    elif breakeven_stop and pos['pnl_percent'] < 0 and pos.get('highest_profit_percent', 0) >= 0.03:
                        # 如果曾经盈利超过3%但现在跌至亏损，则平仓保本
                        close_reason = 'breakeven_stop'

                    # 6. 信号平仓检查
                    elif ((pos['asset'] == code_a and pos[
                        'direction'] == 'long' and current_signal >= -exit_threshold) or
                          (pos['asset'] == code_a and pos[
                              'direction'] == 'short' and current_signal <= exit_threshold)):
                        close_reason = 'signal'

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
                        'position_size': pos['position_size'],
                        'pnl': pos['pnl'] - fee,
                        'pnl_percent': pos['pnl_percent'] - (fee / pos['position_size']),
                        'status': 'closed',
                        'exit_reason': reason,
                        'signal_value': pos.get('entry_signal', 0)
                    })
                    trade_id += 1

                    # 移除平仓的头寸
                    current_positions.pop(j)

                # 检查开仓信号
                if len(current_positions) < max_positions:
                    # 计算每个头寸的大小
                    if position_size_type == 'fixed':
                        position_amount = min(position_size, cash)
                    elif position_size_type == 'kelly':
                        # 简化的Kelly公式实现
                        win_rate = profitable_trades / max(1, profitable_trades + losing_trades)
                        avg_win = total_profit / max(1, profitable_trades)
                        avg_loss = total_loss / max(1, losing_trades)
                        if avg_loss > 0:
                            kelly_f = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
                            kelly_f = max(0, min(0.5, kelly_f))  # 限制在0-50%之间，保守使用
                            position_amount = min(cash * kelly_f, cash)
                        else:
                            position_amount = min(cash * 0.1, cash)  # 默认使用10%
                    else:  # percent
                        position_amount = min(cash * (position_size / 100), cash)

                    # 如果有足够资金，检查开仓信号
                    if 0 < position_amount <= cash:
                        # 根据策略类型和信号值决定交易方向
                        trade_signal = None

                        # 信号高于阈值，做空资产A（或做多资产B，取决于对冲模式）
                        if current_signal > entry_threshold:
                            trade_signal = 'short'
                        # 信号低于负阈值，做多资产A（或做空资产B）
                        elif current_signal < -entry_threshold:
                            trade_signal = 'long'

                        if trade_signal:
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
                                        'entry_signal': current_signal
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
                                        'entry_signal': current_signal
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
                                        'entry_signal': current_signal
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
                                        'entry_signal': current_signal
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
                                    'entry_signal': current_signal
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
                    'signal': current_signal,
                    'ratio': current_ratio
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
                    'position_size': pos['position_size'],
                    'pnl': pos['pnl'] - fee,
                    'pnl_percent': pos['pnl_percent'] - (fee / pos['position_size']),
                    'status': 'closed',
                    'exit_reason': 'end_of_backtest',
                    'signal_value': pos.get('entry_signal', 0)
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
            annual_return = (1 + total_return) ** (1 / max(years, 0.01)) - 1

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
                    'strategy_type': strategy_type,
                    'entry_threshold': entry_threshold,
                    'exit_threshold': exit_threshold,
                    'stop_loss': stop_loss * 100,  # 转为百分比
                    'take_profit': take_profit * 100,  # 转为百分比
                    'trailing_stop': trailing_stop * 100,  # 转为百分比
                    'time_stop': time_stop,
                    'hedge_mode': hedge_mode,
                    'position_size_type': position_size_type,
                    'position_size': position_size
                }
            }

            return backtest_result

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
        # 计算价格比率
        ratios = [a / b for a, b in zip(prices_a, prices_b)]

        # 保存所有可能的阈值组合的结果
        results = []

        # 测试不同的入场/出场阈值组合
        for entry_t in np.arange(1.0, 3.1, 0.1):
            for exit_t in np.arange(0.1, min(entry_t, 1.1), 0.1):
                # 模拟使用这些阈值的交易
                trades = []
                in_position = False
                entry_price = 0

                for i in range(lookback, len(ratios)):
                    # 计算移动窗口的均值和标准差
                    window = ratios[i - lookback:i]
                    mean = np.mean(window)
                    std = np.std(window)

                    # 计算当前Z-score
                    z_score = (ratios[i] - mean) / std if std > 0 else 0

                    if not in_position:
                        # 寻找入场点
                        if abs(z_score) > entry_t:
                            in_position = True
                            entry_price = ratios[i]
                            entry_date = dates[i]
                            entry_zscore = z_score
                    else:
                        # 寻找出场点
                        if abs(z_score) < exit_t:
                            in_position = False
                            exit_price = ratios[i]
                            # 计算这笔交易的收益
                            if entry_zscore > 0:  # 做空入场
                                profit = entry_price - exit_price
                            else:  # 做多入场
                                profit = exit_price - entry_price

                            profit_pct = profit / entry_price

                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': dates[i],
                                'entry_zscore': entry_zscore,
                                'exit_zscore': z_score,
                                'profit': profit_pct
                            })

                # 计算策略绩效
                if trades:
                    avg_profit = np.mean([t['profit'] for t in trades])
                    win_rate = sum(1 for t in trades if t['profit'] > 0) / len(trades)
                    profit_factor = sum(max(0, t['profit']) for t in trades) / abs(
                        sum(min(0, t['profit']) for t in trades)) if sum(
                        min(0, t['profit']) for t in trades) < 0 else float('inf')

                    results.append({
                        'entry_threshold': entry_t,
                        'exit_threshold': exit_t,
                        'avg_profit': avg_profit,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'trade_count': len(trades),
                        'score': avg_profit * win_rate * profit_factor  # 综合得分
                    })

        # 按综合得分排序
        results.sort(key=lambda x: x['score'], reverse=True)

        # 返回最佳结果
        if results:
            best_result = results[0]
            return {
                'optimal_entry_threshold': best_result['entry_threshold'],
                'optimal_exit_threshold': best_result['exit_threshold'],
                'expected_win_rate': best_result['win_rate'],
                'expected_avg_profit': best_result['avg_profit'],
                'profit_factor': best_result['profit_factor'],
                'trade_count': best_result['trade_count'],
                'all_results': results[:5]  # 返回前5个最佳结果
            }
        else:
            return {
                'error': '无法计算最优阈值，数据不足或其他问题'
            }
