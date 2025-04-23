import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional

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
                - position_size_type: 仓位计算方式 ('fixed'/'percent')
                - position_size: 仓位大小
                - entry_threshold: 入场阈值
                - exit_threshold: 出场阈值
                - stop_loss: 止损比例(%)
                - take_profit: 止盈比例(%)
                - max_positions: 最大持仓数量
                - trading_fee: 交易费率
                
        返回:
            回测结果字典
        """
        try:
            # 获取参数
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

            # 计算价格比值
            df['ratio'] = df['close_a'] / df['close_b']

            # 计算价格比值的滚动均值和标准差（60天窗口）
            window = 60
            df['ratio_ma'] = df['ratio'].rolling(window=window).mean()
            df['ratio_std'] = df['ratio'].rolling(window=window).std()
            df['zscore'] = (df['ratio'] - df['ratio_ma']) / df['ratio_std']

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

            # 开始回测
            for i in range(len(df)):
                current_date = df['date'].iloc[i]
                current_zscore = df['zscore'].iloc[i]
                current_price_a = df['close_a'].iloc[i]
                current_price_b = df['close_b'].iloc[i]

                # 更新当前持仓的价值
                for pos in current_positions:
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

                # 检查止损、止盈和平仓信号
                positions_to_close = []
                for j, pos in enumerate(current_positions):
                    # 检查止损
                    if pos['pnl_percent'] <= -stop_loss:
                        positions_to_close.append((j, 'stop_loss'))
                    # 检查止盈
                    elif pos['pnl_percent'] >= take_profit:
                        positions_to_close.append((j, 'take_profit'))
                    # 检查信号平仓条件
                    elif ((pos['asset'] == code_a and pos[
                        'direction'] == 'long' and current_zscore >= -exit_threshold) or
                          (pos['asset'] == code_a and pos[
                              'direction'] == 'short' and current_zscore <= exit_threshold)):
                        positions_to_close.append((j, 'signal'))

                # 从后往前平仓
                for j, reason in reversed(positions_to_close):
                    pos = current_positions[j]

                    # 计算交易费用
                    fee = pos['position_size'] * trading_fee

                    # 更新资金
                    cash += pos['position_size'] + pos['pnl'] - fee

                    # 记录交易
                    trades.append({
                        'id': trade_id,
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': pos['current_price'],
                        'position_type': f"{pos['direction']}_{pos['asset']}",
                        'position_size': pos['position_size'],
                        'pnl': pos['pnl'] - fee,
                        'pnl_percent': pos['pnl_percent'] - (fee / pos['position_size']),
                        'status': 'closed',
                        'exit_reason': reason
                    })
                    trade_id += 1

                    # 移除平仓的头寸
                    current_positions.pop(j)

                # 检查开仓信号
                if len(current_positions) < max_positions:
                    # 计算每个头寸的大小
                    if position_size_type == 'fixed':
                        position_amount = min(position_size, cash)
                    else:  # percent
                        position_amount = min(cash * (position_size / 100), cash)

                    # 如果有足够资金，检查开仓信号
                    if position_amount > 0 and cash >= position_amount:
                        # 价差高于阈值，做空资产A
                        if current_zscore > entry_threshold:
                            # 扣除费用
                            fee = position_amount * trading_fee
                            actual_position = position_amount - fee

                            # 添加头寸
                            current_positions.append({
                                'asset': code_a,
                                'direction': 'short',
                                'entry_date': current_date,
                                'entry_price': current_price_a,
                                'current_price': current_price_a,
                                'position_size': actual_position,
                                'pnl': -fee,
                                'pnl_percent': -fee / actual_position
                            })

                            # 更新现金
                            cash -= position_amount

                        # 价差低于阈值，做多资产A
                        elif current_zscore < -entry_threshold:
                            # 扣除费用
                            fee = position_amount * trading_fee
                            actual_position = position_amount - fee

                            # 添加头寸
                            current_positions.append({
                                'asset': code_a,
                                'direction': 'long',
                                'entry_date': current_date,
                                'entry_price': current_price_a,
                                'current_price': current_price_a,
                                'position_size': actual_position,
                                'pnl': -fee,
                                'pnl_percent': -fee / actual_position
                            })

                            # 更新现金
                            cash -= position_amount

                # 计算总权益
                positions_value = sum(pos['position_size'] + pos['pnl'] for pos in current_positions)
                equity = cash + positions_value

                # 更新最大权益（用于计算回撤）
                max_equity = max(max_equity, equity)
                drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0

                # 记录权益曲线
                equity_curve.append({
                    'date': current_date,
                    'equity': equity,
                    'drawdown': -drawdown  # 负值表示回撤
                })

            # 回测结束，平掉所有未平仓的头寸
            for pos in current_positions:
                # 计算交易费用
                fee = pos['position_size'] * trading_fee

                # 记录交易
                trades.append({
                    'id': trade_id,
                    'entry_date': pos['entry_date'],
                    'exit_date': df['date'].iloc[-1],
                    'entry_price': pos['entry_price'],
                    'exit_price': pos['current_price'],
                    'position_type': f"{pos['direction']}_{pos['asset']}",
                    'position_size': pos['position_size'],
                    'pnl': pos['pnl'] - fee,
                    'pnl_percent': pos['pnl_percent'] - (fee / pos['position_size']),
                    'status': 'closed',
                    'exit_reason': 'end_of_backtest'
                })
                trade_id += 1

                # 更新资金
                cash += pos['position_size'] + pos['pnl'] - fee

            # 计算最终权益
            final_equity = cash

            # 计算回测指标
            total_return = (final_equity - initial_capital) / initial_capital

            # 计算年化收益率
            start_dt = datetime.strptime(df['date'].iloc[0], '%Y-%m-%d')
            end_dt = datetime.strptime(df['date'].iloc[-1], '%Y-%m-%d')
            days = (end_dt - start_dt).days
            annual_return = ((1 + total_return) ** (365 / days)) - 1 if days > 0 else 0

            # 计算最大回撤
            equity_array = np.array([ec['equity'] for ec in equity_curve])
            max_drawdown = min([ec['drawdown'] for ec in equity_curve])

            # 计算夏普比率
            if len(equity_array) > 1:
                daily_returns = np.diff(equity_array) / equity_array[:-1]
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(
                    daily_returns) > 0 else 0
            else:
                sharpe_ratio = 0

            # 计算胜率和盈亏比
            win_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(win_trades) / len(trades) if trades else 0

            # 计算盈亏因子
            total_profit = sum(t['pnl'] for t in win_trades) if win_trades else 0
            loss_trades = [t for t in trades if t['pnl'] <= 0]
            total_loss = abs(sum(t['pnl'] for t in loss_trades)) if loss_trades else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # 构建回测结果
            backtest_result = {
                'equity_curve': equity_curve,
                'trades': trades,
                'metrics': {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': abs(max_drawdown),
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'total_trades': len(trades)
                }
            }

            return backtest_result

        except Exception as e:
            import traceback
            print(f"回测执行过程中发生错误: {str(e)}")
            traceback.print_exc()
            return {"error": f"回测执行过程中发生错误: {str(e)}"}
