"""
仪表盘数据服务
提供仪表盘所需的各类数据
"""
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from user_management.db_models import RecentPairModel, FavoriteModel
from get_stock_data.stock_trends_base import StockTrendsDatabase
from get_stock_data.stock_data_base import StockKlineDatabase
from indicators.investment_signals import analyze_current_position, generate_investment_signals
from k_chart_fetcher import get_stock_data_pair, date_alignment


def get_dashboard_data(user_id: int) -> Dict[str, Any]:
    """
    获取仪表盘数据
    
    参数:
        user_id: 用户ID
        
    返回:
        仪表盘数据字典
    """
    # 获取用户最近查看的资产对
    recent_pairs = RecentPairModel.get_user_recent_pairs(user_id, limit=10)

    # 获取用户收藏的资产
    favorite_assets = FavoriteModel.get_user_favorites(user_id)

    # 处理最近查看的资产对数据
    processed_pairs = []
    for pair in recent_pairs:
        pair_data = process_pair_data(pair['code_a'], pair['name_a'], pair['code_b'], pair['name_b'])
        if pair_data:
            processed_pairs.append(pair_data)

    # 处理收藏资产数据
    processed_favorites = []
    for asset in favorite_assets:
        asset_data = process_asset_data(asset['stock_code'], asset['stock_name'])
        if asset_data:
            processed_favorites.append(asset_data)

    # 构建仪表盘响应数据
    dashboard_data = {
        "recent_pairs": processed_pairs,
        "favorite_assets": processed_favorites,
        "default_pair": processed_pairs[0] if processed_pairs else None
    }

    return dashboard_data


def process_pair_data(code_a: str, name_a: str, code_b: str, name_b: str) -> Optional[Dict[str, Any]]:
    """
    处理资产对数据
    
    参数:
        code_a: 资产A代码
        name_a: 资产A名称
        code_b: 资产B代码
        name_b: 资产B名称
        
    返回:
        处理后的资产对数据
    """
    try:
        # 获取股票历史数据
        close_a, close_b, dates_a, dates_b = get_stock_data_pair(code_a, code_b)
        close_a, close_b, dates = date_alignment(close_a, close_b, dates_a, dates_b)

        if not close_a or not close_b or not dates:
            return None

        # 获取趋势数据
        db = StockTrendsDatabase()
        trends_a_raw = db.query_trends(code_a)
        trends_b_raw = db.query_trends(code_b)

        # 将原始趋势数据转换为统一的字典格式，以便后续处理
        trends_a = []
        trends_b = []

        if trends_a_raw:
            for trend in trends_a_raw:
                if isinstance(trend, dict):
                    trends_a.append({
                        'date': trend.get('date', ''),
                        'current_price': float(trend.get('current_price', 0))
                    })
                else:
                    trends_a.append({
                        'date': trend[1] if len(trend) > 1 else '',
                        'current_price': float(trend[2]) if len(trend) > 2 else 0
                    })

        if trends_b_raw:
            for trend in trends_b_raw:
                if isinstance(trend, dict):
                    trends_b.append({
                        'date': trend.get('date', ''),
                        'current_price': float(trend.get('current_price', 0))
                    })
                else:
                    trends_b.append({
                        'date': trend[1] if len(trend) > 1 else '',
                        'current_price': float(trend[2]) if len(trend) > 2 else 0
                    })

        if not trends_a or not trends_b:
            # 如果没有趋势数据，则使用历史数据
            current_ratio = close_a[-1] / close_b[-1] if close_b[-1] != 0 else 0
            change_ratio = 0
            if len(close_a) > 1 and len(close_b) > 1 and close_b[-2] != 0:
                previous_ratio = close_a[-2] / close_b[-2]
                change_ratio = (current_ratio - previous_ratio) / previous_ratio * 100
        else:
            # 检查并确保趋势数据是按时间排序的
            trends_a.sort(key=lambda x: x['date'])
            trends_b.sort(key=lambda x: x['date'])

            # 使用趋势数据计算当前比值
            current_price_a = trends_a[-1]['current_price'] if trends_a else 0
            current_price_b = trends_b[-1]['current_price'] if trends_b else 0
            current_ratio = current_price_a / current_price_b if current_price_b != 0 else 0

            # 时间对齐并计算变化率
            aligned_trends = []
            time_map_b = {trend['date']: trend['current_price'] for trend in trends_b}

            for trend_a in trends_a:
                time = trend_a['date']
                if time in time_map_b:
                    price_a = trend_a['current_price']
                    price_b = time_map_b[time]
                    if price_b != 0:
                        aligned_trends.append({
                            'time': time,
                            'ratio': price_a / price_b
                        })

            # 计算变化率
            change_ratio = 0
            if aligned_trends and len(aligned_trends) >= 2:
                first_ratio = aligned_trends[0]['ratio']
                last_ratio = aligned_trends[-1]['ratio']
                if first_ratio != 0:
                    change_ratio = (last_ratio - first_ratio) / first_ratio * 100

        # 获取日内趋势数据用于图表显示
        trends_a_data = get_stock_trends_data(code_a)
        trends_b_data = get_stock_trends_data(code_b)

        # 计算投资信号
        signal = generate_investment_signals(close_a, close_b, dates, 3, 1.5)
        signals = []
        if len(close_a) > 30 and len(close_b) > 30:
            position_analysis = analyze_current_position(
                code_a, code_b, close_a, close_b, dates,
                signal, trends_a_data, trends_b_data, 3
            )
            if position_analysis and position_analysis.get('nearest_signals'):
                signals = position_analysis.get('nearest_signals', [])[:3]
                recommendation = position_analysis.get('recommendation', '')
                signals.append({'recommendation': recommendation})

        return {
            "code_a": code_a,
            "name_a": name_a,
            "code_b": code_b,
            "name_b": name_b,
            "current_ratio": current_ratio,
            "change_ratio": change_ratio,
            "trends_a": trends_a_data,
            "trends_b": trends_b_data,
            "signals": signals,
            "latest_date": dates[-1] if dates else None
        }
    except Exception as e:
        print(f"处理资产对数据出错: {e}")
        print(f"错误详情: {type(e).__name__}, {e.args}")
        import traceback
        traceback.print_exc()
        return None


def process_asset_data(code: str, name: str) -> Optional[Dict[str, Any]]:
    """
    处理单个资产数据
    
    参数:
        code: 资产代码
        name: 资产名称
        
    返回:
        处理后的资产数据
    """
    try:
        # 获取K线数据
        db = StockKlineDatabase()
        records = db.query_kline(
            stock_code=code,
            start_date=(datetime.now().strftime('%Y-%m-%d')),
            end_date=(datetime.now().strftime('%Y-%m-%d'))
        )

        if not records:
            # 尝试获取最近一条记录
            records = db.query_kline(
                stock_code=code,
                start_date='2015-01-01',
                end_date=(datetime.now().strftime('%Y-%m-%d'))
            )

        if not records:
            return None

        # 获取当日趋势数据
        trends = get_stock_trends_data(code)

        # 计算价格变化率
        price_change = 0
        current_price = float(records[-1][3]) if records else 0  # close价格

        if trends and len(trends) > 1:
            first_price = trends[0]['current_price']
            last_price = trends[-1]['current_price']
            price_change = (last_price - first_price) / first_price * 100 if first_price != 0 else 0

        return {
            "code": code,
            "name": name,
            "current_price": current_price,
            "price_change": price_change,
            "trends": trends
        }
    except Exception as e:
        print(f"处理资产数据出错: {e}")
        return None


def get_stock_trends_data(stock_code: str) -> List[Dict[str, Any]]:
    """
    获取股票当日趋势数据
    
    参数:
        stock_code: 股票代码
        
    返回:
        趋势数据列表, 格式符合前端的TrendData接口
    """
    try:
        db = StockTrendsDatabase()
        raw_trends = db.query_trends(stock_code)

        if not raw_trends:
            return []

        # 转换为前端所需格式
        result = []

        # 记录第一个价格，用于计算变化百分比
        first_price = None

        # 确保趋势数据是排好序的
        sorted_trends = []
        for trend in raw_trends:
            if isinstance(trend, dict):
                sorted_trends.append({
                    'time': trend.get('date', ''),
                    'price': float(trend.get('current_price', 0))
                })
            else:
                sorted_trends.append({
                    'time': trend[1] if len(trend) > 1 else '',
                    'price': float(trend[2]) if len(trend) > 2 else 0
                })

        # 按时间排序
        sorted_trends.sort(key=lambda x: x['time'])

        # 计算价格变化百分比
        for trend in sorted_trends:
            time = trend['time']
            current_price = trend['price']

            # 记录第一个价格
            if first_price is None:
                first_price = current_price

            # 计算价格变化百分比
            change_percent = 0
            if first_price and first_price != 0:
                change_percent = (current_price - first_price) / first_price * 100

            result.append({
                "time": time,
                "current_price": current_price,
                "change_percent": change_percent
            })

        return result
    except Exception as e:
        print(f"获取股票趋势数据出错: {e}")
        print(f"错误详情: {type(e).__name__}, {e.args}")
        import traceback
        traceback.print_exc()
        return []
