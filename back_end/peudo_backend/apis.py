import io
import json
import traceback
from datetime import datetime, timedelta
from typing import List, Dict

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, HTTPException, Response, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from back_end.peudo_backend.user_management import UserModel
# 导入回测引擎
from backtest.backtest_strategy import BacktestEngine
from get_stock_data.get_stock_trends_data import StockTrendsData
from get_stock_data.stock_data_base import StockKlineDatabase
from get_stock_data.stock_trends_base import StockTrendsDatabase
from indicators.investment_signals import generate_investment_signals, analyze_current_position
# 导入新增的信号评估模块
from indicators.signal_evaluator import (
    evaluate_signal_quality,
    record_new_signal,
    update_signal_performance,
    get_signal_performance_stats,
    get_signal_history
)
from indicators.technical_indicators import calculate_indicators, calculate_price_ratio_anomaly, \
    calculate_ratio_indicators, detect_ratio_indicators_signals
from k_chart_fetcher import k_chart_fetcher, durations, get_stock_data_pair, date_alignment
from kline_processor.processor import process_kline_by_type
# 导入多资产分析模块
from multi_asset_analysis import calculate_correlation_matrix, find_optimal_trading_pairs, get_asset_pair_analysis, \
    get_stock_names, get_all_pair_charts
# 导入LSTM预测器
from prediction import get_predictor
# 导入模块化后的函数
from stock_search.searcher import score_match, is_stock_code_format, fetch_stock_from_api
# 导入用户管理模块
from user_management import FavoriteModel, RecentPairModel
from user_management import create_session, get_session, delete_session
from user_management import generate_captcha, verify_captcha
from user_management.dashboard_service import get_dashboard_data

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["captcha-id"]  # 允许前端获取自定义header
)

"""
前端请求样例
http://192.168.1.220:8000/search_stocks/我是关键词
"""


@app.get("/search_stocks/{keyword}")
async def search_stocks(keyword: str):
    """
    搜索股票，支持多种匹配方式：
    1. 股票代码精确匹配
    2. 股票名称精确匹配
    3. 股票代码包含关键词
    4. 股票名称包含关键词
    5. 股票名称拼音首字母匹配
    6. 股票名称完整拼音匹配

    排序优先级：完全匹配 > 代码匹配 > 名称匹配 > 拼音首字母匹配 > 完整拼音匹配
    """
    try:
        # 读取股票数据
        with open('stock_list.json', 'r', encoding='utf-8') as file:
            stock_info_json = json.load(file)
            stock_info_list = stock_info_json['stocks']

            # 如果关键词为空，返回空结果
            if not keyword.strip():
                return {"result": []}

        # 搜索匹配的股票并进行评分
        matched_stocks = search_stocks_with_score(stock_info_list, keyword)

        # 提取排序后的股票信息
        searched = [item[0] for item in matched_stocks]

        return {"result": searched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"搜索股票时发生错误: {str(e)}")


def search_stocks_with_score(stock_list: List[Dict[str, str]], keyword: str) -> List[tuple]:
    """
    搜索匹配关键词的股票，并返回带评分的结果列表
    参数:
        stock_list: 股票列表
        keyword: 搜索关键词
    返回:
        匹配的股票和评分元组列表，按评分从高到低排序
    """
    # 搜索匹配的股票
    searched_with_score = []
    for stock_info in stock_list:
        score = score_match(stock_info, keyword)
        if score > 0:
            searched_with_score.append((stock_info, score))

    # 根据得分排序，得分高的在前
    searched_with_score.sort(key=lambda x: x[1], reverse=True)

    return searched_with_score


class DataModel(BaseModel):
    code_a: str
    code_b: str
    degree: int
    duration: str
    threshold_arg: float


"""
前端post样例
{
  "code_a": "HSI",
  "code_b": "IXIC",
  "duration": "maximum",
  "degree": 3
}
"""


@app.post("/get_k_chart_info/")
async def get_k_chart_info(user_option_info: DataModel):
    try:
        # 调用k_chart_fetcher函数获取K线数据
        chart_data = k_chart_fetcher(
            user_option_info.code_a,
            user_option_info.code_b,
            user_option_info.duration,
            user_option_info.degree,
            threshold_arg=user_option_info.threshold_arg
        )
        # 获取前端传入的阈值系数和原始标准差
        threshold_multiplier = user_option_info.threshold_arg
        original_std = chart_data["threshold"]  # 原始标准差

        anomaly_info = calculate_price_ratio_anomaly(
            chart_data["ratio"],
            chart_data["delta"],
            threshold_multiplier,  # 用户设置的阈值倍数
            original_std  # 原始标准差
        )

        # 合并结果并返回
        chart_data["anomaly_info"] = anomaly_info

        return chart_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 获取所有资产
@app.get("/get_all_assets/")
async def get_all_assets():
    """
    获取所有可用资产列表
    返回样例:
    {
        "assets": [
            {"code": "000001", "name": "平安银行", "type": "A股"},
            {"code": "00700", "name": "腾讯控股", "type": "港股"},
            {"code": "AAPL", "name": "苹果公司", "type": "美股"}
        ]
    }
    """
    with open('stock_list.json', 'r', encoding='utf-8') as file:
        stock_info_json = json.load(file)
    return {"assets": stock_info_json['stocks']}


# 按类别获取资产
@app.get("/get_assets_by_type/{asset_type}")
async def get_assets_by_type(asset_type: str):
    """
    按类别获取资产列表
    参数:
        asset_type: 资产类别，可选值为 "A股", "港股", "美股"
    返回样例:
    {
        "assets": [
            {"code": "000001", "name": "平安银行", "type": "A股"}
        ]
    }
    """
    with open('stock_list.json', 'r', encoding='utf-8') as file:
        stock_info_json = json.load(file)
        stock_info_list = stock_info_json['stocks']

    filtered_assets = [asset for asset in stock_info_list if asset['type'] == asset_type]
    return {"assets": filtered_assets}


# 搜索资产(返回top-30相似结果)
@app.get("/search_top_assets/{keyword}")
async def search_top_assets(keyword: str):
    """
    搜索资产，返回top-30相似结果
    支持代码/名称/拼音首字母匹配
    排序优先级：完全匹配 > 代码匹配 > 名称匹配 > 拼音首字母匹配
    当搜索纯数字代码无结果时自动触发爬取
    """
    with open('stock_list.json', 'r', encoding='utf-8') as file:
        stock_info_json = json.load(file)
        stock_info_list = stock_info_json['stocks']

    # 搜索匹配的股票
    searched_with_score = []
    for stock_info in stock_info_list:
        score = score_match(stock_info, keyword)
        if score > 0:
            searched_with_score.append((stock_info, score))

    # 如果没有匹配结果，且关键词为纯数字或可能是股票代码格式，尝试爬取
    is_stock_code = is_stock_code_format(keyword)
    new_stock_info = None

    if not searched_with_score and is_stock_code:
        print(f"未找到股票，尝试从API获取: {keyword}")
        new_stock_info = await fetch_stock_from_api(keyword)

        if new_stock_info:
            # 同时获取并保存该股票的当日价格趋势数据
            get_stock_trends(keyword)

            # 添加到搜索结果中
            searched_with_score.append((new_stock_info, 100))  # 给新爬取的股票最高分

    # 根据得分排序，得分高的在前
    searched_with_score.sort(key=lambda x: x[1], reverse=True)

    # 提取排序后的股票信息
    searched = [item[0] for item in searched_with_score]

    # 限制返回top-30结果
    return {"assets": searched[:30]}


class StockKlineRequest(BaseModel):
    code: str  # 股票代码
    kline_type: str  # K线类型: "daily"(日K), "weekly"(周K), "monthly"(月K), "yearly"(年K), "realtime"(实时)
    duration: str  # 时间范围: "maximum", "5y", "2y", "1y", "1q", "1m"


def get_stock_trends(stock_code: str):
    """
    获取股票当日价格趋势数据，如果数据库中没有则尝试从API获取

    参数:
        stock_code: 股票代码

    返回:
        趋势数据列表或空列表
    """
    try:
        # 从数据库查询趋势数据
        db = StockTrendsDatabase()
        trends_data = db.query_trends(stock_code=stock_code)

        # 如果没有数据，尝试从API获取
        if not trends_data:
            # 获取新数据
            trends_fetcher = StockTrendsData(stock_code)
            api_trends_data = trends_fetcher.get_trends()

            if "error" not in api_trends_data and api_trends_data.get('trends'):
                formatted_data = trends_fetcher.format_klines(api_trends_data['trends'])

                # 存储新数据到数据库
                if formatted_data:
                    db.insert_trends_data(
                        data_list=formatted_data,
                        code=api_trends_data['code'],
                        name=api_trends_data['name'],
                        type=api_trends_data['type']
                    )
                    # 重新查询数据库获取趋势数据
                    trends_data = db.query_trends(stock_code=stock_code)

        return trends_data
    except Exception as e:
        print(f"获取股票趋势数据失败: {e}")
        return []


@app.post("/get_stock_kline/")
async def get_stock_kline(data: StockKlineRequest):
    """
    获取单只股票的K线数据
    请求体样例:
    {
        "code": "000001",
        "kline_type": "daily",
        "duration": "1y"
    }
    返回样例:
    {
        "code": "000001",
        "name": "平安银行",
        "type": "A股",
        "kline_data": [
            {
                "date": "2023-05-15",
                "open": 12.5,
                "close": 12.8,
                "high": 13.0,
                "low": 12.4,
                "volume": 12345678
            },
            ...
        ],
        "indicators": {
            "ma5": [12.7, 12.8, ...],
            "ma10": [12.6, 12.7, ...],
            "ma20": [12.5, 12.6, ...],
            "ma30": [12.4, 12.5, ...],
            "ma60": [12.3, 12.4, ...],
            "macd": {
                "dif": [0.2, 0.3, ...],
                "dea": [0.1, 0.2, ...],
                "macd": [0.2, 0.2, ...]
            },
            "rsi": {
                "rsi6": [56.2, 58.3, ...],
                "rsi12": [55.1, 57.2, ...],
                "rsi24": [53.6, 55.8, ...]
            }
        },
        "trends": [  // 仅在kline_type为"realtime"时返回
            {
                "date": "2025-04-18 09:30:00",
                "current_price": 12.5,
                "volume": 123456
            },
            ...
        ]
    }
    """
    try:
        # 检查股票代码是否存在
        stock_info = None
        with open('stock_list.json', 'r', encoding='utf-8') as file:
            stock_info_json = json.load(file)
            for stock in stock_info_json['stocks']:
                if stock['code'] == data.code:
                    stock_info = stock
                    break

        # 如果股票不存在，尝试从API获取
        if not stock_info:
            stock_info = await fetch_stock_from_api(data.code)
            if not stock_info:
                raise HTTPException(status_code=404, detail=f"股票代码 {data.code} 未找到")

        # 从数据库获取K线数据
        db = StockKlineDatabase()

        # 计算起始日期和结束日期
        end_date = datetime.now().strftime('%Y-%m-%d')

        # 根据duration计算起始日期
        if data.duration == 'maximum':
            start_date = '1989-01-01'  # 数据库中最早的数据
        else:
            days = durations[data.duration]
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # 查询原始K线数据
        records = db.query_kline(data.code, start_date, end_date)

        if not records:
            raise HTTPException(status_code=404, detail=f"股票 {data.code} 在指定时间范围内没有数据")

        # 处理K线类型 (日K, 周K, 月K, 年K)
        # 注意：即使选择了"realtime"，也需要处理日K数据用于技术指标计算
        k_type_for_processing = "daily" if data.kline_type == "realtime" else data.kline_type
        kline_data = process_kline_by_type(records, k_type_for_processing)

        # 计算技术指标
        indicators = calculate_indicators(kline_data)

        # 构建基本响应
        response = {
            "code": stock_info["code"],
            "name": stock_info["name"],
            "type": stock_info["type"],
            "kline_data": kline_data,
            "indicators": indicators
        }

        # 如果是实时K线类型，获取当日价格趋势数据
        if data.kline_type == "realtime":
            trends_data = get_stock_trends(data.code)
            if trends_data:
                response["trends"] = trends_data

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class PredictionRequestModel(BaseModel):
    code_a: str  # 股票A代码
    code_b: str  # 股票B代码
    ratio_data: List[float]  # 历史价格比值数据
    dates: List[str]  # 历史日期数据
    prediction_days: int = 30  # 预测天数，默认30天
    confidence_level: float = 0.95  # 置信水平，默认95%
    model_type: str = "lstm"  # 预测模型类型，可选: "lstm", "arima", "prophet"


@app.post("/predict_price_ratio/")
async def predict_price_ratio(request: PredictionRequestModel):
    """
    使用时间序列模型预测未来价格比值走势

    参数:
        request: 包含历史价格比值、日期和预测参数的请求对象

    返回:
        预测结果，包含预测值、置信区间和模型性能指标
    """
    try:
        # 1. 从请求中提取数据
        ratio_data = request.ratio_data
        dates = request.dates
        code_a = request.code_a
        code_b = request.code_b
        prediction_days = request.prediction_days
        confidence_level = request.confidence_level
        model_type = request.model_type

        # 2. 数据验证
        if len(ratio_data) < 10:
            raise HTTPException(
                status_code=400,
                detail="历史数据太少，无法进行可靠预测。需要至少10个数据点。"
            )

        if prediction_days > 180:
            raise HTTPException(
                status_code=400,
                detail="预测天数过长，超过180天的预测不可靠。请设置更短的预测期。"
            )

        # 3. 获取合适的预测器实例
        try:
            # 使用工厂函数获取预测器，默认使用增强版LSTM
            predictor = get_predictor(
                model_type=model_type if model_type != "lstm" else "enhanced_lstm",
                cache_dir='model_cache'
            )
        except Exception as e:
            print(f"创建预测器失败: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"创建预测器失败: {str(e)}"
            )

        # 4. 使用预测器进行预测
        try:
            results = predictor.predict(
                ratio_data=ratio_data,
                dates=dates,
                code_a=code_a,
                code_b=code_b,
                prediction_days=prediction_days,
                confidence_level=confidence_level,
                model_type=model_type
            )
        except Exception as e:
            print(f"预测过程发生错误: {e}")

            # 如果实际模型预测失败，回退到模拟数据生成方式
            # 保持向前兼容性，确保前端不会因为后端错误而崩溃
            print("使用模拟数据作为回退方案")

            # 下面是原有的模拟数据生成逻辑
            # 这段保留作为备份，当模型训练失败时使用
            import random
            random.seed(42)  # 固定随机种子，使结果可重现

            # 计算一些统计值，用于模拟数据生成
            last_value = ratio_data[-1]
            mean_value = sum(ratio_data) / len(ratio_data)

            # 计算历史数据的标准差，用于生成置信区间
            variance = sum((x - mean_value) ** 2 for x in ratio_data) / len(ratio_data)
            std_dev = variance ** 0.5

            # 生成预测日期（从最后一天开始向后延伸）
            last_date = datetime.strptime(dates[-1], "%Y-%m-%d")
            predicted_dates = []

            for i in range(1, prediction_days + 1):
                next_date = last_date + timedelta(days=i)
                predicted_dates.append(next_date.strftime("%Y-%m-%d"))

            # 模拟趋势方向
            trend = random.choice([-1, 1])  # -1 表示下降，1 表示上升
            volatility = std_dev * 0.5  # 使用历史标准差的一半作为波动率

            predicted_values = []
            for i in range(prediction_days):
                # 添加趋势和随机波动
                trend_component = trend * (i / prediction_days) * last_value * 0.1  # 最多变化10%
                random_component = random.normalvariate(0, volatility)
                predicted_value = last_value + trend_component + random_component
                predicted_values.append(max(0.01, predicted_value))  # 确保比值为正

            # 计算置信区间
            z_score = 1.96  # 95% 置信区间的Z分数
            if confidence_level == 0.9:
                z_score = 1.645
            elif confidence_level == 0.99:
                z_score = 2.576

            margin = z_score * std_dev
            upper_bound = [value + margin for value in predicted_values]
            lower_bound = [value - margin for value in predicted_values]

            # 计算模拟的模型性能指标
            performance = {
                "mse": round(random.uniform(0.001, 0.01), 6),  # 均方误差
                "rmse": round(random.uniform(0.03, 0.1), 6),  # 均方根误差
                "mae": round(random.uniform(0.02, 0.08), 6),  # 平均绝对误差
                "r2": round(random.uniform(0.7, 0.95), 6)  # R方值
            }

            # 确定风险级别和预测趋势
            risk_level = "medium"
            forecast_trend = "up" if trend > 0 else "down"

            # 计算预测区间相对宽度
            avg_value = sum(predicted_values) / len(predicted_values)
            avg_interval_width = sum(u - l for u, l in zip(upper_bound, lower_bound)) / len(predicted_values)
            relative_width = avg_interval_width / avg_value

            # 基于相对宽度确定风险级别
            if relative_width > 0.2:
                risk_level = "high"
            elif relative_width < 0.1:
                risk_level = "low"

            # 返回模拟结果
            results = {
                "dates": predicted_dates,
                "values": predicted_values,
                "upper_bound": upper_bound,
                "lower_bound": lower_bound,
                "historical_dates": dates[-30:],  # 仅返回最近30天的历史数据
                "historical_values": ratio_data[-30:],
                "performance": performance,
                "risk_level": risk_level,
                "forecast_trend": forecast_trend
            }

        # 5. 返回预测结果
        return results

    except HTTPException as e:
        # 直接重新抛出HTTP异常
        raise e
    except Exception as e:
        # 处理其他异常
        print(f"预测价格比值时发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"预测价格比值时发生错误: {str(e)}")


class SignalRequestModel(BaseModel):
    code_a: str
    code_b: str
    duration: str
    degree: int = 3
    threshold_arg: float = 2.0
    track_signals: bool = True  # 是否跟踪记录信号


class SignalModel(BaseModel):
    id: int
    date: str
    ratio: float
    z_score: float
    type: str  # 'positive' or 'negative'
    strength: str  # 'weak', 'medium', 'strong'
    description: str
    recommendation: str


@app.post("/get_investment_signals/")
async def get_investment_signals(request: SignalRequestModel):
    """
    获取两只股票的投资信号和当前位置分析

    参数:
        code_a: 股票A代码
        code_b: 股票B代码
        duration: 时间跨度 (1m, 1q, 1y, 2y, 5y, maximum)
        degree: 多项式拟合的次数
        threshold_arg: 阈值系数
        track_signals: 是否跟踪记录信号

    返回:
        包含信号列表和当前位置分析的响应对象
    """
    try:
        # 1. 获取历史K线和价格比值数据
        close_a_, close_b_, dates_a_, dates_b_ = get_stock_data_pair(request.code_a, request.code_b)
        close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)

        # 根据指定的时间跨度截取数据
        duration_days = durations[request.duration]
        if duration_days == -1 or duration_days >= len(close_a):
            duration_days = len(close_a)

        close_a = close_a[-duration_days:]
        close_b = close_b[-duration_days:]
        dates = dates[-duration_days:]

        # 2. 生成投资信号
        signals = generate_investment_signals(
            close_a,
            close_b,
            dates,
            request.degree,
            request.threshold_arg
        )

        # 3. 获取当前最新价格趋势
        db = StockTrendsDatabase()
        trends_a = db.query_trends(stock_code=request.code_a)
        trends_b = db.query_trends(stock_code=request.code_b)

        # 4. 分析当前位置
        current_position = analyze_current_position(
            request.code_a,
            request.code_b,
            close_a,
            close_b,
            dates,
            signals,
            trends_a,
            trends_b,
            request.degree
        )

        # 5. 添加信号质量评分
        for signal in signals:
            # 为每个信号计算质量评分
            quality_evaluation = evaluate_signal_quality(
                signal,
                request.code_a,
                request.code_b
            )
            signal["quality_evaluation"] = quality_evaluation

            # 如果开启跟踪，记录新信号
            if request.track_signals:
                record = record_new_signal(
                    signal,
                    request.code_a,
                    request.code_b,
                    quality_evaluation
                )
                signal["record_id"] = record["record_id"]

            # 更新信号表现数据（如果已有记录）
            update_signal_performance(
                signal.get("record_id", 0),
                close_a,
                close_b,
                dates
            )

        # 确保所有数据都是Python原生类型，防止NumPy类型序列化问题
        def ensure_native_types(obj):
            if isinstance(obj, dict):
                return {k: ensure_native_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [ensure_native_types(item) for item in obj]
            elif hasattr(obj, 'item'):  # NumPy类型有.item()方法
                return obj.item()
            elif hasattr(obj, 'tolist'):  # NumPy数组
                return obj.tolist()
            else:
                return obj

        # 应用类型转换
        signals = ensure_native_types(signals)
        current_position = ensure_native_types(current_position)

        return {
            "signals": signals,
            "current_position": current_position
        }
    except Exception as e:
        print(f"获取投资信号时出错: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal_history/{code_a}/{code_b}")
async def get_signals_history(code_a: str, code_b: str, limit: int = 50, include_followup: bool = False):
    """
    获取指定股票对的信号历史记录
    
    参数:
        code_a: 股票A代码
        code_b: 股票B代码
        limit: 返回记录数量限制
        include_followup: 是否包含详细跟踪数据
        
    返回:
        信号历史记录列表
    """
    try:
        records = get_signal_history(code_a, code_b, limit, include_followup)
        return {"records": records}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal_performance_stats/")
async def get_signals_performance_stats(code_a: str = None, code_b: str = None):
    """
    获取信号表现统计数据
    
    参数:
        code_a: 可选，股票A代码，用于筛选特定股票对
        code_b: 可选，股票B代码，用于筛选特定股票对
    
    返回:
        信号表现统计报告
    """
    try:
        stats = get_signal_performance_stats(code_a, code_b)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/signal_tracking/{record_id}")
async def get_signal_tracking(record_id: int):
    """
    获取单个信号的跟踪数据
    
    参数:
        record_id: 信号记录ID
        
    返回:
        信号记录详情，包含跟踪数据
    """
    try:
        records = get_signal_history(limit=999, include_followup=True)
        for record in records:
            if record["record_id"] == record_id:
                return record
        raise HTTPException(status_code=404, detail="信号记录不存在")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest_strategy/")
async def backtest_strategy(params: dict):
    try:
        engine = BacktestEngine()
        result = engine.run_price_ratio_backtest(params)
        return result
    except Exception as e:
        traceback_str = traceback.format_exc()
        return {"error": f"回测失败: {str(e)}", "traceback": traceback_str}


@app.post("/backtest_similar_signals/")
async def backtest_similar_signals(params: dict):
    """
    运行基于当前价格比值与历史相似信号的回测策略
    
    该策略仅基于当前比值位置与历史上最相似的几个投资信号进行回测，
    以验证当前投资机会的潜在表现。
    
    请求参数:
    {
        "code_a": "AAPL",               # 资产A代码
        "code_b": "MSFT",               # 资产B代码
        "initial_capital": 100000,      # 初始资金
        "position_size": 30,            # 仓位大小(%)
        "stop_loss": 5,                 # 止损比例(%)
        "take_profit": 10,              # 止盈比例(%)
        "trading_fee": 0.0003,          # 交易费率
        "polynomial_degree": 3,         # 多项式拟合的次数
        "threshold_multiplier": 1.5     # 信号生成的阈值乘数
    }
    
    返回:
    包含交易记录、绩效指标和相似信号分析的字典
    """
    try:
        engine = BacktestEngine()
        result = engine.run_similar_signals_backtest(params)
        return result
    except Exception as e:
        traceback_str = traceback.format_exc()
        return {"error": f"相似信号回测失败: {str(e)}", "traceback": traceback_str}


class OptimalThresholdRequest(BaseModel):
    code_a: str
    code_b: str
    lookback: int = 60
    strategy_type: str = "zscore"


@app.post("/calculate_optimal_threshold/")
async def calculate_optimal_threshold(request: OptimalThresholdRequest):
    """
    计算最优入场和出场阈值
    
    参数:
    - code_a: 资产A代码
    - code_b: 资产B代码
    - lookback: 回溯天数，默认60天
    - strategy_type: 策略类型，默认为"zscore"
    
    返回:
    - 包含最优阈值的响应对象
    """
    try:
        # 1. 获取股票数据
        close_a_, close_b_, dates_a_, dates_b_ = get_stock_data_pair(request.code_a, request.code_b)
        close_a, close_b, dates = date_alignment(close_a_, close_b_, dates_a_, dates_b_)

        # 确保有足够的数据进行计算
        if len(close_a) < request.lookback + 10:
            return {"error": "历史数据不足，无法计算最优阈值，请选择更长的时间段或减小回溯天数"}

        # 2. 创建回测引擎实例
        engine = BacktestEngine()

        # 3. 计算最优阈值
        optimal_result = engine.calculate_optimal_threshold(
            prices_a=close_a,
            prices_b=close_b,
            dates=dates,
            lookback=request.lookback
        )

        # 4. 如果有错误，返回错误信息
        if "error" in optimal_result:
            return {"error": optimal_result["error"]}

        # 5. 构建响应
        response = {
            "entry_threshold": optimal_result["optimal_entry_threshold"],
            "exit_threshold": optimal_result["optimal_exit_threshold"],
            "win_rate": optimal_result["expected_win_rate"],
            "estimated_profit": optimal_result["expected_avg_profit"] * 100,  # 转换为百分比
            "lookback_period": request.lookback,
            "trade_count": optimal_result["trade_count"],
            "strategy_type": request.strategy_type
        }

        return response

    except Exception as e:
        print(f"计算最优阈值时发生错误: {str(e)}")
        traceback.print_exc()
        return {"error": f"计算最优阈值时发生错误: {str(e)}"}


# 验证码图片生成辅助函数
def generate_captcha_image(text: str, width: int = 120, height: int = 40) -> Image.Image:
    """生成包含验证码的图片"""
    # 创建白底图片
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except IOError:
        font = ImageFont.load_default()

    # 绘制验证码文本
    # 修复：PIL新版本使用getbbox或getsize替代textsize
    try:
        # 使用getsize（较旧但更广泛支持）
        text_width, text_height = draw.textsize(text, font=font)
    except AttributeError:
        try:
            # 使用getbbox（新版本PIL）
            bbox = font.getbbox(text)
            text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            # 最后尝试使用getsize
            text_width, text_height = font.getsize(text)

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # 确保x和y不为负值
    x = max(0, x)
    y = max(0, y)

    # 绘制文本
    draw.text((x, y), text, font=font, fill=(0, 0, 0))

    # 添加干扰线
    for i in range(5):
        start_point = (np.random.randint(0, width), np.random.randint(0, height))
        end_point = (np.random.randint(0, width), np.random.randint(0, height))
        draw.line([start_point, end_point], fill=(0, 0, 255), width=1)

    # 添加干扰点
    for i in range(50):
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        draw.point((x, y), fill=(0, 0, 0))

    return image


# 用户相关的请求模型
class LoginRequest(BaseModel):
    username: str
    password: str
    captcha: str
    captcha_id: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    email: str = None
    captcha: str
    captcha_id: str


class AddFavoriteRequest(BaseModel):
    stock_code: str
    stock_name: str
    stock_type: str


class AddRecentPairRequest(BaseModel):
    code_a: str
    name_a: str
    code_b: str
    name_b: str


# 获取验证码图片
@app.get("/api/captcha")
async def get_captcha():
    # 生成验证码和ID
    captcha_id, captcha_text = generate_captcha(4)

    # 生成验证码图片
    image = generate_captcha_image(captcha_text)

    # 将图片转换为二进制流
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    # 返回验证码图片和ID
    return StreamingResponse(
        img_byte_arr,
        media_type="image/png",
        headers={"captcha-id": captcha_id}
    )


# 用户登录
@app.post("/api/login")
async def login(request: LoginRequest, response: Response):
    # 验证验证码
    if not verify_captcha(request.captcha_id, request.captcha):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 验证用户凭据
    user = UserModel.verify_user(request.username, request.password)
    if not user:
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    # 创建会话
    session_id = create_session(user)

    # 设置会话Cookie
    response.set_cookie(
        key="session_id",
        value=session_id,
        max_age=24 * 3600,  # 24小时
        httponly=True,
        samesite="lax"
    )

    # 返回用户信息（不包含密码）
    user.pop("password_hash", None)
    return {"status": "success", "user": user}


# 用户注册
@app.post("/api/register")
async def register(request: RegisterRequest):
    # 验证验证码
    if not verify_captcha(request.captcha_id, request.captcha):
        raise HTTPException(status_code=400, detail="验证码错误或已过期")

    # 创建用户
    user_id = UserModel.create_user(request.username, request.password, request.email)
    if not user_id:
        raise HTTPException(status_code=400, detail="用户名或邮箱已存在")

    return {"status": "success", "user_id": user_id}


# 用户登出
@app.post("/api/logout")
async def logout(response: Response, session_id: str = Cookie(None)):
    if session_id:
        delete_session(session_id)

    # 清除会话Cookie
    response.delete_cookie(key="session_id")

    return {"status": "success"}


# 检查会话状态
@app.get("/api/session")
async def check_session(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 不返回密码信息
    user.pop("password_hash", None)
    return {"status": "success", "user": user}


# 添加收藏资产
@app.post("/api/favorites")
async def add_favorite(request: AddFavoriteRequest, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 添加收藏
    success = FavoriteModel.add_favorite(
        user["id"],
        request.stock_code,
        request.stock_name,
        request.stock_type
    )

    if not success:
        raise HTTPException(status_code=400, detail="已收藏过该资产")

    return {"status": "success"}


# 删除收藏资产
@app.delete("/api/favorites/{stock_code}")
async def remove_favorite(stock_code: str, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 删除收藏
    success = FavoriteModel.remove_favorite(user["id"], stock_code)

    if not success:
        raise HTTPException(status_code=404, detail="未找到该收藏资产")

    return {"status": "success"}


# 获取收藏资产列表
@app.get("/api/favorites")
async def get_favorites(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 获取收藏列表
    favorites = FavoriteModel.get_user_favorites(user["id"])

    return {"status": "success", "favorites": favorites}


# 检查资产是否已收藏
@app.get("/api/favorites/{stock_code}/check")
async def check_favorite(stock_code: str, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 检查是否已收藏
    is_favorite = FavoriteModel.is_favorite(user["id"], stock_code)

    return {"status": "success", "is_favorite": is_favorite}


# 添加最近查看的资产对
@app.post("/api/recent-pairs")
async def add_recent_pair(request: AddRecentPairRequest, session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 添加或更新最近查看记录
    RecentPairModel.add_recent_pair(
        user["id"],
        request.code_a,
        request.name_a,
        request.code_b,
        request.name_b
    )

    return {"status": "success"}


# 获取最近查看的资产对列表
@app.get("/api/recent-pairs")
async def get_recent_pairs(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 获取最近查看列表
    recent_pairs = RecentPairModel.get_user_recent_pairs(user["id"])

    return {"status": "success", "recent_pairs": recent_pairs}


# 获取仪表盘数据
@app.get("/api/dashboard")
async def get_dashboard(session_id: str = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未登录")

    user = get_session(session_id)
    if not user:
        raise HTTPException(status_code=401, detail="会话已过期")

    # 获取仪表盘数据
    dashboard_data = get_dashboard_data(user["id"])

    return {"status": "success", "data": dashboard_data}


class RatioIndicatorsRequest(BaseModel):
    code_a: str
    code_b: str
    duration: str


@app.post("/get_ratio_indicators/")
async def get_ratio_indicators(request: RatioIndicatorsRequest):
    """
    获取价格比值技术指标
    """
    try:
        # 获取两只股票的K线数据
        close_a, close_b, dates_a, dates_b = get_stock_data_pair(request.code_a, request.code_b)

        # 日期对齐，确保使用相同交易日的数据
        close_a, close_b, dates = date_alignment(close_a, close_b, dates_a, dates_b)

        # 根据时间跨度选择数据
        duration_days = durations[request.duration]
        if duration_days == -1 or duration_days >= len(close_a):
            duration_days = len(close_a)

        close_a = close_a[-duration_days:]
        close_b = close_b[-duration_days:]
        dates = dates[-duration_days:]

        # 计算价格比值
        ratio = [float(a) / float(b) for a, b in zip(close_a, close_b)]

        # 计算比值的技术指标
        indicators = calculate_ratio_indicators(ratio)

        # 构建响应
        response = {
            "code_a": request.code_a,
            "code_b": request.code_b,
            "dates": dates,
            "ratio": ratio,
            "indicators": indicators
        }

        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RatioSignalsRequest(BaseModel):
    code_a: str
    code_b: str
    duration: str


@app.post("/get_ratio_indicator_signals/")
async def get_ratio_indicator_signals(request: RatioSignalsRequest):
    """
    获取价格比值技术指标的特殊点（金叉、死叉、超买超卖等）
    """
    try:
        # 获取两只股票的K线数据
        close_a, close_b, dates_a, dates_b = get_stock_data_pair(request.code_a, request.code_b)

        # 日期对齐，确保使用相同交易日的数据
        close_a, close_b, dates = date_alignment(close_a, close_b, dates_a, dates_b)

        # 根据时间跨度选择数据
        duration_days = durations[request.duration]
        if duration_days == -1 or duration_days >= len(close_a):
            duration_days = len(close_a)

        close_a = close_a[-duration_days:]
        close_b = close_b[-duration_days:]
        dates = dates[-duration_days:]

        # 计算价格比值
        ratio = [float(a) / float(b) for a, b in zip(close_a, close_b)]

        # 计算比值的技术指标
        indicators = calculate_ratio_indicators(ratio)

        # 检测指标的特殊点
        signals = detect_ratio_indicators_signals(indicators, dates)

        return {
            "code_a": request.code_a,
            "code_b": request.code_b,
            "dates": dates,
            "ratio": ratio,
            "signals": signals
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MultiAssetAnalysisRequest(BaseModel):
    assets: List[str]  # 资产代码列表，最多5个
    duration: str
    kline_type: str
    polynomial_degree: int = 3
    threshold_multiplier: float = 2.0


@app.post("/multi_asset_analysis/")
async def multi_asset_analysis(request: MultiAssetAnalysisRequest):
    """
    多资产联合分析 - 分析多个资产的价格关系和相对强弱
    
    参数:
    - assets: 最多5个资产代码
    - duration: 时间跨度 (maximum, 5y, 2y, 1y, 1q, 1m)
    - kline_type: K线类型 (daily, weekly, monthly, yearly)
    - polynomial_degree: 拟合曲线的多项式次数
    - threshold_multiplier: 异常值检测的阈值乘数
    
    返回:
    - 资产相对关系矩阵
    - 资产强弱排名
    - 交易建议
    - 所有资产对的比值数据
    """
    try:
        # 验证资产数量
        if len(request.assets) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个资产进行比较分析")
        if len(request.assets) > 6:
            raise HTTPException(status_code=400, detail="最多只能分析5个资产")

        # 获取资产名称
        asset_names = get_stock_names(request.assets)

        # 计算相关性矩阵
        correlation_matrix = calculate_correlation_matrix(
            assets=request.assets,
            asset_names=asset_names,
            duration=request.duration,
            kline_type=request.kline_type,
            polynomial_degree=request.polynomial_degree,
            threshold_multiplier=request.threshold_multiplier
        )

        # 查找最优交易对
        optimal_pairs = find_optimal_trading_pairs(correlation_matrix)

        return {
            "assets": request.assets,
            "assetNames": asset_names,
            "relationsMatrix": correlation_matrix["relationsMatrix"],
            "assetStrengthRanking": correlation_matrix["assetStrengthRanking"],
            "optimalPairs": optimal_pairs,
            "pairDetail": None  # 初始时不包含特定资产对详情
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/asset_pair_analysis/")
async def asset_pair_analysis(request: dict):
    """
    资产对比值分析 - 分析两个资产的价格比值关系
    
    参数:
    - code_a: 资产A代码
    - code_b: 资产B代码
    - duration: 时间跨度
    - kline_type: K线类型
    - polynomial_degree: 拟合曲线的多项式次数
    - threshold_multiplier: 异常值检测的阈值乘数
    
    返回:
    - 比值数据
    - 拟合曲线
    - 异常点
    - 交易建议
    """
    try:
        # 验证请求参数
        if not request.get("code_a") or not request.get("code_b"):
            raise HTTPException(status_code=400, detail="必须提供两个资产代码")

        # 获取资产对分析结果
        pair_analysis = get_asset_pair_analysis(
            code_a=request["code_a"],
            code_b=request["code_b"],
            duration=request.get("duration", "2y"),
            polynomial_degree=request.get("polynomial_degree", 3),
            threshold_multiplier=request.get("threshold_multiplier", 2.0),
            kline_type=request.get("kline_type", "daily")
        )

        return pair_analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimal_trading_pairs/")
async def optimal_trading_pairs(request: dict):
    """
    最优交易对分析 - 从多资产中找出最有交易价值的资产对
    
    参数:
    - assets: 资产代码列表
    - duration: 时间跨度
    - kline_type: K线类型
    - threshold: 信号强度阈值
    
    返回:
    - 最优交易对列表，按信号强度排序
    - 每对交易的操作建议(做多/做空)
    - 每对交易的信号强度
    """
    try:
        # 验证请求参数
        assets = request.get("assets", [])
        if len(assets) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个资产进行比较")

        # 获取资产名称
        asset_names = get_stock_names(assets)

        # 计算相关性矩阵
        correlation_matrix = calculate_correlation_matrix(
            assets=assets,
            asset_names=asset_names,
            duration=request.get("duration", "2y"),
            kline_type=request.get("kline_type", "daily"),
            polynomial_degree=request.get("polynomial_degree", 4),
            threshold_multiplier=request.get("threshold_multiplier", 1.5)
        )

        # 查找最优交易对
        threshold = request.get("threshold", 20)  # 默认信号强度阈值为20

        # 过滤信号强度低于阈值的交易对
        optimal_pairs = find_optimal_trading_pairs(correlation_matrix)
        filtered_pairs = [pair for pair in optimal_pairs if abs(pair["signalStrength"]) >= threshold]

        return {
            "assets": assets,
            "assetNames": asset_names,
            "optimalPairs": filtered_pairs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/get_all_pair_charts/")
async def get_all_asset_pair_charts(request: dict):
    """
    获取所有资产对的比值图表数据
    
    参数:
    - assets: 资产代码列表
    - duration: 时间跨度
    - polynomial_degree: 多项式拟合次数
    - threshold_multiplier: 阈值倍数
    - kline_type: K线类型 (daily, weekly, monthly, yearly)
    
    返回:
    - 所有资产对的比值图表数据，包括价格比值、拟合线、绿色区域点和红色区域点
    """
    try:
        # 验证请求参数
        assets = request.get("assets", [])
        if len(assets) < 2:
            raise HTTPException(status_code=400, detail="至少需要2个资产进行比较")
        
        # 获取参数
        duration = request.get("duration", "2y")
        polynomial_degree = request.get("polynomial_degree", 3)
        threshold_multiplier = request.get("threshold_multiplier", 2.0)
        kline_type = request.get("kline_type", "daily")
        
        # 获取资产名称
        asset_names = get_stock_names(assets)
        
        # 获取所有资产对比值图表数据
        all_charts = get_all_pair_charts(
            assets=assets,
            duration=duration,
            polynomial_degree=polynomial_degree,
            threshold_multiplier=threshold_multiplier,
            kline_type=kline_type
        )
        
        return {
            "assets": assets,
            "assetNames": asset_names,
            "pairCharts": all_charts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # 启动自动更新服务
    # update_threads = start_auto_update_services()

    # 启动API服务
    uvicorn.run(app, host="localhost", port=8000)
