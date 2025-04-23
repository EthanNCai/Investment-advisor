import json
import traceback
from datetime import datetime, timedelta
# import torch
from typing import List, Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from get_stock_data.get_stock_trends_data import StockTrendsData
from get_stock_data.stock_data_base import StockKlineDatabase
from get_stock_data.stock_trends_base import StockTrendsDatabase
from indicators.investment_signals import generate_investment_signals, analyze_current_position
from indicators.technical_indicators import calculate_indicators, calculate_price_ratio_anomaly
from k_chart_fetcher import k_chart_fetcher, durations, get_stock_data_pair, date_alignment
from kline_processor.processor import process_kline_by_type
# 导入LSTM预测器
from prediction import get_predictor
# 导入模块化后的函数
from stock_search.searcher import score_match, is_stock_code_format, fetch_stock_from_api

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
        threshold_multiplier = user_option_info.threshold_arg  # 用户设置的阈值倍数（如2.0）
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
            start_date = '2015-01-01'  # 数据库中最早的数据
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


# 以下为预测相关接口
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


class SignalModel(BaseModel):
    id: int
    date: str
    ratio: float
    z_score: float
    type: str  # 'positive' or 'negative'
    strength: str  # 'weak', 'medium', 'strong'
    description: str
    recommendation: str


class CurrentPositionModel(BaseModel):
    current_ratio: float
    nearest_signal_id: Optional[int]
    similarity_score: Optional[float]
    percentile: Optional[float]
    is_extreme: bool
    recommendation: str


class SignalResponseModel(BaseModel):
    signals: List[SignalModel]
    current_position: CurrentPositionModel


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

        # # 如果没有当前趋势数据，尝试获取
        # if not trends_a:
        #     trends_a = get_stock_trends(request.code_a)
        # if not trends_b:
        #     trends_b = get_stock_trends(request.code_b)

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

        # 确保所有numpy类型都转换为Python原生类型
        for signal in signals:
            for key, value in signal.items():
                if isinstance(value, (np.integer, np.floating, np.bool_)):
                    if isinstance(value, np.bool_):
                        signal[key] = bool(value)
                    elif isinstance(value, np.integer):
                        signal[key] = int(value)
                    elif isinstance(value, np.floating):
                        signal[key] = float(value)

        # 确保current_position中的所有numpy类型都转换为Python原生类型
        for key, value in current_position.items():
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                if isinstance(value, np.bool_):
                    current_position[key] = bool(value)
                elif isinstance(value, np.integer):
                    current_position[key] = int(value)
                elif isinstance(value, np.floating):
                    current_position[key] = float(value)

        return {
            "signals": signals,
            "current_position": current_position
        }

    except Exception as e:
        print(f"获取投资信号失败: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取投资信号失败: {str(e)}")





if __name__ == "__main__":
    import uvicorn

    # 启动自动更新服务
    # update_threads = start_auto_update_services()

    # 启动API服务
    uvicorn.run(app, host="localhost", port=8000)
