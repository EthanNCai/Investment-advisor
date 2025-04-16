import json
from datetime import datetime, timedelta
# import torch
from typing import List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from get_stock_data.stock_data_base import StockKlineDatabase
from indicators.technical_indicators import calculate_indicators, calculate_price_ratio_anomaly
from k_chart_fetcher import k_chart_fetcher, durations
from kline_processor.processor import process_kline_by_type
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
    threshold_arg: float = 2.0  # 添加阈值参数，默认为2.0


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
        
        # 获取前端传入的阈值系数
        threshold_multiplier = user_option_info.threshold_arg
        
        # 根据前端的阈值系数调整标准差
        adjusted_std = chart_data["threshold"]
        
        # 计算价差异常值，将调整后的标准差传递给异常检测函数
        anomaly_info = calculate_price_ratio_anomaly(
            chart_data["ratio"],
            chart_data["delta"],
            adjusted_std
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
    kline_type: str  # K线类型: "daily"(日K), "weekly"(周K), "monthly"(月K), "yearly"(年K)
    duration: str  # 时间范围: "maximum", "5y", "2y", "1y", "1q", "1m"


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
        }
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
        kline_data = process_kline_by_type(records, data.kline_type)
        
        # 计算技术指标
        indicators = calculate_indicators(kline_data)
        
        # 构建响应
        return {
            "code": stock_info["code"],
            "name": stock_info["name"],
            "type": stock_info["type"],
            "kline_data": kline_data,
            "indicators": indicators
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
