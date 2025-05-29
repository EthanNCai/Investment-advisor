import re
import os
import json
from typing import Dict, Any, Optional, List, Tuple
from pypinyin import lazy_pinyin

# from back_end.peudo_backend.get_stock_data.get_stock_data_A_and_G import EastMoneyKLineSpider
# from back_end.peudo_backend.get_stock_data.stock_data_base import StockKlineDatabase


from get_stock_data.stock_data_base import StockKlineDatabase
from get_stock_data.get_stock_data_A_and_G import EastMoneyKLineSpider


def get_pinyin_initials(text: str) -> str:
    """
    获取中文文本的拼音首字母
    
    参数:
        text: 中文文本
        
    返回:
        拼音首字母字符串
    """
    if not text:
        return ""
    initials = ''.join([p[0] for p in lazy_pinyin(text)])
    return initials.lower()


def score_match(asset: Dict[str, str], keyword: str) -> int:
    """
    计算匹配得分，用于排序
    完全匹配 > 代码匹配 > 名称匹配 > 拼音首字母匹配
    
    参数:
        asset: 资产信息字典
        keyword: 搜索关键词
        
    返回:
        得分，越高越匹配
    """
    keyword_lower = keyword.lower()
    code_lower = asset['code'].lower()
    name_lower = asset['name'].lower()
    pinyin_initials = get_pinyin_initials(asset['name'])
    
    # 完全匹配 (100分)
    if code_lower == keyword_lower or name_lower == keyword_lower:
        return 100
    
    # 代码包含匹配 (80分)
    if keyword_lower in code_lower:
        position = code_lower.find(keyword_lower)
        # 前置匹配得分更高
        return 80 - position
    
    # 名称包含匹配 (60分)
    if keyword_lower in name_lower:
        position = name_lower.find(keyword_lower)
        return 60 - position
    
    # 拼音首字母匹配 (40分)
    if keyword_lower in pinyin_initials:
        position = pinyin_initials.find(keyword_lower)
        return 40 - position
    
    return 0  # 不匹配


async def fetch_stock_from_api(code: str) -> Optional[Dict[str, Any]]:
    """
    从API获取股票信息并保存到数据库和stock_list.json
    
    参数:
        code: 股票代码
        
    返回:
        股票信息字典，如果获取失败则返回None
    """
    try:
        # 尝试使用爬虫获取数据
        spider = EastMoneyKLineSpider(code)
        kline_data = spider.get_klines()
        
        # 检查是否有错误
        if "error" in kline_data:
            print(f"获取股票数据错误: {kline_data['error']}")
            return None
        
        # 确保获取到了必要数据
        if not kline_data["code"] or not kline_data["name"] or not kline_data["klines"]:
            print(f"股票数据不完整: {code}")
            return None
        
        # 格式化K线数据并保存到数据库
        formatted_klines = spider.format_klines(kline_data["klines"])
        
        if not formatted_klines:
            print(f"股票没有K线数据: {code}")
            return None
        
        # 保存到数据库
        db = StockKlineDatabase()
        db.insert_kline_data(
            formatted_klines, 
            kline_data["code"], 
            kline_data["name"], 
            kline_data["type"]
        )
        
        # 创建股票信息字典
        stock_info = {
            "code": kline_data["code"],
            "name": kline_data["name"],
            "type": kline_data["type"]
        }
        
        # 更新stock_list.json文件
        try:
            # 使用相对路径，确保在正确的目录下更新文件
            stock_list_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_list.json')
            
            # 读取现有stock_list.json
            with open(stock_list_path, 'r', encoding='utf-8') as file:
                stock_data = json.load(file)
                
            # 检查股票是否已存在
            exists = False
            for stock in stock_data['stocks']:
                if stock['code'] == stock_info['code']:
                    exists = True
                    break
                    
            # 如果股票不存在，添加到列表中
            if not exists:
                stock_data['stocks'].append(stock_info)
                # 保存更新后的文件
                with open(stock_list_path, 'w', encoding='utf-8') as file:
                    json.dump(stock_data, file, ensure_ascii=False, indent=2)
                print(f"股票 {stock_info['code']} ({stock_info['name']}) 已添加到stock_list.json")
        except Exception as e:
            print(f"更新stock_list.json失败: {str(e)}")
        
        # 返回股票基本信息
        return stock_info
        
    except Exception as e:
        print(f"获取股票数据异常: {str(e)}")
        return None


def search_stocks(stock_list: List[Dict[str, str]], keyword: str) -> List[Tuple[Dict[str, str], int]]:
    """
    搜索匹配的股票
    
    参数:
        stock_list: 股票列表
        keyword: 搜索关键词
        
    返回:
        匹配的股票列表及其得分
    """
    matched_stocks = []
    
    for stock_info in stock_list:
        score = score_match(stock_info, keyword)
        if score > 0:
            matched_stocks.append((stock_info, score))
    
    return matched_stocks


def is_stock_code_format(keyword: str) -> bool:
    """
    检查关键词是否符合股票代码格式
    
    参数:
        keyword: 搜索关键词
        
    返回:
        是否为股票代码格式
    """
    # 支持A股、港股、美股编码和黄金代码格式
    return bool(re.match(r'^[0-9]{5,6}$|^[0-9]{4,5}$|^[A-Za-z0-9\.]+$|^[Aa][Uu][0-9]{3,4}$|^[Aa][Uu][A-Za-z]+$|^[Nn][Yy][Aa][Uu]$', keyword))
