import json
import random
import re
import time
from typing import Dict, List, Union

import requests

from back_end.peudo_backend.get_stock_data.get_stock_data_A_and_G import MARKET_TYPE_MAPPING


# 获取当天股票走势数据
class StockTrendsData:
    def __init__(self, stock_code: str):
        self.stock_code = stock_code.strip()
        self.base_url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
        self.market_type = self._determine_market()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                          'Chrome/135.0.0.0 Safari/537.36 Edg/135.0.0.0',
            'Referer': 'https://quote.eastmoney.com/'
        }

    def _generate_callback(self) -> str:
        """生成动态回调函数名"""
        if self.market_type == '116' or self.market_type == '105':
            # 港股格式：jQuery + 16位随机数字 + _时间戳
            rand_str = ''.join(str(random.randint(0, 9)) for _ in range(16))
            timestamp = int(time.time() * 1000)
            return f"jQuery{rand_str}_{timestamp}"
        else:
            # A股固定前缀+随机后缀
            return f"jsonp{int(time.time() * 1000)}"

    def build_params(self) -> Dict[str, str]:
        base_params = {
            'secid': f"{self.market_type}.{self.stock_code}",
            'iscr': '0',
            'ndays': '1',
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b'
        }
        if self.market_type in ['113', '118']:
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f17',
                'fields2': 'f51,f53,f54,f55,f56,f57,f58',
                'iscca': '0',
                'ut': 'f057cbcbce2a86e2866ab8877db1d059',
                'cb': self._generate_callback()
            }
        # 字段参数区分市场(港股或美股）
        elif self.market_type == '116' or self.market_type == '105':
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
                '_': str(int(time.time() * 1000)),
                'cb': 'quotepushdata1'
            }
        else:
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f17',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58',
                'iscca': 0,
                'cb': self._generate_callback()
            }

        return {**base_params, **params}

    def _parse_response(self, text: str) -> Dict:
        try:
            # 两种可能的JSONP格式处理
            json_str = re.findall(r'({.*})', text)[0]
            return json.loads(json_str)
        except (IndexError, json.JSONDecodeError) as e:
            print(f"原始响应内容（片段）:\n{text[:300]}...")
            raise ValueError("响应解析失败，请检查接口是否变更") from e

    def get_trends(self) -> Dict[str, Union[str, List]]:
        try:
            # 构造请求
            params = self.build_params()
            response = requests.get(
                self.base_url,
                params=params,
                headers=self.headers,
                timeout=15
            )
            response.raise_for_status()

            # 解析数据
            json_data = self._parse_response(response.text)

            if json_data.get('rc') != 0:
                return {"error": f"接口返回错误: {json_data.get('rt', '未知错误')}"}

            data = json_data.get('data', {})

            return {
                "code": data.get('code', ''),
                "name": data.get('name', ''),
                "trends": data.get('trends', []),
                'type': self.get_market_name()
            }

        except requests.RequestException as e:
            return {"error": f"网络请求失败: {str(e)}"}
        except Exception as e:
            return {"error": f"发生未知错误: {str(e)}"}

    def get_market_name(self):
        if self.market_type in ['113', '118']:
            return '黄金'
        return MARKET_TYPE_MAPPING.get(self.market_type, 'A股')

    def _determine_market(self) -> str:
        code = self.stock_code.upper()

        # 黄金期货判断 (113)
        if (code.startswith('AU') and 5 <= len(code) <= 6 and any(c.isdigit() for c in code)) or \
                (code in ['AUM', 'AUS']):
            return '113'  # 上海黄金期货

        # 黄金现货判断 (118)
        elif code.startswith('NYAU') or \
                code in ['AUTD', 'MAUTD', 'AU100', 'AU9995', 'AU9999', 'SHAU', 'IAU999', 'PT9995',
                         'AGTD', 'IAU995', 'IAU100', 'PGC30', 'AUTN1', 'AUTN2', 'AU995', 'AU50',
                         'AG9999', 'AG999'] or \
                (code.startswith('AU') and ('TD' in code or 'TN' in code)):
            return '118'  # 上海黄金现货

        # 美股判断
        elif code.replace(".", "").isalpha() and 1 <= len(code) <= 5:
            return '105'  # 美股

        # A股判断
        elif code.startswith('6'):
            return '1'  # 沪市
        elif code.startswith(('0', '3', '9', '8')) and len(code) == 6:
            if code.startswith('000') or code.startswith('0000'):
                if code != '000066':
                    return '1'
            return '0'  # 深市/北交所

        # 港股判断
        elif code.startswith(('1', '0')) and len(code) == 5:
            return '116'  # 港股

        else:
            raise ValueError(f"无法识别的股票代码: {self.stock_code}")

    def format_klines(self, raw_klines: List[str]) -> List[Dict]:
        formatted = []
        for k in raw_klines:
            try:
                parts = k.split(',')
                formatted.append({
                    'stock_code': self.stock_code,
                    "date": parts[0],
                    # 当前价格
                    "current_price": float(parts[2]),
                    # 成交量
                    "volume": parts[5]
                })
            except (IndexError, ValueError) as e:
                print(f"数据解析异常，跳过该条记录: {k}")
                print(f"异常信息: {e}")
        return formatted


