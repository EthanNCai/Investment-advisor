import json
import random
import re
import time
from typing import Dict, List, Union

import requests

MARKET_TYPE_MAPPING = {'116': '港股', '105': '美股', '113': '上海黄金期货', '118': '上海黄金现货',
                       '122': '国际黄金现货', '101': '国家黄金期货', '100': '港股'}


# 获取K线数据
class EastMoneyKLineSpider:
    def __init__(self, stock_code: str):
        self.stock_code = stock_code.strip()
        self.base_url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        self.market_type = self._determine_market()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/'
        }

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
        # 现货黄金
        elif code[-2:] == 'AU' or code[-2:] == 'AG':
            return '122'
        # 期货黄金
        elif code[-3:] == '00Y':
            return '101'
        elif code.startswith('HS') and 3 <= len(code) <= 9 or code in ['DJIA', 'SPX', 'NDX']:
            return '100'
        # 美股判断
        elif code.replace(".", "").isalpha() and 1 <= len(code) <= 5:
            return '105'  # 美股

        # A股判断
        elif code.startswith('6'):
            return '1'  # 沪市
        elif code.startswith(('0', '3', '9', '8')) and len(code) == 6:
            if code.startswith('000') or code.startswith('0000'):
                if code != '000066':
                    return '1'  # 上证指数
            return '0'  # 深市/北交所

        # 港股判断
        elif code.startswith(('1', '0')) and len(code) == 5:
            return '116'  # 港股

        else:
            raise ValueError(f"无法识别的股票代码: {self.stock_code}")

    def _generate_callback(self) -> str:
        """生成动态回调函数名"""
        if self.market_type in ['116', '105']:
            # 港股/美股格式：jQuery + 16位随机数字 + _时间戳
            rand_str = ''.join(str(random.randint(0, 9)) for _ in range(16))
            timestamp = int(time.time() * 1000)
            return f"jQuery{rand_str}_{timestamp}"
        else:
            # A股固定前缀+随机后缀
            return f"jsonp{int(time.time() * 1000)}"

    def build_params(self) -> Dict[str, str]:
        base_params = {
            'secid': f"{self.market_type}.{self.stock_code}",
            'beg': '0',
            'end': '20500000',
            'klt': '101',
            'fqt': '1',
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
        }

        # 黄金参数
        if self.market_type in ['113', '118', '101', '122', '100']:
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f62,f63,f64',
                'lmt': '100000',
                'iscca': '1',
                'ut': 'f057cbcbce2a86e2866ab8877db1d059',
                'forcect': '1'
            }
        # 港股美股参数
        elif self.market_type in ['116', '105']:
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'smplmt': '1032.6',
                'lmt': '1000000',
                '_': str(int(time.time() * 1000)),
                'cb': self._generate_callback()
            }
        # A股参数
        else:
            params = {
                'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'rtntype': '6',
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

    def get_klines(self) -> Dict[str, Union[str, List]]:
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
                "klines": data.get('klines', []),
                'type': self.get_market_name()
            }

        except requests.RequestException as e:
            return {"error": f"网络请求失败: {str(e)}"}
        except Exception as e:
            return {"error": f"发生未知错误: {str(e)}"}

    def format_klines(self, raw_klines: List[str]) -> List[Dict]:
        formatted = []
        for k in raw_klines:
            try:
                parts = k.split(',')
                # 格式化数据，保留四位小数
                amplitude = float(parts[7].rstrip('%')) / 100 if parts[7] else 0.0
                change_pct = float(parts[8].rstrip('%')) / 100 if parts[8] else 0.0
                turnover = float(parts[10].rstrip('%')) / 100 if len(parts) > 10 and parts[10] else 0.0

                formatted.append({
                    'stock_code': self.stock_code,
                    "date": parts[0],
                    "open": float(parts[1]),
                    "close": float(parts[2]),
                    "high": float(parts[3]),
                    "low": float(parts[4]),
                    "volume": int(float(parts[5])) if parts[5] else 0,
                    "amount": float(parts[6]) if parts[6] else 0.0,
                    # 振幅 - 保留四位小数
                    "amplitude": round(amplitude, 4),
                    # 涨跌幅 - 保留四位小数
                    "change_pct": round(change_pct, 4),
                    # 涨跌额
                    "change_amt": float(parts[9]) if parts[9] else 0.0,
                    # 换手率 - 保留四位小数
                    "turnover": round(turnover, 4)
                })
            except (IndexError, ValueError) as e:
                print(f"数据解析异常，跳过该条记录: {k}, 错误: {e}")
        return formatted

    def get_market_name(self):
        # 将上海黄金期货和上海黄金现货统一归类为"黄金"类型
        if self.market_type in ['113', '118', '101', '122']:
            return '黄金'
        elif self.stock_code.upper() in ['SPX', 'DJIA', 'NDX']:
            return '美股'
        return MARKET_TYPE_MAPPING.get(self.market_type, 'A股')


# 使用示例
if __name__ == "__main__":
    pass
    # 测试A股
    sh_spider = EastMoneyKLineSpider("SI00Y")
    sh_data = sh_spider.get_klines()
    print(sh_data['name'])
    print(sh_data["klines"])
    print(len(sh_data['klines']))


