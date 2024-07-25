import urllib.parse
from datetime import datetime
import re
import random
import time
import requests
import json
import sqlite3


class StockDataFetcher:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        # 股票代码前缀 1:沪市 0:深,北市 116:港股
        self.sign = 1 if self.stock_code[0] == '6' else 0
        if self.stock_code[0] == '0' and len(self.stock_code) == 5:
            self.sign = 116
        # 动态URL
        self.base_url = 'https://53.push2.eastmoney.com/api/qt/stock/sse?'
        self.url_wap = 'https://wap.eastmoney.com/quote/api/stockinfo?'
        self.url_hk = 'https://push2.eastmoney.com/api/qt/stock/get?'
        # 请求头
        self.headers_wap = self._build_headers()
        self.headers = self._build_headers()
        self.headers_hk = self._build_headers()

    @staticmethod
    def _build_headers():
        headers = {
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, '
                          'like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36 Edg/126.0.0.0'
        }
        return headers

    def build_params(self):
        common_params = {
            'secid': '{}.{}'.format(self.sign, self.stock_code),
            'forcect': '1',
            'ut': 'f057cbcbce2a86e2866ab8877db1d059',
            'fltt': '1',
            'invt': '2',
            'mpi': '1000',
            'wbp2u': '|0|0|0|wap'
        }
        if self.sign != 116:
            common_params[
                'fields'] = 'f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f59,f60,f62,f71,f84,f85,f92,f116,f117,f122,f152,' \
                            'f161,f162,f163,f164,f165,f167,f168,f169,f170,f171,f173,f191,f260,f261,f277,f278,f288,f292,' \
                            'f293,f294,f295'
            return common_params
        else:
            jq = re.sub('\D', '', '3.51.0' + str(random.random()))
            tm = int(time.time() * 1000)
            common_params.update({
                'cb': 'jQuery{}_{}'.format(jq, tm),
                'fields': urllib.parse.unquote(
                    'f43%2Cf44%2Cf45%2Cf46%2Cf47%2Cf48%2Cf49%2Cf55%2Cf59%2Cf60%2Cf71%2Cf84%2Cf85'
                    '%2Cf92%2Cf106%2Cf116%2Cf117%2Cf152%2Cf161%2Cf164%2Cf165%2Cf167%2Cf168%2Cf169'
                    '%2Cf170%2Cf172%2Cf173%2Cf324%2Cf600%2Cf602'),
                'dect': '0',
                '_': '{}'.format(tm)
            })
            return common_params

    def fetch_data(self):
        params = self.build_params()
        params_wap = {
            'quotecode': '{}.{}'.format(self.sign, self.stock_code),
            'env': ''
        }

        try:
            data = {}
            response_wap = requests.get(self.url_wap, headers=self.headers_wap, params=params_wap)
            # 检查 HTTP 响应状态码是否为 200
            response_wap.raise_for_status()
            # 爬取港股
            if self.sign == 116:
                response_hk = requests.get(self.url_hk, headers=self.headers_hk, params=params)
                response_hk.raise_for_status()
                data = self.extract_json_from_jsonp(response_hk.text)
            else:
                # 爬取沪深京个股
                response = requests.get(self.base_url, headers=self.headers, params=params, stream=True)
                response.raise_for_status()

                # 处理基础 URL 响应
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        if chunk.startswith('data: '):
                            chunk = chunk[len('data: '):]
                        json_data = json.loads(chunk)
                        data.update(json_data)
                        break

                # print(response_wap.json().get('name'))
                # print(data.get('data'))
            return response_wap.json().get('name'), data.get('data')

        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")

    def save_data(self, conn):
        # 爬取沪深京个股
        if self._save_to_table(conn, 'stock_prices_A', self._format_data):
            return True
        return False

    def _save_to_table(self, conn, table_name, format_func):
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table_name} WHERE stock_code=?", (self.stock_code,))
        if cur.fetchone():
            print("数据已存在")
            return False

        data = self.fetch_data()
        if data:
            formatted_data = format_func(data)
            cur.execute(
                f"INSERT INTO {table_name} ({','.join(formatted_data.keys())}) VALUES ({','.join(['?'] * len(formatted_data))})",
                tuple(formatted_data.values()))
            print("数据保存成功")
            conn.commit()
            return True
        else:
            # print("无数据源")
            return False

    def _format_data(self, data):
        """
        股票代码 股票名 当前价 开盘收盘价 最高最低价 成交量和成交额 市值 市净率PB和市盈率PE 更新时间
        f43: 最新价(%) f44:最高价(%) f45:最低价(%) f46:今日开盘价(%) f47:成交量 f48:成交额 f60:昨日收盘价(%)
        f49: 外盘（5） f50:量比（%）f51:涨停价(%) f122:涨幅(%) f162:市盈率（动）PE(%) f167:市净率PB(%) f117:市值
        f169:涨跌(%) f170:涨幅(%) f171:振幅(%)
        """
        now = datetime.now()
        if self.sign != 116:
            return {
                'stock_code': self.stock_code,
                'name': data[0],
                'currentPrice': data[1]['f43'] / 100,
                'openPrice': data[1]['f46'] / 100,
                'closePrice': data[1]['f60'] / 100,
                'highPrice': data[1]['f44'] / 100,
                'lowPrice': data[1]['f45'] / 100,
                'volume': data[1]['f47'],
                'turnover': data[1]['f48'],
                'marketCapitalization': data[1]['f117'],
                'peRatio': data[1]['f162'] / 100,
                'pbRatio': data[1]['f167'] / 100,
                'timestamp': now.strftime("%Y-%m-%d %H:%M:%S"),
            }
        else:
            return {
                'stock_code': self.stock_code,
                'name': data[0],
                'currentPrice': data[1]['f43'] / 1000,
                'openPrice': data[1]['f46'] / 1000,
                'closePrice': data[1]['f60'] / 1000,
                'highPrice': data[1]['f44'] / 1000,
                'lowPrice': data[1]['f45'] / 1000,
                'volume': data[1]['f47'],
                'turnover': data[1]['f48'],
                'marketCapitalization': data[1]['f117'],
                'peRatio': data[1]['f164'] / 100,
                'pbRatio': data[1]['f167'] / 100,
                'timestamp': now.strftime("%Y-%m-%d %H:%M:%S"),
            }

    @staticmethod
    def extract_json_from_jsonp(response_text):
        # 提取 JSON 数据
        match = re.search(r'\((.*?)\)', response_text)
        if match:
            json_text = match.group(1)
            return json.loads(json_text)
        return None


if __name__ == "__main__":
    conn = sqlite3.connect('stock_data.db')
    # cur = conn.cursor()
    fetcher = StockDataFetcher('873169')
    # print(fetcher.fetch_data()[1]['f43'] / 100)
    fetcher.save_data(conn)
    conn.close()
