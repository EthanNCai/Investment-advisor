from datetime import datetime

import requests
import json
import sqlite3


class StockDataFetcher:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        # 动态URL
        self.base_url = 'https://53.push2.eastmoney.com/api/qt/stock/sse?'
        self.url_wap = 'https://wap.eastmoney.com/quote/api/stockinfo?'

        # 请求头
        self.headers_wap = {
            'accept': 'application/json, text/javascript, */*; q=0.01',
            'cookie': 'qgqp_b_id=1b08f4259589caf6d5a08dd78c506ccd; quote_lt=1; st_si=74474436456921; wap_ck2=true; '
                      'Hm_lvt_2c85d6015aa17262cef9dd0ab45d0e2e=1721445953; '
                      'Hm_lpvt_2c85d6015aa17262cef9dd0ab45d0e2e=1721445953; HMACCOUNT=72880F2E97DF824A; '
                      'st_asi=delete; HAList=ty-0-834407-%u9A70%u8BDA%u80A1%u4EFD%2Cty-1-900906-%u4E2D%u6BC5%u8FBEB'
                      '%2Cty-1-900914-%u9526%u5728%u7EBFB%2Cty-1-603285-%u952E%u90A6%u80A1%u4EFD%2Cty-1-688515-%u88D5'
                      '%u592A%u5FAE-U%2Cty-0-832982-%u9526%u6CE2%u751F%u7269%2Cty-0-000001-%u5E73%u5B89%u94F6%u884C'
                      '%2Cty-0-872541-%u94C1%u5927%u79D1%u6280%2Cty-1-000077-%u4FE1%u606F%u7B49%u6743%2Cty-1-688562'
                      '-%u822A%u5929%u8F6F%u4EF6; cdv_guba.eastmoney.com=1; cdv_finance.eastmoney.com=1; '
                      'st_pvi=52519945943447; st_sp=2024-06-06%2011%3A20%3A48; '
                      'st_inirUrl=https%3A%2F%2Fcn.bing.com%2F; st_sn=233; '
                      'st_psi=20240720213729820-113807304294-3090027531',
            'host': 'wap.eastmoney.com',
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, '
                          'like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36 Edg/126.0.0.0',
            'x-requested-with': 'XMLHttpRequest'
        }

        self.headers = {
            'accept': 'text/event-stream',
            'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, '
                          'like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36 Edg/126.0.0.0'
        }

    def build_params(self):
        # 股票代码 1:沪市 0:深,北市
        sign = 1 if self.stock_code[0] == '6' else 0
        return {
            'secid': '{}.{}'.format(sign, self.stock_code),
            'forcect': '1',
            'fields': 'f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f55,f59,f60,f62,f71,f84,f85,f92,f116,f117,f122,f152,'
                      'f161,f162,f163,f164,f165,f167,f168,f169,f170,f171,f173,f191,f260,f261,f277,f278,f288,f292,'
                      'f293,f294,f295',
            'ut': 'f057cbcbce2a86e2866ab8877db1d059',
            'fltt': '1',
            'invt': '2',
            'mpi': '1000',
            'wbp2u': '|0|0|0|wap'
        }

    def fetch_data(self):
        params = self.build_params()
        params_wap = {
            'quotecode': '{}.{}'.format(1 if self.stock_code[0] == '6' else 0, self.stock_code),
            'env': ''
        }

        try:
            response = requests.get(self.base_url, headers=self.headers, params=params, stream=True)
            response.raise_for_status()

            response_wap = requests.get(self.url_wap, headers=self.headers_wap, params=params_wap)
            # 检查 HTTP 响应状态码是否为 200
            response_wap.raise_for_status()

            # 处理基础 URL 响应
            data = {}
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
            print("无数据源")
            return False

    def _format_data(self, data):
        """
        股票代码 股票名 当前价 开盘收盘价 最高最低价 成交量和成交额 市值 市净率PB和市盈率PE 更新时间
        f43: 最新价(%) f44:最高价(%) f45:最低价(%) f46:今日开盘价(%) f47:成交量 f48:成交额 f60:昨日收盘价(%)
        f49: 外盘（5） f50:量比（%）f51:涨停价(%) f122:涨幅(%) f162:市盈率（动）PE(%) f167:市净率PB(%) f117:市值
        f169:涨跌(%) f170:涨幅(%) f171:振幅(%)
        """
        now = datetime.now()
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


if __name__ == "__main__":
    conn = sqlite3.connect('stock_data.db')
    # cur = conn.cursor()
    fetcher = StockDataFetcher('300554')
    # print(fetcher.fetch_data()[1]['f43'] / 100)
    fetcher.save_data(conn)
    conn.close()
