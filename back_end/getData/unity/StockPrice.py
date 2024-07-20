import easyquotation
import sqlite3


class StockPrice:
    def __init__(self, stock_code):
        self.stock_code = stock_code
        self.quotation = easyquotation.use('tencent')  # 默认使用腾讯数据源
        # print(easyquotation.update_stock_codes())

    # 获取股票数据
    def fetch_data(self, data_source=None):
        if data_source:
            self.quotation = easyquotation.use(data_source)
        data = self.quotation.real(self.stock_code)
        return data.get(self.stock_code) if data else None

    # 保存股票数据到数据库
    def save_data(self, conn):
        # 尝试保存至A股数据表
        if self._save_to_table(conn, 'stock_prices_A', self._format_tencent_data, '数据已存在，A股数据更新中...',
                               'A股数据保存成功！'):
            return True
        # 若A股数据表保存失败，尝试保存至港股数据表
        self.quotation = easyquotation.use('hkquote')  # 切换至港股数据源
        return self._save_to_table(conn, 'stock_prices_G', self._format_hkquote_data, '数据已存在，港股数据更新中...',
                                   '港股数据更新成功！')

    # 通用保存方法
    def _save_to_table(self, conn, table_name, format_func, exists_message, success_message):
        cur = conn.cursor()
        cur.execute(f"SELECT * FROM {table_name} WHERE stock_code=?", (self.stock_code,))
        if cur.fetchone():
            print(exists_message)
            return False

        data = self.fetch_data()
        if data:
            formatted_data = format_func(data)
            cur.execute(
                f"INSERT INTO {table_name} ({','.join(formatted_data.keys())}) VALUES ({','.join(['?'] * len(formatted_data))})",
                tuple(formatted_data.values()))
            print(success_message)
            conn.commit()
            return True
        else:
            # print("无数据源")
            return False

    # 格式化A股数据源的数据
    def _format_tencent_data(self, data):
        return {
            'stock_code': self.stock_code,
            'name': data['name'],
            'currentPrice': data['now'],
            'openPrice': data['open'],
            'closePrice': data['close'],
            'highPrice': data['high'],
            'lowPrice': data['low'],
            'volume': data['成交量(手)'],
            'turnover': data['成交额(万)'],
            'marketCapitalization': data['总市值'],
            'peRatio': data['PE'],
            'pbRatio': data['PB'],
            'timestamp': data['datetime'],
        }

    # 格式化港股数据源的数据
    def _format_hkquote_data(self, data):
        return {
            'stock_code': self.stock_code,
            'name': data['name'],
            'currentPrice': data['price'],
            'openPrice': data['openPrice'],
            'highPrice': data['high'],
            'lowPrice': data['low'],
            'volume': data['amount'],
            'dtd': data['dtd'],
            'timestamp': data['time'],
        }


if __name__ == '__main__':
    conn = sqlite3.connect('stock_data.db')
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS stock_prices_A
                     (stock_code TEXT PRIMARY KEY,
                      name TEXT,
                      currentPrice REAL,
                      openPrice REAL,
                      closePrice REAL,
                      highPrice REAL,
                      lowPrice REAL,
                      volume REAL,
                      turnover REAL,
                      marketCapitalization REAL,
                      peRatio REAL,
                      pbRatio REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

    cur.execute('''CREATE TABLE IF NOT EXISTS stock_prices_G
                     (stock_code TEXT PRIMARY KEY,
                      name TEXT,
                      currentPrice REAL,
                      openPrice REAL,
                      highPrice REAL,
                      lowPrice REAL,
                      volume REAL,
                      dtd REAL,
                      timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    sp = StockPrice('301603')
    sp.save_data(conn)
    conn.close()
