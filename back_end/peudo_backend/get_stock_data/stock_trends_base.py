import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from back_end.peudo_backend.get_stock_data.get_stock_trends_data import StockTrendsData


class StockTrendsDatabase:
    def __init__(self, db_path='stock_kline_data.db'):
        pass  # 单例模式下由__new__初始化

    _instance = None  # 单例控制

    def __new__(cls, db_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化数据库连接，使用与K线数据相同的数据库文件
            project_root = Path(__file__).parent
            default_db = project_root / "stock_kline_data.db"
            default_db.parent.mkdir(exist_ok=True, parents=True)
            cls._instance.conn = sqlite3.connect(str(default_db))
            cls._instance._setup_database()
        return cls._instance

    def _setup_database(self):
        """初始化数据库配置"""
        self.conn.execute("PRAGMA journal_mode = WAL")  # 写优化
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self._create_trends_table()  # 确保趋势表存在

    def _create_trends_table(self):
        """创建股票价格趋势表"""
        self.conn.execute(f'''
            CREATE TABLE IF NOT EXISTS stock_trends (
                stock_code TEXT,
                date TEXT,           
                current_price REAL,
                volume INTEGER,
                record_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (stock_code, date)
            ) WITHOUT ROWID
        ''')

        # 创建索引以提高查询性能
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stock_trends_date ON stock_trends (date)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_stock_trends_stock_code ON stock_trends (stock_code)")

    def insert_trends_data(self, data_list: List[Dict], code: str, name: str, type: str):
        """
        批量插入价格趋势数据
        如果已存在相同股票代码和日期的数据，将被替换
        """
        try:
            # 插入前先清理该股票非当天的数据
            self.clear_outdated_trends(code)

            for data in data_list:
                # 执行插入（如果已存在则替换）
                self.conn.execute('''
                    INSERT OR REPLACE INTO stock_trends 
                    (stock_code, date, current_price, volume)
                    VALUES (?, ?, ?, ?)
                ''', (
                    data['stock_code'],
                    data['date'],
                    data['current_price'],
                    data['volume']
                ))
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"插入趋势数据失败: {e}")
            return False

    def query_trends(self, stock_code: str, date: str = None) -> List[Dict]:
        """
        查询指定股票的价格趋势数据
        
        参数:
            stock_code: 股票代码
            date: 日期，格式为'YYYY-MM-DD'，如果不提供则查询当天
            
        返回:
            趋势数据列表
        """
        cursor = self.conn.cursor()

        # 获取当天日期
        today = datetime.now().strftime('%Y-%m-%d')

        if date:
            query_date = date
        else:
            query_date = today

        # 查询指定日期的数据
        cursor.execute('''
            SELECT stock_code, date, current_price, volume, record_time
            FROM stock_trends
            WHERE stock_code = ? AND date LIKE ?
            ORDER BY date ASC
        ''', (stock_code, f"{query_date}%"))

        # 将结果转换为字典列表
        columns = [column[0] for column in cursor.description]
        result = []
        for row in cursor.fetchall():
            result.append(dict(zip(columns, row)))

        return result

    def clear_outdated_trends(self, stock_code: str = None):
        """
        清理非当天的数据
        
        参数:
            stock_code: 股票代码，如果提供则只清理该股票的数据
        """
        today = datetime.now().strftime('%Y-%m-%d')
        try:
            if stock_code:
                # 清理指定股票的非当天数据
                self.conn.execute('''
                    DELETE FROM stock_trends
                    WHERE stock_code = ? AND date NOT LIKE ?
                ''', (stock_code, f"{today}%"))
            else:
                # 清理所有股票的非当天数据
                self.conn.execute('''
                    DELETE FROM stock_trends
                    WHERE date NOT LIKE ?
                ''', (f"{today}%",))

            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"清理非当天数据失败: {e}")
            return False

    def get_all_stock_codes(self) -> List[str]:
        """获取表中所有股票代码"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT DISTINCT stock_code FROM stock_trends
        ''')
        return [row[0] for row in cursor.fetchall()]

    def check_stock_date_current(self, stock_code: str) -> Tuple[bool, str]:
        """
        检查股票数据是否为当天
        
        返回:
            (是否当天, 最新日期)
        """
        today = datetime.now().strftime('%Y-%m-%d')
        cursor = self.conn.cursor()

        cursor.execute('''
            SELECT date FROM stock_trends
            WHERE stock_code = ?
            ORDER BY date DESC
            LIMIT 1
        ''', (stock_code,))

        row = cursor.fetchone()
        if not row:
            return False, ""

        latest_date = row[0].split()[0]  # 提取日期部分 YYYY-MM-DD
        return latest_date == today, latest_date

    def clear_trends_data(self, stock_code: str = None, date: str = None):
        """
        清除趋势数据
        
        参数:
            stock_code: 股票代码，如果提供则只清除该股票的数据
            date: 日期，格式为'YYYY-MM-DD'，如果提供则只清除该日期的数据
        """
        try:
            if stock_code and date:
                # 清除特定股票特定日期的数据
                self.conn.execute('''
                    DELETE FROM stock_trends
                    WHERE stock_code = ? AND date LIKE ?
                ''', (stock_code, f"{date}%"))
            elif stock_code:
                # 清除特定股票的所有数据
                self.conn.execute('''
                    DELETE FROM stock_trends
                    WHERE stock_code = ?
                ''', (stock_code,))
            elif date:
                # 清除特定日期的所有数据
                self.conn.execute('''
                    DELETE FROM stock_trends
                    WHERE date LIKE ?
                ''', (f"{date}%",))
            else:
                # 清除所有数据
                self.conn.execute('DELETE FROM stock_trends')

            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(f"清除数据失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if hasattr(self, 'conn'):
            self.conn.close()


if __name__ == '__main__':
    # 获取数据
    trends_fetcher = StockTrendsData("000001")
    trends_data = trends_fetcher.get_trends()
    formatted_data = trends_fetcher.format_klines(trends_data['trends'])

    # 存储数据
    db = StockTrendsDatabase()
    db.insert_trends_data(
        data_list=formatted_data,
        code=trends_data['code'],
        name=trends_data['name'],
        type=trends_data['type']
    )

    # 查询数据
    result = db.query_trends(stock_code='000001')
    print(result)

    # 不再调用auto_update_trends函数
    pass
