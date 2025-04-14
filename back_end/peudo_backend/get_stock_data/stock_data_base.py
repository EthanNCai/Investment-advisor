import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict

from back_end.peudo_backend.get_stock_data.get_stock_data_A_and_G import EastMoneyKLineSpider


class StockKlineDatabase:
    def __init__(self, db_path='stock_kline_data.db'):
        # self.conn = sqlite3.connect(db_path)
        # self._setup_database()
        pass  # 单例模式下由__new__初始化

    _instance = None  # 单例控制

    def __new__(cls, db_path=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 初始化数据库连接
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
        self._create_main_table()  # 确保主表存在

    def _create_main_table(self):
        """创建主表（当前年份数据）"""
        self.conn.execute(f'''
            CREATE TABLE IF NOT EXISTS kline_main (
                stock_code TEXT,
                date DATE,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                amount REAL,
                amplitude REAL,
                change_pct REAL,
                change_amt REAL,
                turnover REAL,
                PRIMARY KEY (stock_code, date)
            ) WITHOUT ROWID
        ''')

    def _get_partition_table(self, year: int):
        """根据年份计算分表名称"""
        if year == datetime.now().year:
            return "kline_main", year
        base_year = year if year % 2 == 1 else year - 1
        table_name = f"kline_{base_year}_{base_year + 1}"
        return table_name, base_year

    def _create_table_if_not_exists(self, table_name: str):
        """动态创建分表"""
        self.conn.execute(f'''
            CREATE TABLE IF NOT EXISTS {table_name} (
                stock_code TEXT,
                date DATE,
                open REAL,
                close REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                amount REAL,
                amplitude REAL,
                change_pct REAL,
                change_amt REAL,
                turnover REAL,
                PRIMARY KEY (stock_code, date)
            ) WITHOUT ROWID
        ''')

    def insert_kline_data(self, data_list: list, code, name, type):
        """批量插入K线数据"""
        try:
            for data in data_list:
                # 注意：不再更新stock_list.json，这个功能由fetch_stock_from_api处理
                date_str = data['date']
                year = int(date_str.split('-')[0])
                table_name, base_year = self._get_partition_table(year)
                # 只要2015年以后的数据
                if base_year <= 2014:
                    continue
                self._create_table_if_not_exists(table_name)

                # 执行插入（自动去重）
                self.conn.execute(f'''
                    INSERT OR REPLACE INTO {table_name}
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['stock_code'],
                    date_str,
                    data['open'],
                    data['close'],
                    data['high'],
                    data['low'],
                    data['volume'],
                    data['amount'],
                    data['amplitude'],
                    data['change_pct'],
                    data['change_amt'],
                    data['turnover']
                ))
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def query_kline(self, stock_code: str, start_date: str, end_date: str) -> list:
        """查询指定时间范围的K线数据"""
        tables = set()
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])

        # 确定需要查询的分表
        for year in range(start_year, end_year + 1):
            table, base_year = self._get_partition_table(year)
            if table not in tables:
                tables.add(table)

        # 构建联合查询和参数列表
        union_queries = []
        params = []
        for table in tables:
            union_queries.append(
                f"SELECT * FROM {table} "
                f"WHERE stock_code = ? AND date BETWEEN ? AND ?"
            )
            params.extend([stock_code, start_date, end_date])

        if not union_queries:
            return []  # 无相关表时返回空

        full_query = " UNION ALL ".join(union_queries) + " ORDER BY date ASC"

        cursor = self.conn.cursor()
        cursor.execute(full_query, params)
        return cursor.fetchall()

    def cleanup_old_data(self):
        """清理5年前的历史分表"""
        # current_year = datetime.now().year
        # cutoff_year = current_year - 5
        # 设定保留数据的起始年份
        retain_start_year = 2015

        # 获取所有分表
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        # 删除旧表
        for table in tables:
            if table == "kline_main":
                continue
            try:
                _, start, end = table.split("_")
                if int(end) < retain_start_year:
                    # print(f"删除旧表: {table}")
                    self.conn.execute(f"DROP TABLE {table}")
            except ValueError:
                continue  # 忽略非分表

    def close(self):
        """关闭数据库连接"""
        self.conn.close()


if __name__ == '__main__':
    splider = EastMoneyKLineSpider("00700")
    data = splider.get_klines()
    data_list = splider.format_klines(data['klines'])
    db = StockKlineDatabase()
    db.insert_kline_data(data_list, data['code'], data['name'], data['type'])
    result = db.query_kline(
        stock_code='00700',
        start_date='2023-07-10',
        end_date='2025-04-13'
    )
    print(result)
    # db.cleanup_old_data()
    db.close()




