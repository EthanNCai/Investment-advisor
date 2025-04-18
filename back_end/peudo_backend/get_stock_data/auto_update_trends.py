import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

from back_end.peudo_backend.get_stock_data.get_stock_data_A_and_G import EastMoneyKLineSpider
from back_end.peudo_backend.get_stock_data.get_stock_trends_data import StockTrendsData
from back_end.peudo_backend.get_stock_data.stock_data_base import StockKlineDatabase
from back_end.peudo_backend.get_stock_data.stock_trends_base import StockTrendsDatabase


def auto_update_trends(interval_minutes: int = 3):
    """
    自动更新所有股票的当日价格趋势数据
    
    参数:
        interval_minutes: 更新间隔(分钟)
    """
    try:
        db = StockTrendsDatabase()
        print(f"[{datetime.now()}] 开始自动更新股票趋势数据...")

        # 获取所有股票代码
        stock_codes = db.get_all_stock_codes()
        # 首先清理所有非当天数据
        db.clear_outdated_trends()
        # 如果没有股票，从stock_list.json获取一些股票
        if not stock_codes:
            try:
                stock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                          'stock_list.json')
                with open(stock_file, 'r', encoding='utf-8') as f:
                    stock_data = json.load(f)
                    # 获取前30只股票作为示例
                    stock_codes = [stock['code'] for stock in stock_data.get('stocks', [])[:30]]
            except Exception as e:
                print(f"获取股票列表失败: {e}")
        # 更新每只股票的数据
        update_count = 0
        for code in stock_codes:
            try:

                # 清理该股票的旧数据
                db.clear_trends_data(stock_code=code)

                # 获取新数据
                trends_fetcher = StockTrendsData(code)
                trends_data = trends_fetcher.get_trends()

                if "error" not in trends_data and trends_data.get('trends'):
                    formatted_data = trends_fetcher.format_klines(trends_data['trends'])

                    # 存储新数据
                    if formatted_data:
                        db.insert_trends_data(
                            data_list=formatted_data,
                            code=trends_data['code'],
                            name=trends_data['name'],
                            type=trends_data['type']
                        )
                        update_count += 1
                    else:
                        print(f"股票 {code} 没有可用的趋势数据")
                else:
                    error_msg = trends_data.get('error', '没有可用数据')
                    print(f"获取股票 {code} 的趋势数据失败: {error_msg}")

                # 添加延迟避免请求过于频繁
                time.sleep(1)
            except Exception as e:
                print(f"处理股票 {code} 时出错: {e}")

        print(f"[{datetime.now()}] 股票趋势数据更新完成，共更新 {update_count} 只股票")

    except Exception as e:
        print(f"自动更新过程中出错: {e}")
    finally:
        # 设置下一次运行
        if interval_minutes > 0:
            print(f"下一次更新将在 {interval_minutes} 分钟后进行")
            time.sleep(interval_minutes * 60)
            auto_update_trends(interval_minutes)


def auto_update_kline_history(batch_size: int = 20, interval_hours: float = 0.5):
    """
    自动更新所有股票的历史K线数据
    
    参数:
        batch_size: 每批处理的股票数量
        interval_hours: 更新间隔(小时)，默认为0.5小时（30分钟）
    """
    try:
        db = StockKlineDatabase()
        print(f"[{datetime.now()}] 开始自动更新股票历史K线数据...")

        # 获取所有股票列表
        stock_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stock_list.json')
        all_stocks = []

        try:
            with open(stock_file, 'r', encoding='utf-8') as f:
                stock_data = json.load(f)
                all_stocks = stock_data.get('stocks', [])
                print(f"从stock_list.json加载了 {len(all_stocks)} 只股票")
        except Exception as e:
            print(f"读取股票列表失败: {e}")
            return

        # 将股票分成多个批次处理
        total_stocks = len(all_stocks)
        batches = [all_stocks[i:i + batch_size] for i in range(0, total_stocks, batch_size)]

        print(f"总共 {total_stocks} 只股票，分成 {len(batches)} 个批次进行处理")

        update_count = 0
        failed_count = 0

        # 处理每个批次
        for batch_idx, batch in enumerate(batches):
            print(f"处理第 {batch_idx + 1}/{len(batches)} 批股票...")

            # 处理批次中的每只股票
            for stock in batch:
                stock_code = stock['code']
                stock_name = stock['name']

                try:
                    print(f"更新股票 {stock_code} ({stock_name}) 的历史K线数据")
                    spider = EastMoneyKLineSpider(stock_code)
                    kline_data = spider.get_klines()

                    if "error" not in kline_data and kline_data.get('klines'):
                        # 格式化数据
                        formatted_data = spider.format_klines(kline_data['klines'])

                        if formatted_data:
                            # 插入数据库
                            success = db.insert_kline_data(formatted_data, kline_data['code'], kline_data['name'],
                                                           kline_data['type'])
                            if success:
                                update_count += 1
                                print(
                                    f"成功更新股票 {stock_code} ({stock_name}) 的K线数据，共 {len(formatted_data)} 条记录")
                            else:
                                failed_count += 1
                                print(f"插入股票 {stock_code} 的K线数据失败")
                        else:
                            print(f"股票 {stock_code} 没有可用的K线数据")
                    else:
                        error_msg = kline_data.get('error', '没有可用数据')
                        print(f"获取股票 {stock_code} 的K线数据失败: {error_msg}")
                        failed_count += 1

                    # 添加延迟避免请求过于频繁
                    time.sleep(2)
                except Exception as e:
                    print(f"处理股票 {stock_code} 时出错: {e}")
                    failed_count += 1

            # 批次处理完成后，给服务器一些喘息时间
            if batch_idx < len(batches) - 1:
                print(f"批次 {batch_idx + 1} 处理完成，休息10秒后继续...")
                time.sleep(10)

        # 清理5年以上的历史数据
        print("清理5年以上的历史数据...")
        db.cleanup_old_data()

        print(f"[{datetime.now()}] 股票历史K线数据更新完成")
        print(f"成功更新: {update_count} 只股票")
        print(f"更新失败: {failed_count} 只股票")

    except Exception as e:
        print(f"自动更新历史K线数据过程中出错: {e}")
    finally:
        # 设置下一次运行
        if interval_hours > 0:
            print(f"下一次更新将在 {interval_hours} 小时后进行")
            time.sleep(interval_hours * 3600)
            auto_update_kline_history(batch_size, interval_hours)


def start_auto_update_services():
    """
    启动自动更新服务
    """
    print(f"[{datetime.now()}] 启动自动更新服务...")

    # 启动趋势数据更新线程
    trends_thread = threading.Thread(
        target=auto_update_trends,
        args=(3,),
        daemon=True,
        name="TrendsUpdateThread"
    )
    trends_thread.start()
    print("趋势数据自动更新服务已启动")

    # # 启动历史K线数据更新线程（每30分钟更新一次）
    # kline_thread = threading.Thread(
    #     target=auto_update_kline_history,
    #     args=(20, 0.5),
    #     daemon=True,
    #     name="KlineUpdateThread"
    # )
    # kline_thread.start()
    # print("历史K线数据自动更新服务已启动，更新频率：每30分钟")

    return {
        "trends_thread": trends_thread,
        # "kline_thread": kline_thread
    }


if __name__ == '__main__':
    pass
