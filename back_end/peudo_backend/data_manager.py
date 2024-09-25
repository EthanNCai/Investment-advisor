import pandas as pd
import  json
output = {}
stocks = ['DJI-10', 'HSI-10', 'IXIC-10', 'SPX-10']

def format_date(date_str):
    date_str = str(date_str)
    if len(date_str) != 8:
        return "日期格式错误"
    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return formatted_date

for stock in stocks:
    df = pd.read_csv(f'{stock}.csv')
    code = df.loc[0, 'ts_code']
    date_list = [format_date(date) for date in df['trade_date'].to_list()]
    close_list = [close for close in df['close']]
    output[code] = {}
    output[code]['close'] = close_list
    output[code]['dates'] = date_list

with open('stock_info_base.json', 'w') as f:
    json.dump(output, f)
