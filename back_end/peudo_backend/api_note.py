request = {
    "duration": "one_year",
    "code_a": "APPL",
    "code_b": "HK0725",
    "fitting_degree": "2",
    "peak_alg": "IQR",
}

durations = ['maximum', ' 5y', ' 2y', '1y', '1q', '1m', '1w']
fitting_degrees = ['1', '2', '3', '4', '5', '6']
peak_algs = ['IQR']

## RETURNS

ret = {
    "ratio": [1, 2, 3, 4],
    "fitting_lines": [1, 2, 3, 4],
    "dates":['2021-03-01', '2020-03-01', '2011-03-01'],
    "marked_dates": ['2021-03-01', '2020-03-01', '2011-03-01'],
    "colors": ['green', 'orange', 'green'],
}
