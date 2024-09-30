

get_k_chart_info (POST)

send
```json
{
  "code_a": "SPX",
  "code_b": "HSI",
  "degree": 2,
  "duration": "1y",
  "threshold": 1.5
}
```

receive

```json
{
  "ratio": [1,2,3,4],
  "dates": [],
  "date_splitters": [],
  "colors": [],
  "close_a": [],
  "close_b": []
}
```