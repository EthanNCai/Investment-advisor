
目前后端仅有两个API



1./search_stocks/${search_keyword} (GET请求)


receive

```json
{
  "result": [
  {
    "name": "",
    "code": "",
    "type": ""
  },
  {
    "name": "",
    "code": "",
    "type": ""
  }
]}

```


2./get_k_chart_info (POST请求)

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