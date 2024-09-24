import json

types = ['hk', 'a', 'us']

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 允许所有源访问，可以根据需要进行定制
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/search_stocks/{keyword}")
def search_stocks(keyword: str):
    with open('stock_list.json', 'r', encoding='utf-8') as file:
        stock_info_json = json.load(file)
        stock_info_list = stock_info_json['stocks']

    searched = []

    for stock_info in stock_info_list:
        # 检查code
        if stock_info['code'].find(keyword) == -1 and stock_info['name'].find(keyword) == -1:
            continue
        searched.append(stock_info)

    return {"result": searched}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)