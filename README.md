# Investment-advisor

##  文件结构

```
.
├── documentations/ # 用于存放doc、pdf、markdown这些东西
│   └── . 
├── back_end/ # 用于存放后端代码
│   └── peudo_backend/ # 主要后端代码目录
│        ├── apis.py # API接口
│        ├── k_chart_fetcher.py # K线图数据处理
│        └── get_stock_data/ # 股票数据获取模块
├── front_end/ # 用于存放前端的代码
│   └── stock-front-end/ # React前端项目
├── other_codes/ # 用于存放既不属于前端又不属于后端的代码
│   └── .
└── RECORD.md # 工作日志

```

## 项目功能

- 股票数据分析和比较
- 资产选择和展示
- K线图展示及技术分析

## 如何运行项目

### 后端运行步骤

1. 导航到后端目录：
   ```bash
   cd back_end/peudo_backend
   ```

2. 安装依赖（如果尚未安装）：
   ```bash
   pip install fastapi uvicorn numpy matplotlib
   ```

3. 启动FastAPI服务器：
   ```bash
   python -m uvicorn apis:app --reload
   ```

4. 后端服务将在 http://localhost:8000 上运行

### 前端运行步骤

1. 导航到前端目录：
   ```bash
   cd front_end/stock-front-end
   ```

2. 安装依赖（如果尚未安装）：
   ```bash
   npm install --legacy-peer-deps
   ```

3. 启动开发服务器：
   ```bash
   npm run dev
   ```

4. 前端应用将在 http://localhost:5173 (或 5174) 上运行

### 测试后端模块

如果需要单独测试后端模块，可以使用如下命令：

```bash
cd back_end/peudo_backend/get_stock_data
python test.py
```

## API文档

可以通过访问 http://localhost:8000/docs 查看自动生成的API文档。

