# Investment-advisor

##  文件结构

```
├── documentations/ # 项目文档目录
├── back_end/ # 后端代码
│   └── peudo_backend/ # 主要后端代码目录
│        ├── apis.py # API接口
│        ├── k_chart_fetcher.py # K线图数据获取与处理
│        ├── get_stock_data/ # 股票数据获取模块
│        ├── indicators/ # 技术指标计算模块
│        │   ├── technical_indicators.py # 常用技术指标实现
│        │   └── investment_signals.py # 投资信号生成  
│        ├── prediction/ # 预测模块
│        │   ├── model_trainer.py # LSTM模型训练
│        │   └── model_predictor.py # 价格比值预测实现
│        └── backtest/ # 回测模块
│            └── backtest_strategy.py # 回测策略实现
├── front_end/ # 前端代码
│   └── stock-front-end/ # React前端项目
│        ├── src/components/ # React组件
│        │   ├── asset-selection/ # 资产选择组件
│        │   ├── kline-chart/ # K线图组件
│        │   ├── ratio-analysis/ # 价格比值分析组件
│        │   └── prediction/ # 预测结果展示组件
│        └── src/pages/ # 页面组件
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



### 依赖需求

后端依赖：
```
fastapi==0.95.0
uvicorn==0.21.1
pydantic==1.10.7
numpy==1.24.2
scikit-learn==1.2.2
pandas==2.0.0
tensorflow==2.12.0
matplotlib==3.7.1
```

### 安装步骤

1. 克隆项目代码
2. 安装后端依赖
   ```
   cd back_end/peudo_backend
   pip install -r requirements.txt
   ```
3. 安装前端依赖
   ```
   cd front_end/stock-front-end
   npm install
   ```

### 启动服务

1. 启动后端
   ```
   cd back_end/peudo_backend
   uvicorn apis:app --reload --host 0.0.0.0 --port 8000
   ```
2. 启动前端
   ```
   cd front_end/stock-front-end
   npm start
   ```

