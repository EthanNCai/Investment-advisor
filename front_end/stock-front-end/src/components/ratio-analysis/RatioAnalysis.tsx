import React, { useState, useEffect } from 'react';
import { Select, message, notification, Spin, Space, Tabs, Card, Typography } from 'antd';
import ReactECharts from 'echarts-for-react';
import AnomalyDetection from './AnomalyDetection';
import { SliderSelector } from '../selectors/SliderSelector';

// 使用Typography组件
const { Text } = Typography;

// 时间跨度定义，用于界面展示和后端交互
interface DurationOption {
  label: string;     // 显示的文本 
  backendValue: string;  // 传给后端的值
}

const DURATION_OPTIONS: DurationOption[] = [
  { label: "1个月", backendValue: "1m" },
  { label: "3个月", backendValue: "1q" },
  { label: "1年", backendValue: "1y" },
  { label: "2年", backendValue: "2y" },
  { label: "5年", backendValue: "5y" },
  { label: "全部", backendValue: "maximum" }
];

interface Anomaly {
  index: number;
  value: number;
  z_score: number;
  deviation: number;
}

interface AnomalyInfo {
  mean: number;
  std: number;
  anomalies: Array<Anomaly>;
  warning_level: 'normal' | 'medium' | 'high';
  upper_bound: number;
  lower_bound: number;
}

interface ChartData {
  ratio: number[];
  dates: string[];
  close_a: number[];
  close_b: number[];
  fitting_line: number[];
  delta: number[];
  threshold: number;
  anomaly_info: AnomalyInfo;
}

interface StockInfo {
  code: string;
  name: string;
}

const RatioAnalysis: React.FC = () => {
  const [selectedStockA, setSelectedStockA] = useState<string>('');
  const [selectedStockB, setSelectedStockB] = useState<string>('');
  const [selectedDuration, setSelectedDuration] = useState<string>("1y");
  const [selectedDegree, setSelectedDegree] = useState<number>(3);
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [stockList, setStockList] = useState<StockInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [anomalyThreshold, setAnomalyThreshold] = useState<number>(2.0);
  const [showDelta, setShowDelta] = useState<boolean>(true);
  const [activeTab, setActiveTab] = useState<string>('1');
  
  // 从服务器获取股票列表
  useEffect(() => {
    const fetchStockList = async () => {
      try {
        const response = await fetch('http://localhost:8000/get_all_assets/');
        const data = await response.json();
        setStockList(data.assets);
      } catch (error) {
        console.error('Error fetching stock list:', error);
        message.error('获取股票列表失败');
      }
    };

    fetchStockList();
  }, []);

  // 添加新的useEffect，当两只股票都已选择时自动加载图表数据
  useEffect(() => {
    if (selectedStockA && selectedStockB) {
      console.log("两只股票都已选择，自动加载数据:", selectedStockA, selectedStockB);
      updateChart(selectedStockA, selectedStockB, selectedDuration, selectedDegree, anomalyThreshold, false);
    }
  }, [selectedStockA, selectedStockB]);

  // 更新图表数据 - 重构为直接接收参数
  const updateChart = async (
    stockA: string = selectedStockA, 
    stockB: string = selectedStockB, 
    duration: string = selectedDuration, 
    degree: number = selectedDegree, 
    threshold: number = anomalyThreshold,
    isThresholdAdjustment: boolean = false
  ) => {
    if (!stockA || !stockB) {
      message.warning('请选择两只股票');
      return;
    }
    
    // 检查是否选择了相同的股票
    if (stockA === stockB) {
      message.error('股票A和股票B不能选择相同的股票');
      return;
    }

    setLoading(true);
    try {
      // 记录传递给后端的值
      console.log("发送到后端的时间跨度:", duration);
      
      const response = await fetch('http://localhost:8000/get_k_chart_info/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code_a: stockA,
          code_b: stockB,
          duration: duration, 
          degree: degree,
          threshold_arg: threshold
        }),
      });
      
      const data = await response.json();
      
      // 计算并更新上下界值，确保数据一致性
      const updatedData = updateBoundsInData(data, threshold);
      
      // 设置新的图表数据
      setChartData(updatedData);
      
      // 只显示通知，不再自动切换选项卡
      if (updatedData.anomaly_info.warning_level === 'high') {
        notification.warning({
          message: '高风险预警',
          description: '检测到显著的价差异常值，请谨慎操作！',
          duration: 5
        });
      }
    } catch (error) {
      console.error('Error fetching chart data:', error);
      message.error('获取图表数据失败');
    } finally {
      setLoading(false);
    }
  };

  // 统一计算边界值的辅助函数
  const updateBoundsInData = (data: any, threshold: number) => {
    if (!data || !data.anomaly_info) return data;
    
    // 计算新的上下界
    const mean = data.anomaly_info.mean;
    const std = data.anomaly_info.std;
    const upperBound = mean + threshold * std;
    const lowerBound = mean - threshold * std;
    
    // 返回更新了上下界的新对象
    return {
      ...data,
      anomaly_info: {
        ...data.anomaly_info,
        upper_bound: upperBound,
        lower_bound: lowerBound
      }
    };
  };

  // 当阈值变化时更新边界
  const handleThresholdChange = (value: number) => {
    setAnomalyThreshold(value);
    
    // 如果图表数据已加载，则更新异常检测边界
    if (chartData) {
      // 使用统一的方法更新上下界值
      const updatedChartData = updateBoundsInData(chartData, value);
      
      // 更新状态
      setChartData(updatedChartData);
      
      // 当用户手动调整阈值后，也发送请求获取新的异常检测结果
      if (selectedStockA && selectedStockB) {
        updateChart(selectedStockA, selectedStockB, selectedDuration, selectedDegree, value, true);
      }
    }
  };

  // 获取高亮的异常点标记
  const getAnomalyMarkPoints = () => {
    if (!chartData || !chartData.anomaly_info.anomalies.length) return [];
    
    return chartData.anomaly_info.anomalies.map(anomaly => ({
      name: '异常点',
      value: chartData.ratio[anomaly.index],
      xAxis: anomaly.index,
      yAxis: chartData.ratio[anomaly.index],
      itemStyle: {
        color: anomaly.z_score > 3 ? '#ff4d4f' : anomaly.z_score > 2.5 ? '#faad14' : '#1890ff'
      },
      symbol: 'circle',
      symbolSize: anomaly.z_score > 3 ? 10 : anomaly.z_score > 2.5 ? 8 : 6
    }));
  };

  // 比值图表配置
  const getRatioChartOption = () => {
    if (!chartData) return {};
    
    // 使用异常信息中的边界值
    const { upper_bound, lower_bound } = chartData.anomaly_info;
    
    return {
      title: {
        text: '股票价格比值分析',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params: any) {
          const dateIndex = params[0].dataIndex;
          const date = chartData.dates[dateIndex];
          let html = `<div><strong>${date}</strong></div>`;
          
          params.forEach((param: any) => {
            if (param.seriesName === '比值') {
              html += `<div>${param.seriesName}: ${param.value.toFixed(4)}</div>`;
              html += `<div>股票A价格: ${chartData.close_a[dateIndex].toFixed(2)}</div>`;
              html += `<div>股票B价格: ${chartData.close_b[dateIndex].toFixed(2)}</div>`;
            } else {
              html += `<div>${param.seriesName}: ${param.value.toFixed(4)}</div>`;
            }
          });
          
          // 检查是否为异常点
          const anomaly = chartData.anomaly_info.anomalies.find(a => a.index === dateIndex);
          if (anomaly) {
            html += `<div style="color: #ff4d4f;"><strong>异常点!</strong></div>`;
            html += `<div>Z分数: ${anomaly.z_score.toFixed(2)}</div>`;
            html += `<div>偏离度: ${(anomaly.deviation * 100).toFixed(2)}%</div>`;
          }
          
          return html;
        }
      },
      legend: {
        data: ['比值', '拟合线', '上边界', '下边界'],
        top: 30
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: chartData.dates,
        axisLabel: {
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          },
          interval: Math.floor(chartData.dates.length / 10)
        }
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: '比值',
          type: 'line',
          data: chartData.ratio,
          symbol: 'emptyCircle',
          symbolSize: 4,
          showSymbol: false,  // 默认不显示数据点
          // 只在鼠标经过时显示点
          emphasis: {
            focus: 'series',
            scale: true,
            symbolSize: 6
          },
          markPoint: {
            data: getAnomalyMarkPoints(),
            symbol: 'circle',
            symbolSize: 6,
            label: {
              show: false  // 不显示标签文字
            }
          },
          markArea: {
            silent: true,
            data: [
              [
                {
                  yAxis: upper_bound,
                  itemStyle: {
                    color: 'rgba(255, 77, 79, 0.1)'
                  }
                },
                {
                  yAxis: lower_bound
                }
              ]
            ]
          }
        },
        {
          name: '拟合线',
          type: 'line',
          data: chartData.fitting_line,
          lineStyle: {
            type: 'dashed',
            width: 2
          },
          symbol: 'none'
        },
        {
          name: '上边界',
          type: 'line',
          data: Array(chartData.ratio.length).fill(upper_bound),
          lineStyle: {
            type: 'dotted',
            color: '#ff4d4f'
          },
          symbol: 'none'
        },
        {
          name: '下边界',
          type: 'line',
          data: Array(chartData.ratio.length).fill(lower_bound),
          lineStyle: {
            type: 'dotted',
            color: '#ff4d4f'
          },
          symbol: 'none'
        }
      ]
    };
  };

  // 差值图表配置
  const getDeltaChartOption = () => {
    if (!chartData) return {};
    
    return {
      title: {
        text: '比值与拟合线差值',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params: any) {
          const dateIndex = params[0].dataIndex;
          const date = chartData.dates[dateIndex];
          const delta = params[0].value;
          const threshold = chartData.threshold * anomalyThreshold;
          
          let html = `<div><strong>${date}</strong></div>`;
          html += `<div>差值: ${delta.toFixed(4)}</div>`;
          html += `<div>阈值: ±${threshold.toFixed(4)}</div>`;
          
          // 检查是否为异常点
          const anomaly = chartData.anomaly_info.anomalies.find(a => a.index === dateIndex);
          if (anomaly) {
            html += `<div style="color: #ff4d4f;"><strong>异常点!</strong></div>`;
            html += `<div>Z分数: ${anomaly.z_score.toFixed(2)}</div>`;
          }
          
          return html;
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: chartData.dates,
        axisLabel: {
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          },
          interval: Math.floor(chartData.dates.length / 10)
        }
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          type: 'line',
          data: chartData.delta,
          showSymbol: false,  // 默认不显示数据点
          symbol: 'emptyCircle',
          symbolSize: 4,
          // 只在鼠标经过时显示点
          emphasis: {
            focus: 'series',
            scale: true,
            symbolSize: 6
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [{
                offset: 0,
                color: 'rgba(255, 77, 79, 0.3)' // 上半部分颜色
              }, {
                offset: 0.5,
                color: 'rgba(255, 255, 255, 0.1)'
              }, {
                offset: 1,
                color: 'rgba(24, 144, 255, 0.3)' // 下半部分颜色
              }]
            }
          },
          markLine: {
            silent: true,
            data: [
              {
                yAxis: 0,
                lineStyle: {
                  color: '#000',
                  type: 'solid',
                  width: 1
                }
              },
              {
                yAxis: chartData.threshold * anomalyThreshold,
                lineStyle: {
                  color: '#ff4d4f',
                  type: 'dashed'
                }
              },
              {
                yAxis: -chartData.threshold * anomalyThreshold,
                lineStyle: {
                  color: '#ff4d4f',
                  type: 'dashed'
                }
              }
            ]
          },
          markPoint: {
            data: chartData.anomaly_info.anomalies.map(anomaly => ({
              name: '异常点',
              value: chartData.delta[anomaly.index],
              xAxis: anomaly.index,
              yAxis: chartData.delta[anomaly.index],
              itemStyle: {
                color: anomaly.z_score > 3 ? '#ff4d4f' : anomaly.z_score > 2.5 ? '#faad14' : '#1890ff'
              },
              symbol: 'circle',
              symbolSize: anomaly.z_score > 3 ? 10 : anomaly.z_score > 2.5 ? 8 : 6,
              label: {
                show: false  // 不显示标签文字
              }
            }))
          }
        }
      ]
    };
  };

  // 添加股票选择前的验证函数
  const validateStockSelection = async (code: string): Promise<boolean> => {
    // 先检查是否在现有列表中
    const stockExists = stockList.some(stock => stock.code === code);
    
    if (stockExists) {
      return true; // 股票存在，无需爬取
    }
    
    // 显示加载提示
    message.loading({ content: `正在获取 ${code} 的股票数据...`, key: code });
    
    try {
      // 尝试从API获取股票信息
      console.log(`尝试获取不存在的股票: ${code}`);
      const response = await fetch(`http://localhost:8000/search_top_assets/${code}`);
      
      if (!response.ok) {
        message.error({ content: `股票 ${code} 获取失败`, key: code });
        return false;
      }
      
      const data = await response.json();
      
      // 检查是否成功获取到股票
      if (data.assets && data.assets.length > 0) {
        // 找到与代码匹配的股票
        const matchedStock = data.assets.find((stock: any) => stock.code === code);
        
        if (matchedStock) {
          // 添加到本地列表
          setStockList(prevList => {
            // 避免重复添加
            if (prevList.some(s => s.code === matchedStock.code)) {
              return prevList;
            }
            return [...prevList, matchedStock];
          });
          
          message.success({ content: `已获取 ${matchedStock.name} (${code}) 的数据`, key: code });
          return true;
        }
      }
      
      message.error({ content: `未找到股票 ${code}`, key: code });
      return false;
    } catch (error) {
      console.error("获取股票数据失败:", error);
      message.error({ content: `股票 ${code} 数据获取异常`, key: code });
      return false;
    }
  };

  // 股票A选择处理函数
  const handleStockAChange = async (value: string) => {
    // 检查是否与股票B相同
    if (value === selectedStockB) {
      message.error('股票A和股票B不能选择相同的股票');
      return;
    }
    
    // 验证股票是否存在，不存在则尝试爬取
    const isValid = await validateStockSelection(value);
    
    if (isValid) {
      setSelectedStockA(value);
      // 依靠useEffect处理更新
    }
  };

  // 股票B选择处理函数
  const handleStockBChange = async (value: string) => {
    // 检查是否与股票A相同
    if (value === selectedStockA) {
      message.error('股票A和股票B不能选择相同的股票');
      return;
    }
    
    // 验证股票是否存在，不存在则尝试爬取
    const isValid = await validateStockSelection(value);
    
    if (isValid) {
      setSelectedStockB(value);
      // 依靠useEffect处理更新
    }
  };

  // 时间跨度选择处理函数
  const handleDurationChange = (value: string) => {
    console.log("选择的时间跨度:", value);
    setSelectedDuration(value);
    // 直接传递新值给updateChart，不依赖于state更新
    if (selectedStockA && selectedStockB) {
      updateChart(selectedStockA, selectedStockB, value, selectedDegree, anomalyThreshold, false);
    }
  };

  // 拟合曲线选择处理函数
  const handleDegreeChange = (value: number) => {
    setSelectedDegree(value);
    // 直接传递新值给updateChart，不依赖于state更新
    if (selectedStockA && selectedStockB) {
      updateChart(selectedStockA, selectedStockB, selectedDuration, value, anomalyThreshold, false);
    }
  };

  return (
    <div className="ratio-analysis-container">
      <Card>
        <div style={{ marginBottom: 16 }}>
          <Space size="large" wrap>
            <Space>
              <Text>股票A:</Text>
              <Select
                style={{ width: 200 }}
                placeholder="选择股票A"
                showSearch
                optionFilterProp="children"
                value={selectedStockA}
                onChange={handleStockAChange}
              >
                {stockList.map(stock => (
                  <Select.Option key={stock.code} value={stock.code}>
                    {stock.name} ({stock.code})
                  </Select.Option>
                ))}
              </Select>
            </Space>
            
            <Space>
              <Text>股票B:</Text>
              <Select
                style={{ width: 200 }}
                placeholder="选择股票B"
                showSearch
                optionFilterProp="children"
                value={selectedStockB}
                onChange={handleStockBChange}
              >
                {stockList.map(stock => (
                  <Select.Option key={stock.code} value={stock.code}>
                    {stock.name} ({stock.code})
                  </Select.Option>
                ))}
              </Select>
            </Space>
            
            <Space>
              <Text>时间跨度:</Text>
              <Select
                style={{ width: 120 }}
                value={selectedDuration}
                onChange={handleDurationChange}
              >
                {DURATION_OPTIONS.map(option => (
                  <Select.Option key={option.backendValue} value={option.backendValue}>
                    {option.label}
                  </Select.Option>
                ))}
              </Select>
            </Space>
            
            <Space>
              <Text>拟合曲线:</Text>
              <Select
                style={{ width: 120 }}
                value={selectedDegree}
                onChange={handleDegreeChange}
              >
                <Select.Option value={1}>1次多项式</Select.Option>
                <Select.Option value={2}>2次多项式</Select.Option>
                <Select.Option value={3}>3次多项式</Select.Option>
                <Select.Option value={4}>4次多项式</Select.Option>
                <Select.Option value={5}>5次多项式</Select.Option>
              </Select>
            </Space>
            
            {chartData && (
              <Space>
                <Text>异常值阈值:</Text>
                <SliderSelector
                  title="异常值阈值"
                  min={1}
                  max={5}
                  step={0.1}
                  value={anomalyThreshold}
                  onChange={handleThresholdChange}
                />
              </Space>
            )}
          </Space>
        </div>

        {loading ? (
          <div style={{ textAlign: 'center', padding: '50px 0' }}>
            <Spin size="large" tip="加载中..." />
          </div>
        ) : chartData ? (
          <Tabs
            activeKey={activeTab}
            onChange={setActiveTab}
            items={[
              {
                key: '1',
                label: '比值分析',
                children: (
                  <div>
                    <ReactECharts
                      option={getRatioChartOption()}
                      style={{ height: 400 }}
                      notMerge={true}
                    />
                    {showDelta && (
                      <ReactECharts
                        option={getDeltaChartOption()}
                        style={{ height: 300, marginTop: 16 }}
                        notMerge={true}
                      />
                    )}
                  </div>
                )
              },
              {
                key: '2',
                label: '异常检测',
                children: chartData && (
                  <AnomalyDetection
                    anomalyInfo={chartData.anomaly_info}
                    dates={chartData.dates}
                    threshold={anomalyThreshold}
                    onThresholdChange={handleThresholdChange}
                  />
                )
              }
            ]}
          />
        ) : (
          <div style={{ textAlign: 'center', padding: '50px 0' }}>
            请选择两只股票进行比值分析
          </div>
        )}
      </Card>
    </div>
  );
};

export default RatioAnalysis; 