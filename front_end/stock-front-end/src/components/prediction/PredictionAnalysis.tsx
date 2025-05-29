import React, { useState, useEffect } from 'react';
import { Card, Spin, Typography, Space, Select, InputNumber, Button, Tabs, message, Alert, Form, Switch } from 'antd';
import ReactECharts from 'echarts-for-react';
import { useLocalStorage } from '../../LocalStorageContext';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

// 预测数据接口定义
interface PredictionData {
  dates: string[];  // 预测日期
  values: number[]; // 预测值
  upper_bound?: number[]; // 上置信区间 (可选)
  lower_bound?: number[]; // 下置信区间 (可选)
  historical_dates: string[]; // 历史日期（用于显示连续图表）
  historical_values: number[]; // 历史值
  performance?: {
    mse?: number; // 均方误差
    rmse?: number; // 均方根误差
    mae?: number; // 平均绝对误差
    r2?: number; // R方值
  };
  risk_level: 'low' | 'medium' | 'high'; // 风险级别评估
  forecast_trend: 'up' | 'down' | 'stable'; // 预测趋势
}

interface PredictionAnalysisProps {
  chartData: any | null; // 传入的历史数据
  stockA: string; // 股票A代码
  stockB: string; // 股票B代码
}

const PredictionAnalysis: React.FC<PredictionAnalysisProps> = ({ chartData, stockA, stockB }) => {
  // 储存持久化设置
  const [predictionDays, setPredictionDays] = useLocalStorage<number>('prediction-days', 30);
  const [confidenceLevel, setConfidenceLevel] = useLocalStorage<number>('confidence-level', 0.95);
  const [modelType, setModelType] = useLocalStorage<string>('model-type', 'enhanced_lstm');
  const [predictionData, setPredictionData] = useLocalStorage<PredictionData | null>('prediction-data', null);
  const [usePretrainedModel, setUsePretrainedModel] = useLocalStorage<boolean>('use-pretrained-model', true);
  
  // 本地状态
  const [loading, setLoading] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<string>('1');

  const modelOptions = [
    { value: 'enhanced_lstm', label: '增强版LSTM模型' },
    { value: 'lstm', label: '标准LSTM模型' }
  ];

  // 当股票数据变化时，清除预测数据
  useEffect(() => {
    if (chartData) {
      setPredictionData(null);
    }
  }, [chartData?.close_a, chartData?.close_b]);

  // 生成预测
  const generatePrediction = async () => {
    if (!chartData || !stockA || !stockB) {
      message.warning('请先选择股票并加载历史数据');
      return;
    }

    setLoading(true);
    try {
      let response;
      
      if (usePretrainedModel) {
        // 使用预训练模型API
        response = await fetch('http://localhost:8000/predict_ratio_model/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            code_a: stockA,
            code_b: stockB,
            days_to_predict: predictionDays,
            end_date: null, // 使用最新日期
          }),
        });
      } else {
        // 使用原有API
        response = await fetch('http://localhost:8000/predict_price_ratio/', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            code_a: stockA,
            code_b: stockB,
            ratio_data: chartData.ratio,
            dates: chartData.dates,
            prediction_days: predictionDays,
            confidence_level: confidenceLevel,
            model_type: modelType,
          }),
        });
      }
      
      const data = await response.json();
      setPredictionData(data);
      message.success('预测完成');
    } catch (error) {
      console.error('生成预测时出错:', error);
      message.error('预测失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 预测图表选项
  const getPredictionChartOption = () => {
    if (!predictionData || !chartData) return {};

    // 合并历史数据和预测数据以显示连续图表
    const allDates = [...predictionData.historical_dates, ...predictionData.dates];
    
    // 设置历史数据点的大小和颜色
    const historySymbolSize = 4;
    const predictionSymbolSize = 5;
    
    const series = [
      {
        name: '历史数据',
        type: 'line',
        symbolSize: historySymbolSize,
        symbol: 'circle',
        data: [...predictionData.historical_values, ...new Array(predictionData.values.length).fill(null)],
        itemStyle: {
          color: '#1890ff'
        }
      },
      {
        name: '预测数据',
        type: 'line',
        symbolSize: predictionSymbolSize,
        symbol: 'circle',
        data: [...new Array(predictionData.historical_values.length).fill(null), ...predictionData.values],
        itemStyle: {
          color: '#52c41a'
        },
        lineStyle: {
          width: 2,
          type: 'dashed'
        }
      }
    ];
    
    // 只有在有置信区间数据时添加相关系列
    if (predictionData.upper_bound && predictionData.lower_bound) {
      series.push(
        {
          name: '上置信区间',
          type: 'line',
          symbol: 'none',
          data: [...new Array(predictionData.historical_values.length).fill(null), ...predictionData.upper_bound],
          lineStyle: {
            width: 1,
            type: 'dotted',
            opacity: 0.6,
            color: '#faad14'
          },
          areaStyle: {
            opacity: 0.2,
            color: '#faad14'
          }
        },
        {
          name: '下置信区间',
          type: 'line',
          symbol: 'none',
          data: [...new Array(predictionData.historical_values.length).fill(null), ...predictionData.lower_bound],
          lineStyle: {
            width: 1,
            type: 'dotted',
            opacity: 0.6,
            color: '#faad14'
          }
        }
      );
    }
    
    // 动态设置图例
    const legendData = ['历史数据', '预测数据'];
    if (predictionData.upper_bound && predictionData.lower_bound) {
      legendData.push('上置信区间', '下置信区间');
    }
    
    return {
      title: {
        text: '价格比值预测分析',
        left: 'center'
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params: any) {
          const dateIndex = params[0].dataIndex;
          const date = allDates[dateIndex];
          let html = `<div><strong>${date}</strong></div>`;
          
          for (const param of params) {
            if (param.seriesName === '历史数据') {
              html += `<div>${param.seriesName}: ${param.value ? param.value.toFixed(4) : '-'}</div>`;
            } else if (param.seriesName === '预测数据') {
              html += `<div>${param.seriesName}: ${param.value ? param.value.toFixed(4) : '-'}</div>`;
            } else if (param.seriesName === '上置信区间' || param.seriesName === '下置信区间') {
              html += `<div>${param.seriesName}: ${param.value ? param.value.toFixed(4) : '-'}</div>`;
            }
          }
          return html;
        }
      },
      legend: {
        data: legendData,
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
        boundaryGap: false,
        data: allDates,
        axisLabel: {
          formatter: function(value: string) {
            return value.substring(5); // 只显示月-日
          },
          rotate: 45,
          margin: 8
        }
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      dataZoom: [
        {
          type: 'inside',
          start: 50, // 默认显示后半部分（包括预测部分）
          end: 100
        },
        {
          start: 50,
          end: 100
        }
      ],
      series: series
    };
  };

  // 获取风险级别显示
  const getRiskLevelDisplay = () => {
    if (!predictionData) return null;
    
    const { risk_level, forecast_trend } = predictionData;
    
    let riskColor = '#52c41a'; // 默认绿色
    let trendIcon = '→';
    let trendColor = '#1890ff';
    
    if (risk_level === 'high') {
      riskColor = '#ff4d4f';
    } else if (risk_level === 'medium') {
      riskColor = '#faad14';
    }
    
    if (forecast_trend === 'up') {
      trendIcon = '↑';
      trendColor = '#52c41a';
    } else if (forecast_trend === 'down') {
      trendIcon = '↓';
      trendColor = '#ff4d4f';
    }
    
    return (
      <Space direction="vertical" style={{ width: '100%' }}>
        <Alert
          message={
            <div>
              预测风险级别: 
              <Text strong style={{ color: riskColor }}>
                {risk_level === 'high' ? ' 高风险' : risk_level === 'medium' ? ' 中等风险' : ' 低风险'}
              </Text>
            </div>
          }
          type={risk_level === 'high' ? 'error' : risk_level === 'medium' ? 'warning' : 'success'}
          showIcon
        />
        <Alert
          message={
            <div>
              预测趋势方向: 
              <Text strong style={{ color: trendColor }}>
                {' '}{trendIcon} {forecast_trend === 'up' ? '上升' : forecast_trend === 'down' ? '下降' : '稳定'}
              </Text>
            </div>
          }
          type="info"
          showIcon
        />
      </Space>
    );
  };

  // 渲染模型性能指标
  const renderPerformanceMetrics = () => {
    if (!predictionData || !predictionData.performance) return null;
    
    const { performance } = predictionData;
    
    return (
      <Card size="small" title="模型性能指标" style={{ marginTop: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div>
            <div><Text strong>均方误差 (MSE):</Text> {performance.mse?.toFixed(4) || '未提供'}</div>
            <div><Text strong>均方根误差 (RMSE):</Text> {performance.rmse?.toFixed(4) || '未提供'}</div>
          </div>
          <div>
            <div><Text strong>平均绝对误差 (MAE):</Text> {performance.mae?.toFixed(4) || '未提供'}</div>
            <div><Text strong>决定系数 (R²):</Text> {performance.r2?.toFixed(4) || '未提供'}</div>
          </div>
        </div>
      </Card>
    );
  };

  return (
    <Card 
      title={
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Title level={5} style={{ margin: 0 }}>价格比值预测</Title>
          <div>
            <Button 
              type="primary" 
              onClick={generatePrediction} 
              loading={loading}
              disabled={!chartData}
            >
              生成预测
            </Button>
          </div>
        </div>
      }
      style={{ marginTop: 16 }}
    >
      <Spin spinning={loading}>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="预测设置" key="1">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Card size="small" title="预测参数">
                {/* 模型选择开关 */}
                <div style={{ marginBottom: 16 }}>
                  <Form.Item label="使用预训练模型">
                    <Switch 
                      checked={usePretrainedModel}
                      onChange={(checked) => setUsePretrainedModel(checked)}
                    />
                    <Text type="secondary" style={{ marginLeft: 8 }}>
                      {usePretrainedModel ? '使用预训练LSTM模型' : '使用实时训练模型'}
                    </Text>
                  </Form.Item>
                </div>
              
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                  <div style={{ width: '48%' }}>
                    <div style={{ marginBottom: 8 }}><Text>预测天数</Text></div>
                    <InputNumber 
                      min={5} 
                      max={90} 
                      value={predictionDays} 
                      onChange={(value) => setPredictionDays(value as number)} 
                      style={{ width: '100%' }}
                    />
                  </div>
                  {!usePretrainedModel && (
                    <div style={{ width: '48%' }}>
                      <div style={{ marginBottom: 8 }}><Text>置信水平</Text></div>
                      <Select 
                        value={confidenceLevel} 
                        onChange={(value) => setConfidenceLevel(value)}
                        style={{ width: '100%' }}
                        disabled={usePretrainedModel}
                      >
                        <Option value={0.9}>90%</Option>
                        <Option value={0.95}>95%</Option>
                        <Option value={0.99}>99%</Option>
                      </Select>
                    </div>
                  )}
                </div>
                
                {!usePretrainedModel && (
                  <div style={{ marginBottom: 16 }}>
                    <Form.Item label="预测模型">
                      <Select
                        value={modelType}
                        onChange={setModelType}
                        options={modelOptions}
                        style={{ width: 170 }}
                        disabled={usePretrainedModel}
                      />
                    </Form.Item>
                  </div>
                )}
              </Card>
              
              {predictionData && getRiskLevelDisplay()}
            </Space>
          </TabPane>
          
          <TabPane tab="预测结果" key="2" disabled={!predictionData}>
            {predictionData ? (
              <>
                <div style={{ height: 400 }}>
                  <ReactECharts 
                    option={getPredictionChartOption()} 
                    style={{ height: '100%' }} 
                  />
                </div>
                {renderPerformanceMetrics()}
              </>
            ) : (
              <div style={{ textAlign: 'center', padding: '20px 0' }}>
                <Text type="secondary">请先生成预测数据</Text>
              </div>
            )}
          </TabPane>
        </Tabs>
      </Spin>
    </Card>
  );
};

export default PredictionAnalysis; 