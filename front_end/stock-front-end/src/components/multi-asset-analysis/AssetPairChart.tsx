import React, { useEffect, useRef } from 'react';
import { Card, Row, Col, Statistic, Tag, Divider, Tooltip, Switch, Space } from 'antd';
import * as echarts from 'echarts';
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { debounce } from 'lodash';

interface AssetPairChartProps {
  data: any;
  assetA: string;
  assetB: string;
  thresholdMultiplier: number;
  showFittingCurve?: boolean;
  showAnomalies?: boolean;
}

const AssetPairChart: React.FC<AssetPairChartProps> = ({
  data,
  assetA,
  assetB,
  thresholdMultiplier,
  showFittingCurve: initialShowFittingCurve = true,
  showAnomalies: initialShowAnomalies = true
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);
  const [showAnomalies, setShowAnomalies] = React.useState<boolean>(initialShowAnomalies);
  const [showFittingCurve, setShowFittingCurve] = React.useState<boolean>(initialShowFittingCurve);

  // 处理异常点函数 - 移动到组件顶层作用域
  const processAnomalyPoints = (dates, ratio, fittingLine, anomalies) => {
    const positiveAnomalies = []; // 正差值点（比值高于拟合线，差值为正）
    const negativeAnomalies = []; // 负差值点（比值低于拟合线，差值为负）

    if (!anomalies || !Array.isArray(anomalies)) return { positiveAnomalies, negativeAnomalies };

    try {
      anomalies.forEach(anomaly => {
        const index = anomaly.index;
        if (index < dates.length && index < ratio.length && index < fittingLine.length) {
          const date = dates[index];
          const ratioValue = ratio[index];
          const fittingValue = fittingLine[index];
          const diff = ratioValue - fittingValue;

          if (diff > 0) {
            // 正差值点（比值高于拟合线，差值为正）
            positiveAnomalies.push([date, ratioValue]);
          } else {
            // 负差值点（比值低于拟合线，差值为负）
            negativeAnomalies.push([date, ratioValue]);
          }
        }
      });
    } catch (error) {
      console.error('处理异常点时发生错误:', error);
    }

    return { positiveAnomalies, negativeAnomalies };
  };

  useEffect(() => {
    const handleResize = debounce(() => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    }, 300);

    window.addEventListener('resize', handleResize);

    if (!chartRef.current || !data || !data.dates || !data.ratio) return;

    const chart = echarts.init(chartRef.current);
    chartInstance.current = chart;

    try {
      // 解构数据
      const {
        dates = [],
        ratio = [],
        fitted_curve = [],
        upper_threshold = [],
        lower_threshold = [],
        anomaly_info = {}
      } = data;

      // 获取异常点信息
      const anomalies = anomaly_info.is_anomaly ?
        anomaly_info.is_anomaly.map((isAnomaly, idx) => {
          if (isAnomaly) {
            return {
              index: idx,
              z_score: anomaly_info.z_scores ? anomaly_info.z_scores[idx] : 0
            };
          }
          return null;
        }).filter(Boolean) : [];

      // 处理异常点
      const { positiveAnomalies, negativeAnomalies } = processAnomalyPoints(dates, ratio, fitted_curve, anomalies);

      // 更新图表
      updateChart(positiveAnomalies, negativeAnomalies);
    } catch (error) {
      console.error('Error updating chart:', error);
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, [data, showAnomalies, showFittingCurve]);

  const updateChart = (positiveAnomalies = [], negativeAnomalies = []) => {
    if (!chartInstance.current || !data) return;

    const {
      dates = [],
      ratio = [],
      fitted_curve = [],
      upper_threshold = [],
      lower_threshold = []
    } = data;

    // 构建图表选项
    const option = {
      grid: {
        left: '10%',
        right: '5%',
        top: '12%',
        bottom: '15%'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        }
      },
      legend: {
        data: ['价格比值', '拟合曲线', '上限阈值', '下限阈值', '绿色区域', '红色区域']
      },
      xAxis: {
        type: 'category',
        data: dates,
        scale: true,
        boundaryGap: false,
        axisLabel: {
          formatter: (value) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear().toString().substr(-2)}`;
          },
          interval: Math.floor(dates.length / 8)
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
          name: '价格比值',
          type: 'line',
          data: dates.map((date, i) => [date, parseFloat(ratio[i]).toFixed(4)]),
          showSymbol: false,
          lineStyle: {
            width: 1
          }
        },
        ...(showFittingCurve ? [
          {
            name: '拟合曲线',
            type: 'line',
            data: dates.map((date, i) => [date, parseFloat(fitted_curve[i].toFixed(4))]),
            showSymbol: false,
            smooth: true,
            lineStyle: {
              width: 2,
              type: 'dashed',
              color: '#722ED1'
            }
          }
        ] : []),
        ...(showAnomalies ? [
          {
            name: '上限阈值',
            type: 'line',
            data: dates.map((date, i) => [date, parseFloat(upper_threshold[i].toFixed(4))]),
            showSymbol: false,
            lineStyle: {
              width: 1,
              type: 'dotted',
              color: '#FFB200',
              opacity: 0.5
            }
          },
          {
            name: '下限阈值',
            type: 'line',
            data: dates.map((date, i) => [date, parseFloat(lower_threshold[i].toFixed(4))]),
            showSymbol: false,
            lineStyle: {
              width: 1,
              type: 'dotted',
              color: '#FFB200',
              opacity: 0.5
            }
          }
        ] : []),
        ...(showAnomalies && positiveAnomalies.length ? [
          {
            name: '绿色区域',
            type: 'scatter',
            data: positiveAnomalies,
            symbolSize: 8,
            itemStyle: {
              color: '#91CC75'  // 绿色，与矩阵颜色保持一致
            }
          }
        ] : []),
        ...(showAnomalies && negativeAnomalies.length ? [
          {
            name: '红色区域',
            type: 'scatter',
            data: negativeAnomalies,
            symbolSize: 8,
            itemStyle: {
              color: '#EE6666'  // 红色，与矩阵颜色保持一致
            }
          }
        ] : [])
      ]
    };

    chartInstance.current.setOption(option, true);
  };

  // 如果没有数据，显示加载中
  if (!data) {
    return (
      <Card title={`${assetA}/${assetB} 价格比值分析`} style={{ height: '100%', minHeight: 400 }}>
        <div style={{ textAlign: 'center', padding: 100 }}>加载中...</div>
      </Card>
    );
  }

  const {
    signal_strength = 0,
    current_ratio = 0,
    ma10_ratio = 0,
    historical_percentile = 50,
    recommendation = '观望'
  } = data.analysis || {};


  const getRecommendationColor = (rec: string, strengthValue: number) => {
    if (strengthValue > 0) return '#EE6666';  // 红色，表示做多A做空B
    if (strengthValue < 0) return '#91CC75';  // 绿色，表示做空A做多B
    return 'orange';  // 观望，黄色
  };

  // 获取信号强度的描述和颜色
  const getSignalStrengthDescription = (strength: number) => {
    const absStrength = Math.abs(strength);
    if (absStrength >= 80) return '极强';
    if (absStrength >= 60) return '很强';
    if (absStrength >= 40) return '中等';
    if (absStrength >= 20) return '较弱';
    return '微弱';
  };


  const getSignalStrengthColor = (strength: number) => {
    if (strength > 0) return '#EE6666';  // 做多A做空B，红色
    if (strength < 0) return '#91CC75';  // 做空A做多B，绿色
    return '#666';  // 无信号，灰色
  };

  const getSimpleRecommendation = (strength: number) => {
    if (Math.abs(strength) < 20) return '观望';
    if (strength > 0) return `做多${assetA}做空${assetB}`;  // 做多A做空B
    return `做空${assetA}做多${assetB}`;  // 做空A做多B
  };

  // 异常点标记
  const renderAnomalyMarkers = () => {
    if (!data || !data.anomaly_info) return null;

    return (
      <div style={{ marginTop: 16 }}>
        <Row>
          <Col span={12}>
            <div className="anomaly-marker">
              <span className="marker" style={{ backgroundColor: '#91CC75', display: 'inline-block', width: 12, height: 12, marginRight: 4 }}></span>
              <span className="text">绿色区域: 做空{assetA}做多{assetB}</span>
            </div>
          </Col>
          <Col span={12}>
            <div className="anomaly-marker">
              <span className="marker" style={{ backgroundColor: '#EE6666', display: 'inline-block', width: 12, height: 12, marginRight: 4 }}></span>
              <span className="text">红色区域: 做多{assetA}做空{assetB}</span>
            </div>
          </Col>
        </Row>
        <div style={{ fontSize: '0.85em', color: '#888', marginTop: 8 }}>
          注: 绿色区域表示比值高于拟合线，红色区域表示比值低于拟合线
        </div>
      </div>
    );
  };

  // 修正: 简化版的交易建议
  const simpleRecommendation = getSimpleRecommendation(signal_strength);

  return (
    <Card
      title={`${assetA}/${assetB} 价格比值分析`}
      style={{ height: '100%', minHeight: 400 }}
      extra={
        <Space>
          <Tooltip title="显示/隐藏异常点">
            <Switch
              checkedChildren="异常点"
              unCheckedChildren="异常点"
              checked={showAnomalies}
              onChange={setShowAnomalies}
            />
          </Tooltip>
          <Tooltip title="显示/隐藏拟合曲线">
            <Switch
              checkedChildren="拟合曲线"
              unCheckedChildren="拟合曲线"
              checked={showFittingCurve}
              onChange={setShowFittingCurve}
            />
          </Tooltip>
        </Space>
      }
    >
      <Row gutter={16} style={{ marginBottom: 16 }}>
        <Col span={6}>
          <Statistic
            title="当前比值"
            value={current_ratio}
            precision={4}
            valueStyle={{ color: current_ratio > ma10_ratio ? '#3f8600' : '#cf1322' }}
            prefix={current_ratio > ma10_ratio ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title="90日均值"
            value={ma10_ratio}
            precision={4}
          />
        </Col>
        <Col span={6}>
          <Statistic
            title={
              <span>
                历史百分位 
                <Tooltip title="当前比值在历史数据中的百分位，值越高表示当前比值相对历史越高">
                  <InfoCircleOutlined style={{ marginLeft: 5 }} />
                </Tooltip>
              </span>
            }
            value={historical_percentile}
            precision={1}
            suffix="%"
          />
        </Col>
        <Col span={6}>
          <Statistic
            title={
              <span>
                信号强度 
                <Tooltip title="当前交易信号的强度，正值表示做多第一个资产做空第二个资产，负值表示相反">
                  <InfoCircleOutlined style={{ marginLeft: 5 }} />
                </Tooltip>
              </span>
            }
            value={signal_strength}
            precision={0}
            valueStyle={{ color: getSignalStrengthColor(signal_strength) }}
            suffix={` (${getSignalStrengthDescription(signal_strength)})`}
          />
        </Col>
      </Row>

      <Divider style={{ margin: '12px 0' }} />

      <div style={{ marginBottom: 16 }}>
        <span style={{ marginRight: 8 }}>交易建议:</span>
        <Tag color={getRecommendationColor(simpleRecommendation, signal_strength)} style={{ fontSize: 16, padding: '4px 8px' }}>
          {simpleRecommendation}
        </Tag>
      </div>

      <div style={{ height: 400, width: '100%' }} ref={chartRef}></div>

      {renderAnomalyMarkers()}
    </Card>
  );
};

export default AssetPairChart; 