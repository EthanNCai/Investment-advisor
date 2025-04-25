import React from 'react';
import { Card, Alert, Descriptions, Progress, Space, Typography, Divider, Row, Col, Statistic, Tag, Table, Empty } from 'antd';
import { InfoCircleOutlined, WarningOutlined, CheckCircleOutlined, ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { Signal } from './InvestmentSignal';

const { Text, Paragraph, Title } = Typography;

interface CurrentPositionProps {
  currentPositionInfo: any | null;
  signals: Signal[];
}

const CurrentPositionAnalysis: React.FC<CurrentPositionProps> = ({ 
  currentPositionInfo, 
  signals 
}) => {
  if (!currentPositionInfo) {
    return (
      <Alert
        message="数据不足"
        description="无法分析当前位置，请确保选择了有效的股票对。"
        type="warning"
        showIcon
      />
    );
  }
  
  const { 
    current_ratio, 
    nearest_signals,
    similarity_score, 
    percentile, 
    is_extreme,
    z_score,
    deviation_from_trend,
    volatility_level,
    historical_signal_pattern, 
    recommendation 
  } = currentPositionInfo;
  
  // 获取波动性级别对应的标签颜色
  const getVolatilityColor = (level: string) => {
    switch (level) {
      case 'low': return 'success';
      case 'medium': return 'warning';
      case 'high': return 'error';
      default: return 'default';
    }
  };
  
  // 获取波动性级别的中文描述
  const getVolatilityText = (level: string) => {
    switch (level) {
      case 'low': return '低';
      case 'medium': return '中';
      case 'high': return '高';
      default: return '未知';
    }
  };

  // 获取历史信号模式的描述与颜色
  const getPatternTag = (pattern: string | null) => {
    if (!pattern) return null;

    let color = 'default';
    switch (pattern) {
      case '连续超买':
        color = 'error';
        break;
      case '连续超卖':
        color = 'success';
        break;
      case '震荡切换':
        color = 'warning';
        break;
      case '混合模式':
        color = 'processing';
        break;
    }

    return <Tag color={color}>{pattern}</Tag>;
  };

  // 相似信号表格列定义
  const similarSignalColumns = [
    {
      title: '日期',
      dataIndex: 'date',
      key: 'date',
    },
    {
      title: '比值',
      dataIndex: 'ratio',
      key: 'ratio',
      render: (ratio: number) => ratio.toFixed(4)
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'positive' ? 'red' : 'green'}>
          {type === 'positive' ? '超买' : '超卖'}
        </Tag>
      )
    },
    {
      title: '强度',
      dataIndex: 'strength',
      key: 'strength',
      render: (strength: string) => {
        let color = 'blue';
        if (strength === 'strong') color = 'red';
        else if (strength === 'medium') color = 'orange';
        return <Tag color={color}>{strength === 'strong' ? '强' : strength === 'medium' ? '中' : '弱'}</Tag>;
      }
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      render: (similarity: number) => (
        <Progress 
          percent={similarity * 100} 
          size="small" 
          format={percent => `${percent?.toFixed(0)}%`}
          strokeColor={similarity > 0.8 ? '#52c41a' : similarity > 0.6 ? '#faad14' : '#1890ff'}
        />
      )
    }
  ];
  
  // 生成历史分布图表选项
  const getHistogramOption = () => {
    if (!signals.length) {
      return {
        title: {
          text: '暂无历史比值数据',
          left: 'center'
        }
      };
    }
    
    // 提取所有比值用于统计
    const ratioValues = signals.map(s => s.ratio);
    
    // 计算比值范围
    const minRatio = Math.min(...ratioValues, current_ratio) * 0.95;
    const maxRatio = Math.max(...ratioValues, current_ratio) * 1.05;
    
    // 计算直方图的区间数（根据数据量动态调整）
    const binCount = Math.min(20, Math.max(10, Math.floor(Math.sqrt(ratioValues.length))));
    
    // 计算每个区间的宽度
    const binWidth = (maxRatio - minRatio) / binCount;
    
    // 创建区间
    const bins = Array(binCount).fill(0).map((_, i) => ({
      min: minRatio + i * binWidth,
      max: minRatio + (i + 1) * binWidth,
      count: 0
    }));
    
    // 统计每个区间的数量
    ratioValues.forEach(ratio => {
      const binIndex = Math.min(binCount - 1, Math.max(0, Math.floor((ratio - minRatio) / binWidth)));
      bins[binIndex].count++;
    });
    
    // 找到当前比值落在哪个区间
    const currentBinIndex = Math.min(binCount - 1, Math.max(0, Math.floor((current_ratio - minRatio) / binWidth)));
    
    // 创建图表数据
    const xAxisData = bins.map((bin, index) => {
      const midpoint = (bin.min + bin.max) / 2;
      return midpoint.toFixed(4);
    });
    
    const seriesData = bins.map((bin, index) => {
      return {
        value: bin.count,
        itemStyle: {
          color: index === currentBinIndex ? '#ff4d4f' : '#5470c6'
        }
      };
    });
    
    return {
      title: {
        text: '比值历史分布',
        left: 'center',
        top: 10
      },
      tooltip: {
        trigger: 'item',
        formatter: function(params: any) {
          if (params.seriesType === 'bar') {
            const bin = bins[params.dataIndex];
            return `比值区间: ${bin.min.toFixed(4)} - ${bin.max.toFixed(4)}<br/>数量: ${params.value}`;
          }
          return params.name;
        }
      },
      grid: {
        top: 60,
        left: 60,
        right: 60,
        bottom: 60
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        name: '价格比值',
        nameLocation: 'middle',
        nameGap: 30,
        axisLabel: {
          rotate: 45,
          formatter: function(value: string) {
            // 保留4位小数
            return parseFloat(value).toFixed(4);
          }
        }
      },
      yAxis: {
        type: 'value',
        name: '出现频次',
        nameLocation: 'middle',
        nameGap: 40
      },
      series: [
        {
          name: '比值分布',
          type: 'bar',
          data: seriesData,
          markLine: {
            symbol: ['none', 'none'],
            silent: true,
            data: [
              {
                name: '当前比值',
                xAxis: currentBinIndex,
                lineStyle: {
                  color: '#ff4d4f',
                  width: 2,
                  type: 'solid'
                },
                label: {
                  show: true,
                  formatter: '当前比值',
                  position: 'insideEndTop',
                  color: '#ff4d4f',
                  fontSize: 12,
                  fontWeight: 'bold'
                }
              }
            ]
          }
        }
      ]
    };
  };
  
  // 生成Z分数仪表盘图表
  const getZScoreGaugeOption = () => {
    const zValue = z_score || 0;
    const absZScore = Math.abs(zValue);
    
    // 确定颜色区域
    const colorStops = [
      { offset: 0, color: '#91cc75' },    // 绿色区域（中性）
      { offset: 0.4, color: '#91cc75' },  // 绿色区域（中性）
      { offset: 0.6, color: '#fac858' },  // 黄色区域（中等偏离）
      { offset: 0.8, color: '#ee6666' },  // 红色区域（高偏离）
      { offset: 1, color: '#73c0de' }     // 蓝色区域（极端偏离）
    ];
    
    return {
      series: [
        {
          type: 'gauge',
          min: -4,
          max: 4,
          splitNumber: 8,
          radius: '100%',
          axisLine: {
            lineStyle: {
              width: 30,
              color: [
                [0.2, '#91cc75'],  // 绿色区域（中性）
                [0.8, '#fac858'],  // 黄色区域（中等偏离）
                [1, '#ee6666']     // 红色区域（高偏离）
              ]
            }
          },
          pointer: {
            icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
            length: '12%',
            width: 20,
            offsetCenter: [0, '-60%'],
            itemStyle: {
              color: 'auto'
            }
          },
          axisTick: {
            length: 12,
            lineStyle: {
              color: 'auto',
              width: 2
            }
          },
          splitLine: {
            length: 20,
            lineStyle: {
              color: 'auto',
              width: 3
            }
          },
          axisLabel: {
            color: '#464646',
            fontSize: 14,
            distance: -60,
            formatter: function(value: number) {
              if (value === 0) return '0';
              if (value === -2 || value === 2) return value.toString();
              if (value === -4 || value === 4) return value.toString();
              return '';
            }
          },
          title: {
            offsetCenter: [0, '-20%'],
            fontSize: 14
          },
          detail: {
            fontSize: 20,
            offsetCenter: [0, '0%'],
            valueAnimation: true,
            formatter: function(value: number) {
              return value.toFixed(2);
            },
            color: 'auto'
          },
          data: [
            {
              value: zValue,
              name: 'Z分数'
            }
          ]
        }
      ]
    };
  };
  
  // 生成偏离趋势图表
  const getDeviationChartOption = () => {
    // 没有偏离趋势数据时显示默认图表
    if (deviation_from_trend === null) {
      return {
        title: {
          text: '暂无偏离趋势数据',
          left: 'center'
        }
      };
    }
    
    const deviationValue = deviation_from_trend;
    const isPositive = deviationValue > 0;
    
    // 颜色根据偏离方向和大小变化
    let color = '#1890ff'; // 默认蓝色
    if (Math.abs(deviationValue) > 10) {
      color = isPositive ? '#ff4d4f' : '#52c41a'; // 大幅偏离：正偏离红色，负偏离绿色
    } else if (Math.abs(deviationValue) > 5) {
      color = isPositive ? '#faad14' : '#1890ff'; // 中等偏离：正偏离黄色，负偏离蓝色
    }
    
    return {
      series: [
        {
          type: 'gauge',
          min: -20,
          max: 20,
          splitNumber: 8,
          radius: '100%',
          axisLine: {
            lineStyle: {
              width: 30,
              color: [
                [0.25, '#52c41a'],  // 绿色区域（负偏离）
                [0.5, '#1890ff'],   // 蓝色区域（中性）
                [0.75, '#faad14'],  // 黄色区域（轻微正偏离）
                [1, '#ff4d4f']      // 红色区域（大幅正偏离）
              ]
            }
          },
          pointer: {
            icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
            length: '12%',
            width: 20,
            offsetCenter: [0, '-60%'],
            itemStyle: {
              color
            }
          },
          axisTick: {
            length: 12,
            lineStyle: {
              color: 'auto',
              width: 2
            }
          },
          splitLine: {
            length: 20,
            lineStyle: {
              color: 'auto',
              width: 3
            }
          },
          axisLabel: {
            color: '#464646',
            fontSize: 14,
            distance: -60,
            formatter: function(value: number) {
              if (value === 0) return '0%';
              if (value === -10 || value === 10) return value + '%';
              if (value === -20 || value === 20) return value + '%';
              return '';
            }
          },
          title: {
            offsetCenter: [0, '-20%'],
            fontSize: 14
          },
          detail: {
            fontSize: 20,
            offsetCenter: [0, '0%'],
            valueAnimation: true,
            formatter: function(value: number) {
              return value.toFixed(2) + '%';
            },
            color
          },
          data: [
            {
              value: deviationValue,
              name: '偏离趋势线'
            }
          ]
        }
      ]
    };
  };
  
  // 生成百分位图表
  const getPercentileChartOption = () => {
    const percentileValue = parseFloat(String(percentile)) * 100 || 0;
    
    return {
      series: [
        {
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          min: 0,
          max: 100,
          radius: '100%',
          itemStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 1,
              y2: 0,
              colorStops: [
                {
                  offset: 0,
                  color: '#52c41a'  // 左侧绿色
                },
                {
                  offset: 0.5,
                  color: '#1890ff'  // 中间蓝色
                },
                {
                  offset: 1,
                  color: '#ff4d4f'  // 右侧红色
                }
              ]
            }
          },
          progress: {
            show: true,
            roundCap: true,
            width: 18
          },
          pointer: {
            icon: 'path://M2090.36389,615.30999 L2090.36389,615.30999 C2091.48372,615.30999 2092.40383,616.23010 2092.40383,617.34993 L2092.40383,617.34993 C2092.40383,618.46975 2091.48372,619.38987 2090.36389,619.38987 L2090.36389,619.38987 C2089.24406,619.38987 2088.32395,618.46975 2088.32395,617.34993 L2088.32395,617.34993 C2088.32395,616.23010 2089.24406,615.30999 2090.36389,615.30999 Z',
            length: '75%',
            width: 16,
            offsetCenter: [0, '5%']
          },
          axisLine: {
            roundCap: true,
            lineStyle: {
              width: 18
            }
          },
          axisTick: {
            splitNumber: 2,
            lineStyle: {
              width: 2,
              color: '#999'
            }
          },
          splitLine: {
            length: 12,
            lineStyle: {
              width: 3,
              color: '#999'
            }
          },
          axisLabel: {
            distance: 30,
            color: '#999',
            fontSize: 14
          },
          title: {
            show: false
          },
          detail: {
            valueAnimation: true,
            width: '60%',
            lineHeight: 40,
            height: 40,
            borderRadius: 8,
            offsetCenter: [0, '35%'],
            fontSize: 18,
            fontWeight: 'bolder',
            formatter: `${percentileValue.toFixed(2)}%`,
            color: 'auto'
          },
          data: [
            {
              value: percentileValue
            }
          ]
        }
      ]
    };
  };

  // 获取警告提示组件
  const getPositionAlert = () => {
    if (is_extreme) {
      return (
        <Alert
          message="价格比值异常警告"
          description={`当前比值处于历史极端位置，具有较强的套利机会，Z分数为 ${z_score?.toFixed(2)}。`}
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          style={{ marginBottom: 16 }}
        />
      );
    } else if (similarity_score && similarity_score > 0.9) {
      return (
        <Alert
          message="历史信号高度吻合"
          description="当前比值与历史投资信号高度相似，可参考历史信号表现判断后市走势。"
          type="info"
          showIcon
          icon={<InfoCircleOutlined />}
          style={{ marginBottom: 16 }}
        />
      );
    } else if (volatility_level === 'high') {
      return (
        <Alert
          message="高波动性警告"
          description="当前价格比值波动幅度较大，交易时需谨慎控制风险。"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />
      );
    } else {
      return (
        <Alert
          message="价格比值正常"
          description="当前比值在正常范围内，未检测到明显异常。"
          type="success"
          showIcon
          icon={<CheckCircleOutlined />}
          style={{ marginBottom: 16 }}
        />
      );
    }
  };
  
  return (
    <div className="current-position-analysis">
      {getPositionAlert()}
      
      <Card title="当前比值位置分析" bordered={false}>
        <Row gutter={[16, 16]}>
          <Col span={24} md={12}>
            <Card type="inner" title="基本信息">
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="当前比值">{current_ratio.toFixed(4)}</Descriptions.Item>
                <Descriptions.Item label="历史分位">
                  {percentile !== null ? (percentile * 100).toFixed(2) + '%' : '未知'}
                </Descriptions.Item>
                <Descriptions.Item label="Z得分">
                  {z_score !== null ? (
                    <>
                      {z_score.toFixed(2)} 
                      {z_score > 0 ? 
                        <ArrowUpOutlined style={{ color: '#ff4d4f', marginLeft: 4 }} /> : 
                        <ArrowDownOutlined style={{ color: '#52c41a', marginLeft: 4 }} />
                      }
                    </>
                  ) : '未知'}
                </Descriptions.Item>
                <Descriptions.Item label="波动性">
                  {volatility_level ? 
                    <Tag color={getVolatilityColor(volatility_level)}>
                      {getVolatilityText(volatility_level)}波动
                    </Tag> : '未知'
                  }
                </Descriptions.Item>
                <Descriptions.Item label="偏离趋势">
                  {deviation_from_trend !== null ? deviation_from_trend.toFixed(2) + '%' : '未知'}
                </Descriptions.Item>
                <Descriptions.Item label="历史信号模式">
                  {getPatternTag(historical_signal_pattern)}
                </Descriptions.Item>
              </Descriptions>
            </Card>
          </Col>
          
          <Col span={24} md={12}>
            <Card type="inner" title="位置评估">
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <ReactECharts 
                    option={getPercentileChartOption()} 
                    style={{ height: 180 }}
                  />
                  <div style={{ textAlign: 'center', marginTop: 8 }}>
                    <Text strong>历史分位百分比</Text>
                  </div>
                </Col>
                <Col span={12}>
                  <ReactECharts 
                    option={getZScoreGaugeOption()} 
                    style={{ height: 180 }}
                  />
                  <div style={{ textAlign: 'center', marginTop: 8 }}>
                    <Text strong>Z分数</Text>
                  </div>
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
        
        <Divider />
        
        <Row gutter={[16, 16]}>
          <Col span={24} md={12}>
            <Card type="inner" title="历史分布">
              <ReactECharts 
                option={getHistogramOption()} 
                style={{ height: 280 }}
              />
            </Card>
          </Col>
          
          <Col span={24} md={12}>
            <Card type="inner" title="趋势偏离">
              <ReactECharts 
                option={getDeviationChartOption()} 
                style={{ height: 280 }}
              />
            </Card>
          </Col>
        </Row>
        
        <Divider />
        
        <Row gutter={[16, 16]}>
          <Col span={24}>
            <Card type="inner" title="相似历史信号">
              {nearest_signals && nearest_signals.length > 0 ? (
                <Table 
                  dataSource={nearest_signals.map((signal, index) => ({...signal, key: index}))} 
                  columns={similarSignalColumns} 
                  pagination={false} 
                  size="small"
                />
              ) : (
                <Empty description="暂无相似历史信号" />
              )}
            </Card>
          </Col>
        </Row>
        
        <Divider />
        
        <Card type="inner" title="投资建议">
          <Paragraph style={{ fontSize: 16 }}>
            <blockquote style={{ padding: '12px 16px', borderLeft: '4px solid #1890ff', background: '#f0f5ff' }}>
              {recommendation || '暂无投资建议'}
            </blockquote>
          </Paragraph>
        </Card>
      </Card>
    </div>
  );
};

export default CurrentPositionAnalysis; 