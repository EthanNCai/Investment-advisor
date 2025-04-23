import React from 'react';
import { Card, Alert, Descriptions, Progress, Space, Typography, Divider, Row, Col, Statistic } from 'antd';
import { InfoCircleOutlined, WarningOutlined, CheckCircleOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { Signal } from './InvestmentSignal';

const { Text, Paragraph } = Typography;

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
    nearest_signal_id, 
    similarity_score, 
    percentile, 
    is_extreme, 
    recommendation 
  } = currentPositionInfo;
  
  // 找到最近的信号
  const nearestSignal = signals.find(signal => signal.id === nearest_signal_id);
  
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
    
    // 为了在直方图上标记当前比值的位置
    const markLineData = [
      {
        name: '当前比值',
        xAxis: current_ratio.toFixed(4),
        itemStyle: {
          color: '#ff4d4f'
        },
        label: {
          formatter: `当前比值: ${current_ratio.toFixed(4)}`,
          position: 'start',
          fontSize: 14,
          color: '#ff4d4f'
        }
      }
    ];
    
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
  
  // 生成百分位图表
  const getPercentileChartOption = () => {
    const percentileValue = parseFloat(percentile) * 100 || 0;
    const colorMap = [
      {
        value: 20,
        color: '#52c41a'  // 绿色 - 低位区域
      },
      {
        value: 80,
        color: '#1890ff'  // 蓝色 - 中位区域
      },
      {
        value: 100,
        color: '#ff4d4f'  // 红色 - 高位区域
      }
    ];
    
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
            splitNumber: 5,
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
            show: true,
            fontSize: 14,
            color: '#333',
            offsetCenter: [0, '30%']
          },
          detail: {
            valueAnimation: true,
            fontSize: 30,
            offsetCenter: [0, '0%'],
            formatter: function(value: number) {
              return value.toFixed(1) + '%';
            },
            color: 'inherit'
          },
          data: [
            {
              value: percentileValue,
              name: '百分位',
              title: {
                offsetCenter: ['0%', '35%'],
                fontSize: 16
              }
            }
          ]
        }
      ]
    };
  };
  
  // 生成相似度展示图表
  const getSimilarityChartOption = () => {
    const similarityValue = parseFloat((similarity_score * 100).toFixed(1)) || 0;
    const color = similarityValue > 80 ? '#ff4d4f' : similarityValue > 60 ? '#faad14' : '#52c41a';
    
    return {
      series: [
        {
          type: 'gauge',
          startAngle: 180,
          endAngle: 0,
          min: 0,
          max: 100,
          radius: '100%',
          axisLine: {
            lineStyle: {
              width: 18,
              color: [
                [0.6, '#52c41a'],
                [0.8, '#faad14'],
                [1, '#ff4d4f']
              ]
            }
          },
          pointer: {
            icon: 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
            length: '12%',
            width: 20,
            offsetCenter: [0, '-55%'],
            itemStyle: {
              color: 'inherit'
            }
          },
          axisTick: {
            length: 12,
            lineStyle: {
              color: 'inherit',
              width: 2
            }
          },
          splitLine: {
            length: 20,
            lineStyle: {
              color: 'inherit',
              width: 5
            }
          },
          axisLabel: {
            color: '#999',
            fontSize: 12,
            distance: -40,
            formatter: function(value: number) {
              if (value === 100) {
                return '高';
              } else if (value === 0) {
                return '低';
              }
              return '';
            }
          },
          title: {
            offsetCenter: [0, '-20%'],
            fontSize: 16
          },
          detail: {
            valueAnimation: true,
            fontSize: 30,
            formatter: '{value}%',
            color: 'inherit',
            offsetCenter: [0, '0%']
          },
          data: [
            {
              value: similarityValue,
              name: '相似度',
              title: {
                offsetCenter: ['0%', '25%'],
                fontSize: 16
              }
            }
          ]
        }
      ]
    };
  };
  
  const getPositionAlert = () => {
    if (is_extreme) {
      return (
        <Alert
          message="极端位置警告"
          description="当前比值处于历史极端位置，可能存在较大投资机会或风险。"
          type="warning"
          showIcon
          icon={<WarningOutlined />}
          style={{ marginBottom: 16 }}
        />
      );
    }
    
    if (similarity_score > 0.8) {
      return (
        <Alert
          message="高度相似的历史信号"
          description={`当前比值与历史信号高度相似(${(similarity_score * 100).toFixed(1)}%)，请参考历史信号的表现。`}
          type="info"
          showIcon
          icon={<InfoCircleOutlined />}
          style={{ marginBottom: 16 }}
        />
      );
    }
    
    return (
      <Alert
        message="正常范围内"
        description="当前比值在历史正常范围内，未发现明显异常。"
        type="success"
        showIcon
        icon={<CheckCircleOutlined />}
        style={{ marginBottom: 16 }}
      />
    );
  };
  
  return (
    <div className="current-position-analysis">
      {getPositionAlert()}
      
      <Card title="当前位置分析" bordered={false}>
        <Row gutter={[16, 24]}>
          <Col span={24}>
            <Descriptions bordered size="middle" layout="vertical">
              <Descriptions.Item label="当前比值">{current_ratio.toFixed(4)}</Descriptions.Item>
              <Descriptions.Item label="百分位">{(percentile * 100).toFixed(1)}%</Descriptions.Item>
              <Descriptions.Item label="最相似信号">{nearestSignal ? nearestSignal.date : '无'}</Descriptions.Item>
              <Descriptions.Item label="相似度">{(similarity_score * 100).toFixed(1)}%</Descriptions.Item>
            </Descriptions>
          </Col>
          
          <Col xs={24} md={12}>
            <Card type="inner" title="比值历史分布" bordered={false}>
              <ReactECharts 
                option={getHistogramOption()} 
                style={{ height: '300px' }}
              />
            </Card>
          </Col>
          
          <Col xs={24} md={12}>
            <Row gutter={[0, 16]}>
              <Col span={24}>
                <Card type="inner" title="百分位" bordered={false}>
                  <ReactECharts 
                    option={getPercentileChartOption()} 
                    style={{ height: '140px' }}
                  />
                </Card>
              </Col>
              <Col span={24}>
                <Card type="inner" title="相似度" bordered={false}>
                  <ReactECharts 
                    option={getSimilarityChartOption()} 
                    style={{ height: '140px' }}
                  />
                </Card>
              </Col>
            </Row>
          </Col>
          
          {recommendation && (
            <Col span={24}>
              <Card type="inner" title="投资建议" bordered={false}>
                <Paragraph style={{ fontSize: '16px' }}>
                  {recommendation}
                </Paragraph>
              </Card>
            </Col>
          )}
        </Row>
      </Card>
    </div>
  );
}

export default CurrentPositionAnalysis; 