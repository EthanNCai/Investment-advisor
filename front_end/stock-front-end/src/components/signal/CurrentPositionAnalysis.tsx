import React from 'react';
import { Card, Alert, Descriptions, Progress, Space, Typography, Divider } from 'antd';
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
  
  // 生成分布图数据
  const getDistributionOption = () => {
    // 提取信号数据
    const ratioValues = signals.map(s => s.ratio);
    
    if (!ratioValues.length) {
      return {
        title: {
          text: '暂无历史比值数据',
          left: 'center'
        }
      };
    }
    
    // 排序并创建数据点用于绘制
    const sortedRatios = [...ratioValues].sort((a, b) => a - b);
    
    // 拟合多项式曲线（简单起见，使用线性拟合）
    // 获取x轴范围
    const minValue = Math.min(...sortedRatios, current_ratio) * 0.95;
    const maxValue = Math.max(...sortedRatios, current_ratio) * 1.05;
    
    // 计算均值和标准差
    const mean = sortedRatios.reduce((sum, val) => sum + val, 0) / sortedRatios.length;
    const variance = sortedRatios.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / sortedRatios.length;
    const std = Math.sqrt(variance);
    
    // 生成日期数据（用于X轴排序）
    const dates = signals.map(s => s.date);
    
    // 创建按日期排序的数据点
    const sortedData = signals.map(s => [s.date, s.ratio]);
    
    // 计算上下边界线
    const upperBound = mean + 2 * std;
    const lowerBound = mean > 2 * std ? mean - 2 * std : minValue;
    
    return {
      title: {
        text: '股票价格比值分析',
        left: 'center',
        top: 10
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params: any) {
          const date = params[0].axisValue;
          const ratio = params[0].data[1].toFixed(4);
          return `日期: ${date}<br/>比值: ${ratio}`;
        }
      },
      legend: {
        data: ['比值', '拟合线', '上边界', '下边界'],
        top: 40
      },
      grid: {
        top: 80,
        left: 50,
        right: 50,
        bottom: 30
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          type: 'slider',
          start: 0,
          end: 100
        }
      ],
      xAxis: {
        type: 'category',
        boundaryGap: false,
        data: dates,
        axisLabel: {
          rotate: 45,
          formatter: function(value: string) {
            return value.substring(5); // 只显示月-日部分
          }
        }
      },
      yAxis: {
        type: 'value',
        name: '价格比值',
        min: minValue,
        max: maxValue,
        axisLine: { show: true },
        axisLabel: {
          formatter: '{value}'
        }
      },
      series: [
        {
          name: '比值',
          type: 'line',
          data: sortedData,
          symbol: 'emptyCircle',
          symbolSize: 4,
          itemStyle: {
            color: '#5470c6'
          },
          markPoint: {
            data: [
              { 
                name: '当前比值', 
                value: current_ratio.toFixed(4),
                xAxis: dates[dates.length - 1],
                yAxis: current_ratio,
                itemStyle: { color: '#ff4d4f' },
                symbolSize: 10,
                symbol: 'pin'
              }
            ],
            label: {
              formatter: '{b}: {c}'
            }
          }
        },
        {
          name: '拟合线',
          type: 'line',
          smooth: true,
          symbol: 'none',
          data: sortedData.map((_, i) => [
            dates[i], 
            mean
          ]),
          lineStyle: {
            color: '#91cc75',
            type: 'dashed',
            width: 2
          }
        },
        {
          name: '上边界',
          type: 'line',
          symbol: 'none',
          data: sortedData.map((_, i) => [
            dates[i], 
            upperBound
          ]),
          lineStyle: {
            color: '#fac858',
            type: 'dotted',
            width: 2
          }
        },
        {
          name: '下边界',
          type: 'line',
          symbol: 'none',
          data: sortedData.map((_, i) => [
            dates[i], 
            lowerBound
          ]),
          lineStyle: {
            color: '#ee6666',
            type: 'dotted',
            width: 2
          }
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
      
      <Card bordered={false}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Descriptions bordered size="small" column={2}>
            <Descriptions.Item label="当前比值">{current_ratio.toFixed(4)}</Descriptions.Item>
            <Descriptions.Item label="百分位">
              {percentile !== null ? `${(percentile * 100).toFixed(1)}%` : '无法计算'}
            </Descriptions.Item>
            {nearestSignal && (
              <Descriptions.Item label="最相似信号">{nearestSignal.date}</Descriptions.Item>
            )}
            {similarity_score && (
              <Descriptions.Item label="相似度">
                <Progress 
                  percent={Math.round(similarity_score * 100)} 
                  size="small" 
                  status={similarity_score > 0.7 ? "exception" : "normal"}
                />
              </Descriptions.Item>
            )}
          </Descriptions>
          
          <Divider />
          
          <ReactECharts 
            option={getDistributionOption()} 
            style={{ height: '300px' }} 
          />
          
          <Divider orientation="left">投资建议</Divider>
          <Paragraph>
            <Text strong>{recommendation}</Text>
          </Paragraph>
        </Space>
      </Card>
    </div>
  );
};

export default CurrentPositionAnalysis; 