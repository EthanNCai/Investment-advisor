import React, { useState, useEffect } from 'react';
import { Card, Typography, Space, Spin, Empty, Statistic, Row, Col, Divider, Table, Tag } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined, SyncOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { Signal } from './InvestmentSignal';

const { Text, Title } = Typography;

// 定义跟踪数据接口
interface TrackingDataPoint {
  date: string;
  ratio: number;
  return: number;
}

interface SignalTrackingData {
  record_id: number;
  signal_id: number;
  code_a: string;
  code_b: string;
  signal_date: string;
  signal_type: string;
  signal_strength: string;
  was_profitable: boolean | null;
  actual_return: number | null;
  max_drawdown: number | null;
  validation_completed: boolean;
  followup_data: TrackingDataPoint[];
  days_tracked: number;
}

interface SignalTrackingCardProps {
  signal: Signal;
}

const SignalTrackingCard: React.FC<SignalTrackingCardProps> = ({ signal }) => {
  const [trackingData, setTrackingData] = useState<SignalTrackingData | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // 加载信号追踪数据
  useEffect(() => {
    const fetchTrackingData = async () => {
      if (!signal || !signal.record_id) {
        return;
      }
      
      setLoading(true);
      try {
        const response = await fetch(`http://localhost:8000/signal_tracking/${signal.record_id}`);
        
        if (!response.ok) {
          throw new Error('获取信号追踪数据失败');
        }
        
        const data = await response.json();
        setTrackingData(data);
      } catch (err) {
        console.error('获取信号追踪数据错误:', err);
        setError('获取信号追踪数据失败');
      } finally {
        setLoading(false);
      }
    };
    
    fetchTrackingData();
  }, [signal]);
  
  // 刷新追踪数据
  const handleRefresh = () => {
    if (signal && signal.record_id) {
      setLoading(true);
      fetch(`http://localhost:8000/signal_tracking/${signal.record_id}`)
        .then(response => {
          if (!response.ok) {
            throw new Error('刷新数据失败');
          }
          return response.json();
        })
        .then(data => {
          setTrackingData(data);
        })
        .catch(err => {
          console.error('刷新数据错误:', err);
          setError('刷新数据失败');
        })
        .finally(() => {
          setLoading(false);
        });
    }
  };
  
  // 渲染跟踪图表
  const renderTrackingChart = () => {
    if (!trackingData || !trackingData.followup_data || trackingData.followup_data.length === 0) {
      return <Empty description="暂无跟踪数据" />;
    }
    
    const dates = trackingData.followup_data.map(item => item.date);
    const returns = trackingData.followup_data.map(item => item.return);
    const ratios = trackingData.followup_data.map(item => item.ratio);
    
    // 计算上下限范围
    const minReturn = Math.min(...returns);
    const maxReturn = Math.max(...returns);
    const range = Math.max(Math.abs(minReturn), Math.abs(maxReturn));
    const yAxisMin = Math.floor(-range * 1.1 * 10) / 10;
    const yAxisMax = Math.ceil(range * 1.1 * 10) / 10;
    
    const returnChartOption = {
      title: {
        text: '信号收益跟踪',
        left: 'center',
        top: 10
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params: any) {
          const dateIndex = params[0].dataIndex;
          const date = dates[dateIndex];
          const returnValue = returns[dateIndex];
          return `${date}<br/>收益率: ${returnValue.toFixed(2)}%`;
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
        data: dates,
        axisLabel: {
          formatter: function(value: string) {
            return value.substring(5); // 只显示月-日
          },
          interval: Math.floor(dates.length / 5), // 适当间隔显示日期
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        min: yAxisMin,
        max: yAxisMax,
        axisLabel: {
          formatter: '{value}%'
        },
        splitLine: {
          lineStyle: {
            type: 'dashed'
          }
        }
      },
      series: [
        {
          data: returns,
          type: 'line',
          name: '收益率(%)',
          markLine: {
            symbol: 'none',
            data: [
              { 
                yAxis: 0, 
                lineStyle: { 
                  color: '#aaa',
                  type: 'solid'
                }
              }
            ]
          },
          lineStyle: {
            width: 3,
            color: returns[returns.length - 1] >= 0 ? '#52c41a' : '#ff4d4f'
          },
          areaStyle: {
            color: returns[returns.length - 1] >= 0 ? 'rgba(82, 196, 26, 0.2)' : 'rgba(255, 77, 79, 0.2)'
          }
        }
      ]
    };
    
    const ratioChartOption = {
      title: {
        text: '价格比值变化',
        left: 'center',
        top: 10
      },
      tooltip: {
        trigger: 'axis',
        formatter: function(params: any) {
          const dateIndex = params[0].dataIndex;
          const date = dates[dateIndex];
          const ratio = ratios[dateIndex];
          return `${date}<br/>价格比值: ${ratio.toFixed(4)}`;
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
        data: dates,
        axisLabel: {
          formatter: function(value: string) {
            return value.substring(5); // 只显示月-日
          },
          interval: Math.floor(dates.length / 5), // 适当间隔显示日期
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        scale: true
      },
      series: [
        {
          data: ratios,
          type: 'line',
          name: '价格比值',
          lineStyle: {
            width: 2
          },
          symbol: 'circle',
          symbolSize: 4
        }
      ]
    };
    
    return (
      <Row gutter={[0, 16]}>
        <Col span={24}>
          <ReactECharts option={returnChartOption} style={{ height: '240px' }} />
        </Col>
        <Col span={24}>
          <ReactECharts option={ratioChartOption} style={{ height: '240px' }} />
        </Col>
      </Row>
    );
  };
  
  // 渲染表现统计数据
  const renderPerformanceStats = () => {
    if (!trackingData || !trackingData.validation_completed) {
      return (
        <Empty 
          description={
            <span>
              尚未完成验证
              {trackingData?.followup_data?.length > 0 && 
                ` (已跟踪 ${trackingData.followup_data.length} 天)`
              }
            </span>
          }
        />
      );
    }
    
    return (
      <Row gutter={16}>
        <Col span={8}>
          <Statistic
            title="最终收益率"
            value={trackingData.actual_return !== null ? trackingData.actual_return * 100 : 0}
            precision={2}
            valueStyle={{
              color: (trackingData.actual_return || 0) >= 0 ? '#3f8600' : '#cf1322'
            }}
            prefix={(trackingData.actual_return || 0) >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
            suffix="%"
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="最大回撤"
            value={trackingData.max_drawdown !== null ? trackingData.max_drawdown * 100 : 0}
            precision={2}
            valueStyle={{ color: '#cf1322' }}
            prefix={<ArrowDownOutlined />}
            suffix="%"
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="跟踪天数"
            value={trackingData.days_tracked || 0}
            suffix="天"
          />
        </Col>
        <Col span={24} style={{ marginTop: 16 }}>
          <Space>
            <Text>信号结果:</Text>
            {trackingData.was_profitable === null ? (
              <Tag color="blue">验证中</Tag>
            ) : trackingData.was_profitable ? (
              <Tag color="success">成功</Tag>
            ) : (
              <Tag color="error">失败</Tag>
            )}
          </Space>
        </Col>
      </Row>
    );
  };
  
  return (
    <Card 
      title={
        <Space>
          <InfoCircleOutlined />
          <span>信号追踪</span>
        </Space>
      }
      extra={
        <Space>
          {loading && <SyncOutlined spin />}
          <a onClick={handleRefresh}>刷新数据</a>
        </Space>
      }
      bordered={false}
      className="signal-tracking-card"
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: '30px 0' }}>
          <Spin />
          <div style={{ marginTop: 8 }}>加载跟踪数据...</div>
        </div>
      ) : error ? (
        <div style={{ textAlign: 'center', color: '#ff4d4f' }}>
          {error}
        </div>
      ) : !signal.record_id ? (
        <Empty description="该信号未被记录，无法追踪" />
      ) : (
        <>
          <Divider orientation="left">表现统计</Divider>
          {renderPerformanceStats()}
          
          <Divider orientation="left">趋势图表</Divider>
          {renderTrackingChart()}
        </>
      )}
    </Card>
  );
};

export default SignalTrackingCard; 