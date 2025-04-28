import React, { useState, useEffect } from 'react';
import { Card, Typography, Space, Spin, Empty, Statistic, Row, Col, Divider, Progress, Tooltip } from 'antd';
import { InfoCircleOutlined, SyncOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';

const { Text, Title, Paragraph } = Typography;

// 定义统计数据接口
interface SignalTypeStats {
  count: number;
  success_rate: number;
  avg_return: number;
}

interface ConfidenceLevelStats {
  count: number;
  success_rate: number;
  avg_return: number;
}

interface SignalPerformanceStats {
  total_signals: number;
  validated_signals: number;
  success_rate: number;
  avg_return: number;
  avg_risk: number;
  signal_types: {
    positive?: SignalTypeStats;
    negative?: SignalTypeStats;
  };
  confidence_levels: {
    low?: ConfidenceLevelStats;
    medium?: ConfidenceLevelStats;
    high?: ConfidenceLevelStats;
  };
}

interface SignalStatsCardProps {
  codeA?: string;
  codeB?: string;
}

const SignalStatsCard: React.FC<SignalStatsCardProps> = ({ codeA, codeB }) => {
  const [stats, setStats] = useState<SignalPerformanceStats | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  
  // 加载统计数据
  useEffect(() => {
    fetchStats();
  }, [codeA, codeB]);
  
  const fetchStats = async () => {
    setLoading(true);
    try {
      // 构建URL，根据是否有股票代码增加查询参数
      let url = 'http://localhost:8000/signal_performance_stats/';
      if (codeA && codeB) {
        url += `?code_a=${codeA}&code_b=${codeB}`;
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error('获取信号统计数据失败');
      }
      
      const data = await response.json();
      setStats(data);
    } catch (err) {
      console.error('获取信号统计数据错误:', err);
      setError('获取信号统计数据失败');
    } finally {
      setLoading(false);
    }
  };
  
  // 渲染信号类型统计图表
  const renderSignalTypeChart = () => {
    if (!stats || !stats.signal_types) {
      return <Empty description="暂无类型统计数据" />;
    }
    
    const positive = stats.signal_types.positive || { count: 0, success_rate: 0, avg_return: 0 };
    const negative = stats.signal_types.negative || { count: 0, success_rate: 0, avg_return: 0 };
    
    const option = {
      title: {
        text: '信号类型分析',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        },
        formatter: function(params: any) {
          const data = params[0].data;
          return `${params[0].name}<br/>${params[0].seriesName}: ${data.toFixed(2)}%`;
        }
      },
      legend: {
        data: ['成功率', '平均收益'],
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
        data: ['正向信号', '负向信号']
      },
      yAxis: {
        type: 'value',
        axisLabel: {
          formatter: '{value}%'
        }
      },
      series: [
        {
          name: '成功率',
          type: 'bar',
          data: [
            positive.success_rate * 100,
            negative.success_rate * 100
          ],
          itemStyle: {
            color: '#1890ff'
          },
          label: {
            show: true,
            position: 'top',
            formatter: '{c}%'
          }
        },
        {
          name: '平均收益',
          type: 'bar',
          data: [
            positive.avg_return * 100,
            negative.avg_return * 100
          ],
          itemStyle: {
            color: '#52c41a'
          },
          label: {
            show: true,
            position: 'top',
            formatter: '{c}%'
          }
        }
      ]
    };
    
    return <ReactECharts option={option} style={{ height: '300px' }} />;
  };
  
  // 渲染可信度级别统计图表
  const renderConfidenceLevelChart = () => {
    if (!stats || !stats.confidence_levels) {
      return <Empty description="暂无可信度统计数据" />;
    }
    
    const low = stats.confidence_levels.low || { count: 0, success_rate: 0, avg_return: 0 };
    const medium = stats.confidence_levels.medium || { count: 0, success_rate: 0, avg_return: 0 };
    const high = stats.confidence_levels.high || { count: 0, success_rate: 0, avg_return: 0 };
    
    const option = {
      title: {
        text: '信号可信度分析',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'shadow'
        }
      },
      legend: {
        data: ['信号数量', '成功率', '平均收益'],
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
        data: ['低可信度', '中可信度', '高可信度']
      },
      yAxis: [
        {
          type: 'value',
          name: '信号数量',
          position: 'left',
        },
        {
          type: 'value',
          name: '比率',
          position: 'right',
          axisLabel: {
            formatter: '{value}%'
          },
          max: 100
        }
      ],
      series: [
        {
          name: '信号数量',
          type: 'bar',
          data: [low.count, medium.count, high.count],
          itemStyle: {
            color: '#1890ff'
          }
        },
        {
          name: '成功率',
          type: 'line',
          yAxisIndex: 1,
          data: [
            low.success_rate * 100,
            medium.success_rate * 100,
            high.success_rate * 100
          ],
          lineStyle: {
            color: '#52c41a'
          },
          label: {
            show: true,
            formatter: '{c}%'
          }
        },
        {
          name: '平均收益',
          type: 'line',
          yAxisIndex: 1,
          data: [
            low.avg_return * 100,
            medium.avg_return * 100,
            high.avg_return * 100
          ],
          lineStyle: {
            color: '#faad14'
          },
          label: {
            show: true,
            formatter: '{c}%'
          }
        }
      ]
    };
    
    return <ReactECharts option={option} style={{ height: '300px' }} />;
  };
  
  return (
    <Card 
      title={
        <Space>
          <InfoCircleOutlined />
          <span>信号验证统计</span>
        </Space>
      }
      extra={
        <Space>
          {loading && <SyncOutlined spin />}
          <a onClick={fetchStats}>刷新</a>
        </Space>
      }
      bordered={false}
      className="signal-stats-card"
    >
      {loading ? (
        <div style={{ textAlign: 'center', padding: '30px 0' }}>
          <Spin />
          <div style={{ marginTop: 8 }}>加载统计数据...</div>
        </div>
      ) : error ? (
        <div style={{ textAlign: 'center', color: '#ff4d4f' }}>
          {error}
        </div>
      ) : !stats || stats.validated_signals === 0 ? (
        <Empty description="暂无已验证的信号数据" />
      ) : (
        <>
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Statistic 
                title={
                  <Tooltip title="系统中已记录的信号总数">
                    <Space>
                      <span>信号总数</span>
                      <QuestionCircleOutlined />
                    </Space>
                  </Tooltip>
                }
                value={stats.total_signals} 
                suffix="个"
              />
            </Col>
            <Col span={8}>
              <Statistic 
                title={
                  <Tooltip title="已完成验证的信号数量">
                    <Space>
                      <span>已验证信号</span>
                      <QuestionCircleOutlined />
                    </Space>
                  </Tooltip>
                }
                value={stats.validated_signals} 
                suffix={`个 (${Math.round(stats.validated_signals / stats.total_signals * 100)}%)`}
              />
            </Col>
            <Col span={8}>
              <Statistic 
                title={
                  <Tooltip title="验证信号中产生盈利的比例">
                    <Space>
                      <span>整体成功率</span>
                      <QuestionCircleOutlined />
                    </Space>
                  </Tooltip>
                }
                value={stats.success_rate * 100} 
                precision={1}
                suffix="%" 
                valueStyle={{ color: stats.success_rate >= 0.5 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
          </Row>
          
          <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
            <Col span={12}>
              <Statistic 
                title={
                  <Tooltip title="所有验证信号的平均收益率">
                    <Space>
                      <span>平均收益率</span>
                      <QuestionCircleOutlined />
                    </Space>
                  </Tooltip>
                }
                value={stats.avg_return * 100} 
                precision={2}
                suffix="%" 
                valueStyle={{ color: stats.avg_return >= 0 ? '#3f8600' : '#cf1322' }}
              />
            </Col>
            <Col span={12}>
              <Statistic 
                title={
                  <Tooltip title="所有验证信号的平均最大回撤">
                    <Space>
                      <span>平均最大回撤</span>
                      <QuestionCircleOutlined />
                    </Space>
                  </Tooltip>
                }
                value={stats.avg_risk * 100} 
                precision={2}
                suffix="%" 
                valueStyle={{ color: '#cf1322' }}
              />
            </Col>
          </Row>
          
          <Divider orientation="left">信号类型统计</Divider>
          {renderSignalTypeChart()}
          
          <Divider orientation="left">可信度级别分析</Divider>
          {renderConfidenceLevelChart()}
          
          <Divider />
          <Paragraph type="secondary" style={{ textAlign: 'center' }}>
            统计数据基于 {stats.validated_signals} 个已完成验证的信号
          </Paragraph>
        </>
      )}
    </Card>
  );
};

export default SignalStatsCard; 