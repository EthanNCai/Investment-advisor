import React from 'react';
import { Card, Row, Col, Statistic, Typography, Divider } from 'antd';
import ReactECharts from 'echarts-for-react';
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined, 
  PercentageOutlined, 
  DollarOutlined,
  TrophyOutlined,
  WarningOutlined,
  CalculatorOutlined
} from '@ant-design/icons';

const { Title } = Typography;

interface BacktestResultsProps {
  results: {
    equity_curve: {
      date: string;
      equity: number;
      drawdown: number;
    }[];
    trades: any[];
    metrics: {
      total_return: number;
      annual_return: number;
      sharpe_ratio: number;
      max_drawdown: number;
      win_rate: number;
      profit_factor: number;
      total_trades: number;
    };
  };
}

const BacktestResults: React.FC<BacktestResultsProps> = ({ results }) => {
  const { equity_curve, metrics } = results;
  
  // 整理图表数据
  const chartData = {
    dates: equity_curve.map(item => item.date),
    equity: equity_curve.map(item => item.equity),
    drawdown: equity_curve.map(item => item.drawdown)
  };
  
  // 获取图表选项
  const getEquityCurveOption = () => {
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          animation: false,
          label: {
            backgroundColor: '#505765'
          }
        },
        formatter: function(params: any) {
          const date = params[0].axisValue;
          let html = `<div><strong>${date}</strong></div>`;
          
          params.forEach((param: any) => {
            const color = param.seriesName === '回撤' ? '#ff4d4f' : '#52c41a';
            const value = param.seriesName === '回撤' 
              ? `-${Math.abs(param.value).toFixed(2)}%` 
              : `￥${param.value.toFixed(2)}`;
            
            html += `<div style="color:${color}">
              ${param.marker} ${param.seriesName}: ${value}
            </div>`;
          });
          
          return html;
        }
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '10%',
        top: '10%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: chartData.dates,
        axisLine: { lineStyle: { color: '#aaa' } },
        axisLabel: {
          formatter: (value: string) => {
            // 简化日期显示
            return value.substring(5); // 只显示MM-DD部分
          }
        }
      },
      yAxis: [
        {
          name: '资金',
          type: 'value',
          position: 'left',
          axisLine: { lineStyle: { color: '#52c41a' } },
          axisLabel: {
            formatter: '￥{value}'
          }
        },
        {
          name: '回撤(%)',
          type: 'value',
          position: 'right',
          splitLine: { show: false },
          axisLine: { lineStyle: { color: '#ff4d4f' } },
          axisLabel: {
            formatter: '{value}%'
          },
          // 反转回撤轴使负值向下
          inverse: true,
          min: 0,
          max: Math.max(...chartData.drawdown.map(v => Math.abs(v)), 25)
        }
      ],
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
          name: '资金曲线',
          type: 'line',
          data: chartData.equity,
          smooth: true,
          showSymbol: false,
          itemStyle: {
            color: '#52c41a'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                {
                  offset: 0,
                  color: 'rgba(82, 196, 26, 0.3)'
                },
                {
                  offset: 1,
                  color: 'rgba(82, 196, 26, 0.1)'
                }
              ]
            }
          }
        },
        {
          name: '回撤',
          type: 'line',
          yAxisIndex: 1,
          data: chartData.drawdown.map(v => Math.abs(v)),  // 转为正值
          smooth: true,
          showSymbol: false,
          lineStyle: {
            color: '#ff4d4f',
            width: 1,
            type: 'dashed'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                {
                  offset: 0,
                  color: 'rgba(255, 77, 79, 0.1)'
                },
                {
                  offset: 1,
                  color: 'rgba(255, 77, 79, 0.3)'
                }
              ]
            }
          }
        }
      ]
    };
  };
  
  // 获取性能与风险指标图表
  const getRiskReturnOption = () => {
    const { annual_return, sharpe_ratio, win_rate, max_drawdown } = metrics;
    
    return {
      tooltip: {
        trigger: 'item'
      },
      radar: {
        indicator: [
          { name: '年化收益', max: 100 },
          { name: '夏普比率', max: 5 },
          { name: '胜率', max: 100 },
          { name: '最大回撤', max: 100 }
        ],
        shape: 'circle',
        splitNumber: 5,
        axisName: {
          color: '#333'
        },
        splitLine: {
          lineStyle: {
            color: 'rgba(211, 211, 211, 0.8)'
          }
        },
        splitArea: {
          show: false
        },
        axisLine: {
          lineStyle: {
            color: 'rgba(211, 211, 211, 0.8)'
          }
        }
      },
      series: [
        {
          name: '风险收益指标',
          type: 'radar',
          data: [
            {
              value: [
                annual_return * 100,
                sharpe_ratio,
                win_rate * 100,
                Math.min(max_drawdown * 100, 100)
              ],
              name: '策略表现',
              symbol: 'circle',
              symbolSize: 6,
              lineStyle: {
                color: '#1890ff',
                width: 2
              },
              areaStyle: {
                color: 'rgba(24, 144, 255, 0.2)'
              }
            }
          ]
        }
      ]
    };
  };
  
  return (
    <div className="backtest-results">
      <Card title="回测表现" bordered={false}>
        <Row gutter={[16, 16]}>
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="总收益率"
              value={metrics.total_return * 100}
              precision={2}
              valueStyle={{ color: metrics.total_return >= 0 ? '#3f8600' : '#cf1322' }}
              prefix={metrics.total_return >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
              suffix="%"
            />
          </Col>
          
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="年化收益率"
              value={metrics.annual_return * 100}
              precision={2}
              valueStyle={{ color: metrics.annual_return >= 0 ? '#3f8600' : '#cf1322' }}
              prefix={<PercentageOutlined />}
              suffix="%"
            />
          </Col>
          
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="夏普比率"
              value={metrics.sharpe_ratio}
              precision={2}
              valueStyle={{ color: metrics.sharpe_ratio >= 1 ? '#3f8600' : '#cf1322' }}
              prefix={<CalculatorOutlined />}
            />
          </Col>
          
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="最大回撤"
              value={metrics.max_drawdown * 100}
              precision={2}
              valueStyle={{ color: '#cf1322' }}
              prefix={<WarningOutlined />}
              suffix="%"
            />
          </Col>
          
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="胜率"
              value={metrics.win_rate * 100}
              precision={2}
              valueStyle={{ color: metrics.win_rate >= 0.5 ? '#3f8600' : '#cf1322' }}
              prefix={<TrophyOutlined />}
              suffix="%"
            />
          </Col>
          
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="盈亏比"
              value={metrics.profit_factor}
              precision={2}
              valueStyle={{ color: metrics.profit_factor >= 1 ? '#3f8600' : '#cf1322' }}
              prefix={<DollarOutlined />}
            />
          </Col>
          
          <Col xs={24} sm={12} md={8} lg={6}>
            <Statistic
              title="交易次数"
              value={metrics.total_trades}
              valueStyle={{ color: '#1890ff' }}
            />
          </Col>
        </Row>
        
        <Divider orientation="left">资金曲线与回撤</Divider>
        
        <ReactECharts 
          option={getEquityCurveOption()} 
          style={{ height: '400px' }}
          notMerge={true}
        />
        
        <Divider orientation="left">风险收益分析</Divider>
        
        <Row gutter={[16, 16]}>
          <Col xs={24} md={12}>
            <Card title="策略表现雷达图" bordered={false}>
              <ReactECharts 
                option={getRiskReturnOption()} 
                style={{ height: '300px' }}
                notMerge={true}
              />
            </Card>
          </Col>
          
          <Col xs={24} md={12}>
            <Card title="策略评估" bordered={false}>
              <div style={{ padding: '10px' }}>
                <Title level={5}>策略评估</Title>
                <p>
                  {metrics.annual_return >= 0.15 && metrics.sharpe_ratio >= 1.5 ? 
                    '该策略表现优异，具有较高的收益率和较好的风险控制。' : 
                    metrics.annual_return >= 0.08 && metrics.sharpe_ratio >= 1 ? 
                    '该策略表现良好，收益与风险平衡。' : 
                    '该策略需要改进，收益率偏低或风险偏高。'
                  }
                </p>
                
                <Title level={5}>收益能力</Title>
                <p>
                  年化收益率为{(metrics.annual_return * 100).toFixed(2)}%，
                  {metrics.annual_return >= 0.15 ? '远高于' : 
                   metrics.annual_return >= 0.08 ? '高于' : 
                   metrics.annual_return >= 0.04 ? '接近' : '低于'}
                  市场平均水平。
                </p>
                
                <Title level={5}>风险控制</Title>
                <p>
                  最大回撤为{(metrics.max_drawdown * 100).toFixed(2)}%，
                  {metrics.max_drawdown <= 0.1 ? '风险控制良好。' : 
                   metrics.max_drawdown <= 0.2 ? '风险控制一般。' : 
                   '风险偏高，建议优化止损策略。'}
                </p>
                
                <Title level={5}>交易效率</Title>
                <p>
                  总计进行了{metrics.total_trades}次交易，胜率为{(metrics.win_rate * 100).toFixed(2)}%，
                  盈亏比为{metrics.profit_factor.toFixed(2)}。
                  {metrics.win_rate >= 0.6 && metrics.profit_factor >= 2 ? 
                    '交易效率高，策略稳健。' : 
                    metrics.win_rate >= 0.5 && metrics.profit_factor >= 1.5 ? 
                    '交易效率良好。' : 
                    '交易效率有待提高，建议优化入场和出场条件。'
                  }
                </p>
              </div>
            </Card>
          </Col>
        </Row>
      </Card>
    </div>
  );
};

export default BacktestResults; 