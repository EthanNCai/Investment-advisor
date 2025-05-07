import React from 'react';
import { Card, Row, Col, Statistic, Typography, Divider, Tabs, Table } from 'antd';
import ReactECharts from 'echarts-for-react';
import { 
  ArrowUpOutlined, 
  ArrowDownOutlined, 
  PercentageOutlined, 
  DollarOutlined,
  TrophyOutlined,
  WarningOutlined,
  CalculatorOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
  FundOutlined
} from '@ant-design/icons';

const { Title } = Typography;
const { TabPane } = Tabs;

interface BacktestResultsProps {
  results: {
    equity_curve: {
      date: string;
      equity: number;
      drawdown: number;
      cash: number;
      positions_value: number;
      signal: number;
      ratio: number;
    }[];
    trades: any[];
    initial_capital: number;
    final_equity: number;
    total_return: number;
    annual_return: number;
    max_drawdown: number;
    max_drawdown_duration: number;
    recovery_period: number;
    sharpe_ratio: number;
    sortino_ratio: number;
    calmar_ratio: number;
    total_trades: number;
    profitable_trades: number;
    losing_trades: number;
    win_rate: number;
    avg_profit: number;
    avg_loss: number;
    profit_factor: number;
    avg_holding_period: number;
    strategy_parameters: {
      strategy_type: string;
      entry_threshold: number;
      exit_threshold: number;
      stop_loss: number;
      take_profit: number;
      trailing_stop: number;
      time_stop: number;
      hedge_mode: string;
      position_size_type: string;
      position_size: number;
    };
  };
}

const BacktestResults: React.FC<BacktestResultsProps> = ({ results }) => {
  const { 
    equity_curve, 
    initial_capital, 
    final_equity, 
    total_return, 
    annual_return, 
    max_drawdown, 
    max_drawdown_duration, 
    recovery_period, 
    sharpe_ratio, 
    sortino_ratio, 
    calmar_ratio, 
    total_trades, 
    profitable_trades, 
    losing_trades, 
    win_rate, 
    avg_profit, 
    avg_loss, 
    profit_factor,
    avg_holding_period,
    strategy_parameters
  } = results;
  
  // 整理图表数据
  const chartData = {
    dates: equity_curve.map(item => item.date),
    equity: equity_curve.map(item => item.equity),
    drawdown: equity_curve.map(item => item.drawdown),
    cash: equity_curve.map(item => item.cash),
    positions: equity_curve.map(item => item.positions_value),
    signal: equity_curve.map(item => item.signal),
    ratio: equity_curve.map(item => item.ratio)
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
            let value;
            
            if (param.seriesName === '回撤') {
              value = `-${Math.abs(param.value).toFixed(2)}%`;
            } else if (param.seriesName === '资金曲线') {
              value = `￥${param.value.toFixed(2)}`;
            } else if (param.seriesName === '现金') {
              value = `￥${param.value.toFixed(2)}`;
            } else if (param.seriesName === '持仓价值') {
              value = `￥${param.value.toFixed(2)}`;
            }
            
            html += `<div style="color:${param.color}">
              ${param.marker} ${param.seriesName}: ${value}
            </div>`;
          });
          
          return html;
        }
      },
      legend: {
        data: ['资金曲线', '回撤', '现金', '持仓价值'],
        bottom: 0
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
        },
        {
          name: '现金',
          type: 'line',
          data: chartData.cash,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            color: '#1890ff',
            width: 1,
            type: 'dashed'
          }
        },
        {
          name: '持仓价值',
          type: 'line',
          data: chartData.positions,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            color: '#722ed1',
            width: 1,
            type: 'dashed'
          }
        }
      ]
    };
  };
  
  // 获取信号与比值图表
  const getSignalRatioOption = () => {
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross',
          animation: false,
          label: {
            backgroundColor: '#505765'
          }
        }
      },
      legend: {
        data: ['信号值', '价格比值'],
        bottom: 0
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
            return value.substring(5); // 只显示MM-DD部分
          }
        }
      },
      yAxis: [
        {
          name: '信号值',
          type: 'value',
          position: 'left',
          axisLine: { lineStyle: { color: '#1890ff' } }
        },
        {
          name: '价格比值',
          type: 'value',
          position: 'right',
          splitLine: { show: false },
          axisLine: { lineStyle: { color: '#fa8c16' } }
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
          name: '信号值',
          type: 'line',
          data: chartData.signal,
          smooth: true,
          showSymbol: false,
          itemStyle: {
            color: '#1890ff'
          },
          markLine: {
            silent: true,
            lineStyle: {
              color: '#333',
              type: 'dashed'
            },
            data: [
              {
                yAxis: strategy_parameters.entry_threshold,
                label: { show: true, formatter: '入场阈值' }
              },
              {
                yAxis: -strategy_parameters.entry_threshold,
                label: { show: true, formatter: '入场阈值' }
              },
              {
                yAxis: strategy_parameters.exit_threshold,
                label: { show: true, formatter: '出场阈值' }
              },
              {
                yAxis: -strategy_parameters.exit_threshold,
                label: { show: true, formatter: '出场阈值' }
              },
              { yAxis: 0, label: { show: false } }
            ]
          }
        },
        {
          name: '价格比值',
          type: 'line',
          yAxisIndex: 1,
          data: chartData.ratio,
          smooth: true,
          showSymbol: false,
          lineStyle: {
            color: '#fa8c16',
            width: 1
          }
        }
      ]
    };
  };
  
  // 获取性能与风险指标图表
  const getRiskReturnOption = () => {
    return {
      tooltip: {
        trigger: 'item'
      },
      radar: {
        indicator: [
          { name: '年化收益', max: Math.max(50, Math.ceil(annual_return)) },
          { name: '夏普比率', max: Math.max(5, Math.ceil(sharpe_ratio)) },
          { name: '胜率', max: 100 },
          { name: '最大回撤', max: 100 },
          { name: '索提诺比率', max: Math.max(5, Math.ceil(sortino_ratio)) },
          { name: '卡尔马比率', max: Math.max(5, Math.ceil(calmar_ratio)) }
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
                parseFloat(annual_return.toFixed(4)),
                parseFloat(sharpe_ratio.toFixed(4)),
                parseFloat(win_rate.toFixed(4)),
                parseFloat(max_drawdown.toFixed(4)),
                parseFloat(sortino_ratio.toFixed(4)),
                parseFloat(calmar_ratio.toFixed(4))
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

  // 获取策略参数表格数据
  const getStrategyParams = () => {
    return [
      {
        key: '1',
        parameter: '策略类型',
        value: strategy_parameters.strategy_type === 'zscore' ? 'Z分数策略' : 
               strategy_parameters.strategy_type === 'percent' ? '百分比偏离策略' : 
               strategy_parameters.strategy_type === 'volatility' ? '波动率调整策略' : 
               strategy_parameters.strategy_type === 'trend' ? '趋势跟踪策略' : 
               strategy_parameters.strategy_type
      },
      {
        key: '2',
        parameter: '对冲模式',
        value: strategy_parameters.hedge_mode === 'single' ? '单边交易' : '对冲交易'
      },
      {
        key: '3',
        parameter: '入场阈值',
        value: strategy_parameters.entry_threshold
      },
      {
        key: '4',
        parameter: '出场阈值',
        value: strategy_parameters.exit_threshold
      },
      {
        key: '5',
        parameter: '止损比例',
        value: `${strategy_parameters.stop_loss}%`
      },
      {
        key: '6',
        parameter: '止盈比例',
        value: `${strategy_parameters.take_profit}%`
      },
      {
        key: '7',
        parameter: '追踪止损',
        value: strategy_parameters.trailing_stop > 0 ? `${strategy_parameters.trailing_stop}%` : '未启用'
      },
      {
        key: '8',
        parameter: '时间止损',
        value: strategy_parameters.time_stop > 0 ? `${strategy_parameters.time_stop}天` : '未启用'
      },
      {
        key: '9',
        parameter: '仓位计算方式',
        value: strategy_parameters.position_size_type === 'fixed' ? '固定金额' : 
               strategy_parameters.position_size_type === 'kelly' ? '凯利公式' : 
               '资金百分比'
      },
      {
        key: '10',
        parameter: '仓位大小',
        value: strategy_parameters.position_size_type === 'fixed' ? 
               `￥${strategy_parameters.position_size}` : 
               `${strategy_parameters.position_size}%`
      }
    ];
  };

  const paramColumns = [
    {
      title: '参数',
      dataIndex: 'parameter',
      key: 'parameter',
      width: '50%'
    },
    {
      title: '值',
      dataIndex: 'value',
      key: 'value',
      width: '50%'
    }
  ];
  
  return (
    <div className="backtest-results">
      <Tabs defaultActiveKey="1">
        <TabPane tab={<span><LineChartOutlined /> 资金曲线</span>} key="1">
          <Card>
            <ReactECharts option={getEquityCurveOption()} style={{ height: '400px' }} />
          </Card>
          
          <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="初始资金"
                  value={initial_capital}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<DollarOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="最终权益"
                  value={final_equity}
                  precision={2}
                  valueStyle={{ color: '#52c41a' }}
                  prefix={<DollarOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="总收益率"
                  value={total_return}
                  precision={2}
                  valueStyle={{ color: total_return >= 0 ? '#52c41a' : '#f5222d' }}
                  prefix={total_return >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  suffix="%"
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="年化收益率"
                  value={annual_return}
                  precision={2}
                  valueStyle={{ color: annual_return >= 0 ? '#52c41a' : '#f5222d' }}
                  prefix={annual_return >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                  suffix="%"
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="最大回撤"
                  value={max_drawdown}
                  precision={2}
                  valueStyle={{ color: '#f5222d' }}
                  prefix={<WarningOutlined />}
                  suffix="%"
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="最大回撤持续期"
                  value={max_drawdown_duration}
                  valueStyle={{ color: '#fa8c16' }}
                  prefix={<ClockCircleOutlined />}
                  suffix="天"
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="回撤恢复期"
                  value={recovery_period}
                  valueStyle={{ color: '#fa8c16' }}
                  prefix={<ClockCircleOutlined />}
                  suffix="天"
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="胜率"
                  value={win_rate}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<TrophyOutlined />}
                  suffix="%"
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="盈亏比"
                  value={profit_factor}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<CalculatorOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="夏普比率"
                  value={sharpe_ratio}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<CalculatorOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="索提诺比率"
                  value={sortino_ratio}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<CalculatorOutlined />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="卡尔马比率"
                  value={calmar_ratio}
                  precision={2}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<CalculatorOutlined />}
                />
              </Card>
            </Col>
            
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="交易次数"
                  value={total_trades}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="盈利交易"
                  value={profitable_trades}
                  valueStyle={{ color: '#52c41a' }}
                  suffix={`/${total_trades}`}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="亏损交易"
                  value={losing_trades}
                  valueStyle={{ color: '#f5222d' }}
                  suffix={`/${total_trades}`}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={8} lg={6}>
              <Card>
                <Statistic
                  title="平均持仓时间"
                  value={avg_holding_period.toFixed(1)}
                  valueStyle={{ color: '#1890ff' }}
                  suffix="天"
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
        
        <TabPane tab={<span><LineChartOutlined /> 信号分析</span>} key="2">
          <Card>
            <ReactECharts option={getSignalRatioOption()} style={{ height: '400px' }} />
          </Card>
        </TabPane>
        
        <TabPane tab={<span><FundOutlined /> 策略评估</span>} key="3">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="风险收益分析">
                <ReactECharts option={getRiskReturnOption()} style={{ height: '400px' }} />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="策略参数">
                <Table 
                  dataSource={getStrategyParams()} 
                  columns={paramColumns} 
                  pagination={false} 
                  size="small"
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};

export default BacktestResults; 