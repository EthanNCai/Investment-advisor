import React from 'react';
import { Card, Row, Col, Table, Statistic, Typography, Divider, Progress, Empty, Spin } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons';
import { Line } from '@ant-design/charts';

const { Title, Text } = Typography;

interface SimilarSignalsBacktestResultsProps {
  results: any;
  loading: boolean;
  stockPair?: { codeA: string, nameA: string, codeB: string, nameB: string };
}

const SimilarSignalsBacktestResults: React.FC<SimilarSignalsBacktestResultsProps> = ({ 
  results, 
  loading,
  stockPair 
}) => {
  if (loading) {
    return <Spin tip="正在加载回测结果..." />;
  }
  
  if (!results) {
    return <Empty description="尚未运行回测，请设置参数并点击开始回测" />;
  }

  // 提取回测结果数据
  const {
    trades,
    initial_capital,
    final_equity,
    total_return,
    total_trades,
    profitable_trades,
    win_rate,
    profit_loss_ratio,
    avg_holding_days,
    similar_signals,
    code_a,
    code_b
  } = results;

  // 计算统计数据
  const winning_trades = profitable_trades;
  const losing_trades = total_trades - profitable_trades;
  const profit_factor = profit_loss_ratio || 0;
  const average_return = total_return / total_trades || 0;
  const average_profit = trades.filter((t: any) => t.pnl > 0).reduce((sum: number, trade: any) => sum + trade.pnl_percent, 0) / winning_trades || 0;
  const average_loss = trades.filter((t: any) => t.pnl <= 0).reduce((sum: number, trade: any) => sum + Math.abs(trade.pnl_percent), 0) / losing_trades || 0;

  // 获取连续盈亏次数
  const getConsecutiveCount = (array: any[], condition: (item: any) => boolean) => {
    let maxCount = 0;
    let currentCount = 0;
    
    array.forEach(item => {
      if (condition(item)) {
        currentCount++;
        maxCount = Math.max(maxCount, currentCount);
      } else {
        currentCount = 0;
      }
    });
    
    return maxCount;
  };

  const max_consecutive_wins = getConsecutiveCount(trades, (trade: any) => trade.pnl > 0);
  const max_consecutive_losses = getConsecutiveCount(trades, (trade: any) => trade.pnl <= 0);

  // 检查后端是否提供了portfolio_value_history
  let portfolio_value_history = [];
  
  if (results.portfolio_value_history && results.portfolio_value_history.length > 0) {
    // 使用后端提供的资金曲线
    try {
      // 确保数据格式正确，严格转换日期和数字格式
      portfolio_value_history = results.portfolio_value_history.map((item: any) => {
        // 确保日期格式正确
        let formattedDate = item.date;
        if (typeof formattedDate === 'object' && formattedDate instanceof Date) {
          formattedDate = formattedDate.toISOString().split('T')[0];
        }
        
        // 确保portfolio_value是数字
        let portfolioValue = 0;
        if (typeof item.portfolio_value === 'number') {
          portfolioValue = item.portfolio_value;
        } else if (item.portfolio_value) {
          portfolioValue = parseFloat(String(item.portfolio_value).replace(/,/g, ''));
        }
        
        return {
          date: formattedDate,
          portfolio_value: isNaN(portfolioValue) ? 0 : portfolioValue
        };
      });
      
      // 数据验证
      portfolio_value_history = portfolio_value_history.filter((item: any) => 
        item.date && !isNaN(item.portfolio_value)
      );
      
      console.log("处理后的后端资金曲线数据:", portfolio_value_history);
    } catch (error) {
      console.error("处理后端资金曲线数据出错:", error);
      portfolio_value_history = [];
    }
  }
  
  // 如果后端数据处理失败或不存在，生成模拟数据
  if (portfolio_value_history.length < 2 && trades && trades.length > 0) {
    try {
      // 先按日期排序交易
      const sortedTrades = [...trades].sort((a, b) => 
        new Date(a.entry_date).getTime() - new Date(b.entry_date).getTime()
      );
      
      // 重置资金曲线数组
      portfolio_value_history = [];
      
      // 起始点
      portfolio_value_history.push({
        date: sortedTrades[0]?.entry_date || new Date().toISOString().split('T')[0],
        portfolio_value: initial_capital
      });
      
      // 添加每笔交易的结束点
      let currentValue = initial_capital;
      sortedTrades.forEach(trade => {
        // 确保pnl是数字类型
        const pnl = typeof trade.pnl === 'number' ? trade.pnl : parseFloat(String(trade.pnl || '0').replace(/,/g, ''));
        currentValue += isNaN(pnl) ? 0 : pnl;
        
        portfolio_value_history.push({
          date: trade.exit_date,
          portfolio_value: currentValue
        });
      });
      
      console.log("生成的资金曲线数据:", portfolio_value_history);
    } catch (error) {
      console.error("生成资金曲线时出错:", error);
      // 创建默认数据
      portfolio_value_history = [
        { date: new Date().toISOString().split('T')[0], portfolio_value: initial_capital },
        { date: new Date(new Date().getTime() + 24*60*60*1000).toISOString().split('T')[0], portfolio_value: final_equity }
      ];
    }
  } else if (portfolio_value_history.length < 2) {
    // 没有足够数据时创建一个简单的两点线
    portfolio_value_history = [
      { date: new Date().toISOString().split('T')[0], portfolio_value: initial_capital },
      { date: new Date(new Date().getTime() + 24*60*60*1000).toISOString().split('T')[0], portfolio_value: final_equity }
    ];
  }

  // 交易明细表格列定义
  const tradeColumns = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60
    },
    {
      title: '开仓日期',
      dataIndex: 'entry_date',
      key: 'entry_date',
      width: 110
    },
    {
      title: '平仓日期',
      dataIndex: 'exit_date',
      key: 'exit_date',
      width: 110
    },
    {
      title: '方向',
      dataIndex: 'direction',
      key: 'direction',
      width: 80,
      render: (direction: string) => (
        direction === 'long' ? <Text type="success">做多</Text> : <Text type="danger">做空</Text>
      ),
    },
    {
      title: '开仓比值',
      dataIndex: 'entry_ratio',
      key: 'entry_ratio',
      width: 100,
      render: (price: number) => price.toFixed(4),
    },
    {
      title: '平仓比值',
      dataIndex: 'exit_ratio',
      key: 'exit_ratio',
      width: 100,
      render: (price: number) => price.toFixed(4),
    },
    {
      title: '收益率',
      dataIndex: 'pnl_percent',
      key: 'pnl_percent',
      width: 100,
      render: (returnPct: number) => {
        const isPositive = returnPct > 0;
        return (
          <Text type={isPositive ? 'success' : 'danger'}>
            {isPositive ? '+' : ''}{returnPct.toFixed(2)}%
            {isPositive ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
          </Text>
        );
      },
    },
    {
      title: '天数',
      dataIndex: 'holding_days',
      key: 'holding_days',
      width: 70
    },
    {
      title: '平仓原因',
      dataIndex: 'exit_reason',
      key: 'exit_reason',
      width: 100
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      width: 90,
      render: (similarity: number) => `${(similarity * 100).toFixed(2)}%`,
    },
  ];

  // 相似信号表格列定义
  const signalColumns = [
    {
      title: '信号ID',
      dataIndex: 'id',
      key: 'id',
      width: 80
    },
    {
      title: '日期',
      dataIndex: 'date',
      key: 'date',
      width: 110
    },
    {
      title: '相似度',
      dataIndex: 'similarity',
      key: 'similarity',
      width: 90,
      render: (similarity: number) => `${(similarity * 100).toFixed(2)}%`,
    },
    {
      title: '信号类型',
      dataIndex: 'type',
      key: 'type',
      width: 100,
      render: (type: string) => (
        type === 'positive' ? <Text type="success">正向信号</Text> : <Text type="danger">负向信号</Text>
      ),
    },
    {
      title: '信号强度',
      dataIndex: 'strength',
      key: 'strength',
      width: 90,
      render: (strength: string) => {
        if (strength === 'strong') return <Text type="success">强</Text>;
        if (strength === 'medium') return <Text type="warning">中</Text>;
        return <Text type="danger">弱</Text>;
      },
    },
  ];

  // 图表配置
  const config = portfolio_value_history && portfolio_value_history.length > 0 ? {
    data: portfolio_value_history,
    xField: 'date',
    yField: 'portfolio_value',
    smooth: true,
    xAxis: {
      type: 'time',
      tickCount: 5,
    },
    yAxis: {
      min: Math.min(...portfolio_value_history.map((item: any) => item.portfolio_value)) * 0.95,
      max: Math.max(...portfolio_value_history.map((item: any) => item.portfolio_value)) * 1.05,
      title: {
        text: '投资组合价值',
      },
    },
    tooltip: {
      formatter: (datum: any) => {
        return { 
          name: '资金', 
          value: `¥${datum.portfolio_value.toLocaleString('zh-CN', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
          })}` 
        };
      },
    },
    meta: {
      date: {
        formatter: (value: string) => value
      },
      portfolio_value: {
        formatter: (value: number) => `¥${value.toLocaleString('zh-CN')}`
      }
    },
    color: '#1890ff',
    annotations: [
      {
        type: 'line',
        start: ['min', initial_capital],
        end: ['max', initial_capital],
        style: {
          stroke: '#888',
          lineDash: [4, 4],
        },
      },
    ],
  } : {};

  // 格式化金额显示
  const formatMoney = (value: number) => {
    return value.toLocaleString('zh-CN', {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  // 获取标题显示的股票对名称
  const getPairName = () => {
    if (stockPair) {
      // 如果名称与代码相同，说明未传入实际名称，只显示代码
      const nameA = stockPair.nameA !== stockPair.codeA ? stockPair.nameA : '';
      const nameB = stockPair.nameB !== stockPair.codeB ? stockPair.nameB : '';
      
      return `${nameA ? nameA : ''}(${stockPair.codeA})/${nameB ? nameB : ''}(${stockPair.codeB})`;
    } 
    
    if (results.current_analysis?.stock_pair) {
      return results.current_analysis.stock_pair;
    }
    
    if (code_a && code_b) {
      return `${code_a}/${code_b}`;
    }
    
    return "";
  };

  const pairName = getPairName();

  return (
    <div style={{ width: '100%', padding: '10px 0' }}>
      <Title level={4} style={{ marginBottom: 16 }}>
        相似信号回测结果{pairName ? ` - ${pairName}` : ''}
      </Title>
      
      <Row gutter={[16, 16]}>
        {/* 第一行：综合统计信息 */}
        <Col span={8}>
          <Card title="回测统计" size="small" bodyStyle={{ padding: '12px' }}>
            <Row gutter={[8, 8]}>
              <Col span={12}>
                <Statistic 
                  title="初始资金" 
                  value={formatMoney(initial_capital)} 
                  prefix="¥" 
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="最终资金" 
                  value={formatMoney(final_equity)} 
                  prefix="¥" 
                  valueStyle={{ 
                    fontSize: '16px',
                    color: final_equity >= initial_capital ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="总收益率" 
                  value={total_return} 
                  precision={2}
                  suffix="%" 
                  valueStyle={{ 
                    fontSize: '16px',
                    color: total_return >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                  prefix={total_return >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />} 
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="平均持仓天数" 
                  value={avg_holding_days} 
                  precision={1}
                  suffix="天" 
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>
        
        <Col span={8}>
          <Card title="交易绩效" size="small" bodyStyle={{ padding: '12px' }}>
            <Row gutter={[8, 8]}>
              <Col span={12}>
                <Statistic 
                  title="交易总数" 
                  value={total_trades}
                  valueStyle={{ fontSize: '16px' }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="胜率" 
                  value={win_rate} 
                  precision={2}
                  suffix="%" 
                  valueStyle={{ 
                    fontSize: '16px',
                    color: win_rate >= 50 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="盈利交易" 
                  value={winning_trades} 
                  valueStyle={{ fontSize: '16px', color: '#3f8600' }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="亏损交易" 
                  value={losing_trades} 
                  valueStyle={{ fontSize: '16px', color: '#cf1322' }}
                />
              </Col>
              <Col span={24} style={{ marginTop: 4 }}>
                <Progress 
                  percent={win_rate} 
                  status={win_rate >= 50 ? "success" : "exception"}
                  strokeColor={win_rate >= 50 ? "#3f8600" : "#cf1322"}
                  size="small"
                />
              </Col>
            </Row>
          </Card>
        </Col>
        
        <Col span={8}>
          <Card title="收益指标" size="small" bodyStyle={{ padding: '12px' }}>
            <Row gutter={[8, 8]}>
              <Col span={12}>
                <Statistic 
                  title="盈亏比" 
                  value={profit_factor} 
                  precision={2}
                  valueStyle={{ 
                    fontSize: '16px',
                    color: profit_factor >= 1 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="平均收益率" 
                  value={average_return} 
                  precision={2}
                  suffix="%" 
                  valueStyle={{ 
                    fontSize: '16px',
                    color: average_return >= 0 ? '#3f8600' : '#cf1322' 
                  }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="平均盈利" 
                  value={average_profit} 
                  precision={2}
                  suffix="%" 
                  valueStyle={{ fontSize: '16px', color: '#3f8600' }}
                />
              </Col>
              <Col span={12}>
                <Statistic 
                  title="平均亏损" 
                  value={average_loss} 
                  precision={2}
                  suffix="%" 
                  valueStyle={{ fontSize: '16px', color: '#cf1322' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 第二行：图表和相似信号 */}
        <Col span={12}>
          <Card title="投资组合价值变化" size="small" bodyStyle={{ height: '300px', padding: '8px' }}>
            {portfolio_value_history && portfolio_value_history.length > 1 ? (
              <Line {...config} height={260} />
            ) : (
              <Empty description="暂无投资组合价值变化数据" style={{ paddingTop: '100px' }} />
            )}
          </Card>
        </Col>
        
        <Col span={12}>
          <Card title="相似历史信号" size="small" bodyStyle={{ height: '300px', padding: 0 }}>
            <Table 
              columns={signalColumns} 
              dataSource={similar_signals} 
              pagination={false}
              size="small"
              rowKey="id"
              scroll={{ y: 260 }}
              style={{ height: '100%' }}
            />
          </Card>
        </Col>
        
        {/* 第三行：交易列表 */}
        <Col span={24}>
          <Card title="交易列表" size="small">
            <Table 
              columns={tradeColumns} 
              dataSource={trades} 
              pagination={{ pageSize: 5, size: 'small' }}
              scroll={{ x: 'max-content' }}
              size="small"
              rowKey="id"
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default SimilarSignalsBacktestResults; 