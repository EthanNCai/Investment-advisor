import React, { useState, useEffect } from 'react';
import { Card, Typography, Row, Col, Form, Input, Button, InputNumber, Select, Divider, Spin, Alert, Tabs, Switch, Tooltip, Space, message } from 'antd';
import { QuestionCircleOutlined, SettingOutlined, LineChartOutlined, TableOutlined, InfoCircleOutlined, BarChartOutlined } from '@ant-design/icons';
import BacktestResults from './BacktestResults';
import BacktestTradesList from './BacktestTradesList';
import SimilarSignalsBacktestResults from './SimilarSignalsBacktestResults';
import axios from 'axios';
import moment from 'moment';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface BacktestSystemProps {
  codeA: string;
  codeB: string;
  signals: any[];
}

// 最优阈值接口
interface OptimalThreshold {
  entry_threshold: number;
  exit_threshold: number;
  lookback_period: number;
  estimated_profit: number;
  win_rate: number;
  trade_count: number;
  strategy_type: string;
}

// 回测参数接口
interface BacktestParams {
  code_a: string;
  code_b: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  position_size_type: string; // 'fixed', 'percent', 'kelly'
  position_size: number;
  entry_threshold: number;
  exit_threshold: number;
  stop_loss: number;
  take_profit: number;
  max_positions: number;
  trading_fee: number;
  trailing_stop: number;
  time_stop: number;
  breakeven_stop: boolean;
  strategy_type: string;
  hedge_mode: string;
  secondary_threshold: number;
  volatility_window: number;
  trend_window: number;
  adaptive_threshold: boolean; // 是否启用自适应阈值
  adaptive_period: number; // 自适应周期
}

// 回测结果接口
interface BacktestResult {
  equity_curve: {
    date: string;
    equity: number;
    drawdown: number;
    cash: number;
    positions_value: number;
    signal: number;
    ratio: number;
  }[];
  trades: {
    id: number;
    entry_date: string;
    exit_date: string | null;
    holding_days: number;
    entry_price: number;
    exit_price: number | null;
    position_type: string;
    position_size: number;
    pnl: number;
    pnl_percent: number;
    status: 'open' | 'closed';
    exit_reason: string;
    signal_value: number;
  }[];
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
  error?: string;
}

// 相似信号回测参数接口
interface SimilarSignalsBacktestParams {
  code_a: string;
  code_b: string;
  initial_capital: number;
  position_size: number;
  stop_loss: number;
  take_profit: number;
  trading_fee: number;
  polynomial_degree: number;
  threshold_multiplier: number;
  duration: string; // 添加时间跨度参数
}

// 相似信号回测结果接口
interface SimilarSignalsBacktestResult {
  trades: {
    id: number;
    entry_date: string;
    exit_date: string;
    holding_days: number;
    entry_ratio: number;
    exit_ratio: number;
    direction: string;
    position_size: number;
    pnl: number;
    pnl_percent: number;
    exit_reason: string;
    similar_signal_id: number;
    similarity: number;
  }[];
  initial_capital: number;
  final_equity: number;
  total_return: number;
  total_trades: number;
  profitable_trades: number;
  win_rate: number;
  profit_loss_ratio: number;
  avg_holding_days: number;
  current_analysis: any;
  similar_signals: any[];
  error?: string;
}

const BacktestSystem: React.FC<BacktestSystemProps> = ({ codeA, codeB, signals }) => {
  const [form] = Form.useForm();
  const [similarSignalsForm] = Form.useForm();
  const [loading, setLoading] = useState<boolean>(false);
  const [similarSignalsLoading, setSimilarSignalsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [similarSignalsError, setSimilarSignalsError] = useState<string | null>(null);
  const [backtestResults, setBacktestResults] = useState<BacktestResult | null>(null);
  const [similarSignalsResults, setSimilarSignalsResults] = useState<SimilarSignalsBacktestResult | null>(null);
  const [activeTab, setActiveTab] = useState<string>("1");
  const [showAdvancedOptions, setShowAdvancedOptions] = useState<boolean>(false);
  const [strategyType, setStrategyType] = useState<string>("zscore");
  const [optimalThresholdLoading, setOptimalThresholdLoading] = useState<boolean>(false);
  const [optimalThreshold, setOptimalThreshold] = useState<OptimalThreshold | null>(null);

  // 提取信号日期范围
  const getDateRange = () => {
    if (!signals || signals.length === 0) {
      return { minDate: '', maxDate: '' };
    }
    
    const dates = signals.map(signal => signal.date);
    const sortedDates = [...dates].sort();
    return {
      minDate: sortedDates[0],
      maxDate: sortedDates[sortedDates.length - 1],
    };
  };

  const { minDate, maxDate } = getDateRange();

  // 监听策略类型更改
  const handleStrategyTypeChange = (value: string) => {
    setStrategyType(value);
  };

  // 初始化表单默认值
  React.useEffect(() => {
    form.setFieldsValue({
      code_a: codeA,
      code_b: codeB,
      start_date: minDate,
      end_date: maxDate,
      initial_capital: 100000,
      position_size_type: 'percent',
      position_size: 10,
      entry_threshold: 2.0,
      exit_threshold: 0.5,
      stop_loss: 5,
      take_profit: 10,
      max_positions: 5,
      trading_fee: 0.0003,
      trailing_stop: 0,
      time_stop: 0,
      breakeven_stop: false,
      strategy_type: 'zscore',
      hedge_mode: 'single',
      secondary_threshold: 1.0,
      volatility_window: 20,
      trend_window: 50,
      adaptive_threshold: false,
      adaptive_period: 60
    });
  }, [codeA, codeB, minDate, maxDate]);

  // 执行回测
  const runBacktest = async (values: any) => {
    setLoading(true);
    setError(null);
    
    try {
      // 确保股票代码被包含在请求中
      const backtestParams = {
        ...values,
        code_a: codeA,  // 明确设置股票A代码
        code_b: codeB   // 明确设置股票B代码
      };
      
      console.log('回测参数:', backtestParams); // 调试用
      
      const response = await fetch('http://localhost:8000/backtest_strategy/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(backtestParams),
      });
      
      if (!response.ok) {
        throw new Error('回测失败，请检查参数后重试');
      }
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        return;
      }
      
      setBacktestResults(data);
      setActiveTab("2"); // 自动切换到结果标签页
    } catch (err) {
      console.error('回测错误:', err);
      setError(err instanceof Error ? err.message : '执行回测时发生错误');
    } finally {
      setLoading(false);
    }
  };

  const handleFormFinish = (values: any) => {
    runBacktest(values);
  };

  // 计算最优阈值
  const calculateOptimalThreshold = async () => {
    setOptimalThresholdLoading(true);
    setError(null);
    
    try {
      const currentStrategyType = form.getFieldValue('strategy_type');
      const params = {
        code_a: codeA,
        code_b: codeB,
        lookback: 60, // 默认使用60天的回溯期
        strategy_type: currentStrategyType
      };
      
      const response = await fetch('http://localhost:8000/calculate_optimal_threshold/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });
      
      if (!response.ok) {
        throw new Error('计算最优阈值失败，请稍后重试');
      }
      
      const data = await response.json();
      
      if (data.error) {
        setError(data.error);
        return;
      }
      
      setOptimalThreshold(data);
      
      // 自动设置表单中的阈值
      form.setFieldsValue({
        entry_threshold: data.entry_threshold,
        exit_threshold: data.exit_threshold
      });
      
      // 显示成功提示
      message.success(`已设置最优阈值：入场 ${data.entry_threshold.toFixed(2)}，出场 ${data.exit_threshold.toFixed(2)}`);
    } catch (err) {
      console.error('计算最优阈值错误:', err);
      setError(err instanceof Error ? err.message : '计算最优阈值时发生错误');
    } finally {
      setOptimalThresholdLoading(false);
    }
  };

  // 运行相似信号回测
  const runSimilarSignalsBacktest = async (values: SimilarSignalsBacktestParams) => {
    setSimilarSignalsLoading(true);
    setSimilarSignalsError(null);
    
    try {
      // 确保股票代码被包含在请求中
      const backtestParams = {
        ...values,
        code_a: codeA,
        code_b: codeB
      };
      
      console.log('相似信号回测参数:', backtestParams);
      
      const response = await fetch('http://localhost:8000/backtest_similar_signals/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(backtestParams),
      });
      
      if (!response.ok) {
        throw new Error('相似信号回测失败，请检查参数后重试');
      }
      
      const data = await response.json();
      
      if (data.error) {
        setSimilarSignalsError(data.error);
        return;
      }
      
      setSimilarSignalsResults(data);
      setActiveTab("4"); // 自动切换到相似信号结果标签页
    } catch (err) {
      console.error('相似信号回测错误:', err);
      setSimilarSignalsError(err instanceof Error ? err.message : '执行相似信号回测时发生错误');
    } finally {
      setSimilarSignalsLoading(false);
    }
  };

  const handleSimilarSignalsFormFinish = (values: any) => {
    runSimilarSignalsBacktest(values);
  };

  // 初始化相似信号回测表单默认值
  useEffect(() => {
    similarSignalsForm.setFieldsValue({
      initial_capital: 100000,
      position_size: 20,
      stop_loss: 5,
      take_profit: 10,
      trading_fee: 0.0003,
      polynomial_degree: 3,
      threshold_multiplier: 1.5,
      duration: '1y'
    });
  }, []);

  return (
    <Card title="价差交易策略回测" bordered={false}>
      <Tabs 
        activeKey={activeTab} 
        onChange={setActiveTab}
        type="card"
      >
        <TabPane 
          tab={<span><SettingOutlined />回测参数</span>} 
          key="1"
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleFormFinish}
          >
            <Row gutter={16}>
              <Col span={24}>
                <Title level={5}>基本设置</Title>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="start_date"
                  label="起始日期"
                  rules={[{ required: true, message: '请选择起始日期' }]}
                >
                  <Input type="date" />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="end_date"
                  label="结束日期"
                  rules={[{ required: true, message: '请选择结束日期' }]}
                >
                  <Input type="date" />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="initial_capital"
                  label="初始资金"
                  rules={[{ required: true, message: '请输入初始资金' }]}
                >
                  <InputNumber 
                    min={1000} 
                    step={10000} 
                    style={{ width: '100%' }}
                    formatter={value => `￥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
                  />
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Title level={5}>策略设置</Title>
              </Col>

              <Col span={8}>
                <Form.Item
                  name="strategy_type"
                  label="策略类型"
                  rules={[{ required: true, message: '请选择策略类型' }]}
                >
                  <Select onChange={handleStrategyTypeChange}>
                    <Option value="zscore">Z分数策略</Option>
                    <Option value="percent">百分比偏离策略</Option>
                    <Option value="volatility">波动率调整策略</Option>
                    <Option value="trend">趋势跟踪策略</Option>
                  </Select>
                </Form.Item>
              </Col>

              <Col span={8}>
                <Form.Item
                  name="hedge_mode"
                  label="对冲模式"
                  tooltip="单边交易只做资产A，对冲交易同时做资产A和B"
                  rules={[{ required: true, message: '请选择对冲模式' }]}
                >
                  <Select>
                    <Option value="single">单边交易</Option>
                    <Option value="pair">对冲交易</Option>
                  </Select>
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Title level={5}>仓位管理</Title>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="position_size_type"
                  label="仓位计算方式"
                  rules={[{ required: true, message: '请选择仓位计算方式' }]}
                >
                  <Select>
                    <Option value="fixed">固定金额</Option>
                    <Option value="percent">资金百分比</Option>
                    <Option value="kelly">凯利公式</Option>
                  </Select>
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="position_size"
                  label="仓位大小"
                  tooltip="固定金额(元)或资金百分比(%)"
                  rules={[{ required: true, message: '请输入仓位大小' }]}
                >
                  <InputNumber 
                    min={1} 
                    step={form.getFieldValue('position_size_type') === 'fixed' ? 10000 : 5}
                    max={form.getFieldValue('position_size_type') === 'fixed' ? 1000000 : 100}
                    style={{ width: '100%' }}
                    formatter={value => 
                      form.getFieldValue('position_size_type') === 'fixed' 
                        ? `￥ ${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',') 
                        : `${value} %`
                    }
                  />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="max_positions"
                  label="最大持仓数量"
                  rules={[{ required: true, message: '请输入最大持仓数量' }]}
                >
                  <InputNumber min={1} max={25} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Title level={5}>信号参数</Title>
              </Col>
              
              <Col span={24} style={{ marginBottom: 16 }}>
                <Row gutter={16} align="middle">
                  <Col flex="auto">
                    <Text>设置信号阈值参数</Text>
                  </Col>
                  <Col>
                    <Tooltip title="自动计算最优入场和出场阈值，基于历史数据">
                      <Button 
                        type="primary" 
                        ghost
                        icon={<LineChartOutlined />}
                        loading={optimalThresholdLoading}
                        onClick={calculateOptimalThreshold}
                      >
                        计算最优阈值
                      </Button>
                    </Tooltip>
                  </Col>
                </Row>
                {optimalThreshold && (
                  <Alert
                    style={{ marginTop: 12 }}
                    message="最优阈值分析结果"
                    description={
                      <>
                        <div>策略类型: {optimalThreshold.strategy_type === 'zscore' ? 'Z分数策略' : 
                                        optimalThreshold.strategy_type === 'percent' ? '百分比偏离策略' : 
                                        optimalThreshold.strategy_type === 'volatility' ? '波动率调整策略' : '趋势跟踪策略'}</div>
                        <div>回溯天数: {optimalThreshold.lookback_period}天</div>
                        <div>建议入场阈值: <Text strong>{optimalThreshold.entry_threshold.toFixed(2)}</Text></div>
                        <div>建议出场阈值: <Text strong>{optimalThreshold.exit_threshold.toFixed(2)}</Text></div>
                        <div>估计收益率: <Text type="success">{optimalThreshold.estimated_profit.toFixed(2)}%</Text></div>
                        <div>胜率: {(optimalThreshold.win_rate * 100).toFixed(2)}%</div>
                        <div>模拟交易次数: {optimalThreshold.trade_count}次</div>
                      </>
                    }
                    type="info"
                    showIcon
                  />
                )}
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="entry_threshold"
                  label="入场阈值"
                  tooltip="信号超过该阈值时开仓"
                  rules={[{ required: true, message: '请输入入场阈值' }]}
                >
                  <InputNumber min={0.5} max={5} step={0.1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="exit_threshold"
                  label="出场阈值"
                  tooltip="信号低于该阈值时平仓"
                  rules={[{ required: true, message: '请输入出场阈值' }]}
                >
                  <InputNumber min={0.1} max={2} step={0.1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>

              {strategyType !== 'zscore' && (
                <Col span={8}>
                  <Form.Item
                    name="secondary_threshold"
                    label={
                      strategyType === 'percent' ? "百分比系数" : 
                      strategyType === 'volatility' ? "波动调整系数" : 
                      "趋势权重系数"
                    }
                    tooltip="策略特定参数，用于调整信号灵敏度"
                  >
                    <InputNumber min={0.1} max={5} step={0.1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              )}
              
              {strategyType === 'volatility' && (
                <Col span={8}>
                  <Form.Item
                    name="volatility_window"
                    label="波动率窗口"
                    tooltip="计算波动率的时间窗口（天）"
                  >
                    <InputNumber min={5} max={60} step={1} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              )}
              
              {strategyType === 'trend' && (
                <Col span={8}>
                  <Form.Item
                    name="trend_window"
                    label="趋势窗口"
                    tooltip="计算长期趋势的时间窗口（天）"
                  >
                    <InputNumber min={20} max={200} step={5} style={{ width: '100%' }} />
                  </Form.Item>
                </Col>
              )}
              
              <Col span={24}>
                <Divider />
                <Title level={5}>高级策略选项</Title>
              </Col>

              <Col span={8}>
                <Form.Item
                  name="adaptive_threshold"
                  label="自适应阈值"
                  tooltip="根据市场波动性自动调整入场和出场阈值"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>
              </Col>

              <Col span={8}>
                <Form.Item
                  name="adaptive_period"
                  label="适应周期(天)"
                  tooltip="计算自适应阈值的历史周期长度"
                >
                  <InputNumber min={20} max={120} step={10} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Row align="middle">
                  <Col span={12}>
                    <Title level={5}>风险管理</Title>
                  </Col>
                  <Col span={12} style={{ textAlign: 'right' }}>
                    <Button 
                      type="link" 
                      onClick={() => setShowAdvancedOptions(!showAdvancedOptions)}
                    >
                      {showAdvancedOptions ? '隐藏高级选项' : '显示高级选项'}
                    </Button>
                  </Col>
                </Row>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="stop_loss"
                  label="止损比例(%)"
                  tooltip="亏损达到该比例时强制平仓"
                  rules={[{ required: true, message: '请输入止损比例' }]}
                >
                  <InputNumber min={0} max={50} step={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="take_profit"
                  label="止盈比例(%)"
                  tooltip="盈利达到该比例时强制平仓"
                  rules={[{ required: true, message: '请输入止盈比例' }]}
                >
                  <InputNumber min={0} max={100} step={1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="trading_fee"
                  label="交易费率"
                  tooltip="交易成本比例，如0.0003表示万分之三"
                  rules={[{ required: true, message: '请输入交易费率' }]}
                >
                  <InputNumber 
                    min={0} 
                    max={0.01} 
                    step={0.0001}
                    style={{ width: '100%' }}
                    formatter={value => `${value} (${(Number(value) * 100).toFixed(4)}%)`}
                  />
                </Form.Item>
              </Col>
              
              {showAdvancedOptions && (
                <>
                  <Col span={8}>
                    <Form.Item
                      name="trailing_stop"
                      label="追踪止损(%)"
                      tooltip="从最高盈利点回落该比例时平仓，0表示不启用"
                    >
                      <InputNumber min={0} max={50} step={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  
                  <Col span={8}>
                    <Form.Item
                      name="time_stop"
                      label="时间止损(天)"
                      tooltip="持仓超过该天数时平仓，0表示不启用"
                    >
                      <InputNumber min={0} max={365} step={1} style={{ width: '100%' }} />
                    </Form.Item>
                  </Col>
                  
                  <Col span={8}>
                    <Form.Item
                      name="breakeven_stop"
                      label="保本止损"
                      tooltip="当曾经盈利超过3%后回落至亏损时平仓保本"
                      valuePropName="checked"
                    >
                      <Switch />
                    </Form.Item>
                  </Col>
                </>
              )}
              
              <Col span={24} style={{ textAlign: 'center', marginTop: 24 }}>
                <Form.Item>
                  <Button 
                    type="primary" 
                    htmlType="submit" 
                    loading={loading}
                    style={{ minWidth: 120 }}
                  >
                    运行回测
                  </Button>
                </Form.Item>
              </Col>
            </Row>
          </Form>
          
          {error && (
            <Alert
              message="回测错误"
              description={error}
              type="error"
              showIcon
              style={{ marginTop: 16 }}
            />
          )}
        </TabPane>
        
        <TabPane 
          tab={<span><LineChartOutlined />回测结果</span>} 
          key="2"
          disabled={!backtestResults}
        >
          {backtestResults ? (
            <BacktestResults results={backtestResults} />
          ) : (
            <div style={{ textAlign: 'center', padding: 32 }}>
              <Text type="secondary">请先运行回测</Text>
            </div>
          )}
        </TabPane>
        
        <TabPane 
          tab={<span><TableOutlined />交易明细</span>} 
          key="3"
          disabled={!backtestResults}
        >
          {backtestResults ? (
            <BacktestTradesList trades={backtestResults.trades} />
          ) : (
            <div style={{ textAlign: 'center', padding: 32 }}>
              <Text type="secondary">请先运行回测</Text>
            </div>
          )}
        </TabPane>
        
        <TabPane 
          tab={<span><BarChartOutlined /> 相似信号回测</span>} 
          key="4"
        >
          <Row gutter={[24, 24]}>
            <Col span={24}>
              <Alert
                message="基于相似历史信号的回测"
                description="此功能基于当前价格比值位置找出历史上最相似的信号，并模拟这些信号在当时的交易效果，帮助您理解当前投资机会。"
                type="info"
                showIcon
              />
            </Col>
            
            <Col span={similarSignalsResults ? 6 : 24}>
              <Form
                form={similarSignalsForm}
                layout="vertical"
                onFinish={handleSimilarSignalsFormFinish}
              >
                <Row gutter={[16, 0]}>
                  <Col span={24}>
                    <Form.Item
                      name="initial_capital"
                      label="初始资金"
                      rules={[{ required: true, message: '请输入初始资金' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={1000}
                        max={10000000}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col span={24}>
                    <Form.Item
                      name="position_size"
                      label={
                        <span>
                          仓位大小(%) 
                          <Tooltip title="每笔交易使用的资金百分比">
                            <QuestionCircleOutlined style={{ marginLeft: 4 }} />
                          </Tooltip>
                        </span>
                      }
                      rules={[{ required: true, message: '请输入仓位大小' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={1}
                        max={100}
                      />
                    </Form.Item>
                  </Col>
                </Row>
                
                <Row gutter={[16, 0]}>
                  <Col span={24}>
                    <Form.Item
                      name="stop_loss"
                      label="止损比例(%)"
                      rules={[{ required: true, message: '请输入止损比例' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={0.1}
                        max={50}
                        step={0.1}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col span={24}>
                    <Form.Item
                      name="take_profit"
                      label="止盈比例(%)"
                      rules={[{ required: true, message: '请输入止盈比例' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={0.1}
                        max={50}
                        step={0.1}
                      />
                    </Form.Item>
                  </Col>
                </Row>
                
                <Row gutter={[16, 0]}>
                  <Col span={24}>
                    <Form.Item
                      name="polynomial_degree"
                      label={
                        <span>
                          多项式拟合次数 
                          <Tooltip title="用于拟合价格比值曲线的多项式次数，较高的值能捕捉更复杂的曲线形状">
                            <QuestionCircleOutlined style={{ marginLeft: 4 }} />
                          </Tooltip>
                        </span>
                      }
                      rules={[{ required: true, message: '请输入多项式拟合次数' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={1}
                        max={10}
                        step={1}
                      />
                    </Form.Item>
                  </Col>
                  
                  <Col span={24}>
                    <Form.Item
                      name="threshold_multiplier"
                      label={
                        <span>
                          信号阈值系数 
                          <Tooltip title="用于生成投资信号的阈值乘数，较高的值会产生更少但更可靠的信号">
                            <QuestionCircleOutlined style={{ marginLeft: 4 }} />
                          </Tooltip>
                        </span>
                      }
                      rules={[{ required: true, message: '请输入阈值系数' }]}
                    >
                      <InputNumber
                        style={{ width: '100%' }}
                        min={1}
                        max={5}
                        step={0.1}
                      />
                    </Form.Item>
                  </Col>
                </Row>
                
                <Row gutter={[16, 0]}>
                  <Col span={24}>
                    <Form.Item
                      name="duration"
                      label="时间跨度"
                      tooltip="选择历史数据分析的时间范围"
                      rules={[{ required: true, message: '请选择时间跨度' }]}
                    >
                      <Select>
                        <Option value="1m">1个月</Option>
                        <Option value="3m">3个月</Option>
                        <Option value="1y">1年</Option>
                        <Option value="2y">2年</Option>
                        <Option value="5y">5年</Option>
                        <Option value="maximum">全部</Option>
                      </Select>
                    </Form.Item>
                  </Col>
                </Row>
                
                <Form.Item
                  name="trading_fee"
                  label="交易费率"
                  rules={[{ required: true, message: '请输入交易费率' }]}
                >
                  <InputNumber
                    style={{ width: '100%' }}
                    min={0}
                    max={0.01}
                    step={0.0001}
                    precision={6}
                  />
                </Form.Item>
                
                <Form.Item>
                  <Button
                    type="primary"
                    htmlType="submit"
                    loading={similarSignalsLoading}
                    icon={<LineChartOutlined />}
                    block
                  >
                    运行相似信号回测
                  </Button>
                </Form.Item>
              </Form>
              
              {similarSignalsError && (
                <Alert
                  message="回测错误"
                  description={similarSignalsError}
                  type="error"
                  showIcon
                  style={{ marginTop: 16 }}
                />
              )}
            </Col>
            
            {similarSignalsResults && (
              <Col span={18}>
                <SimilarSignalsBacktestResults 
                  results={similarSignalsResults} 
                  loading={similarSignalsLoading} 
                  stockPair={{
                    codeA: codeA,
                    nameA: codeA, // 这里应当从全局状态获取股票名称，临时用代码代替
                    codeB: codeB,
                    nameB: codeB, // 这里应当从全局状态获取股票名称，临时用代码代替
                  }}
                />
              </Col>
            )}
          </Row>
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default BacktestSystem; 