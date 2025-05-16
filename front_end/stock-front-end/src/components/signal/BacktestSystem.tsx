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
  anomaly_threshold: number;
  polynomial_degree: number;
  mean_reversion_threshold: number;
  estimated_profit: number;
  win_rate: number;
  trade_count: number;
  lookback_period?: number;
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
  stop_loss: number;
  take_profit: number;
  max_positions: number;
  trading_fee: number;
  trailing_stop: number;
  time_stop: number;
  hedge_mode: string;
  // 新的异常点检测参数
  polynomial_degree: number; // 多项式拟合次数
  anomaly_threshold: number; // 异常检测阈值
  mean_reversion_exit: boolean; // 回归均值出场
  mean_reversion_threshold: number; // 回归均值阈值
  reverse_anomaly_exit: boolean; // 反向异常点出场
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
    anomaly_signal: number;
    delta?: number;
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
    anomaly_threshold: number;
    polynomial_degree: number;
    stop_loss: number;
    take_profit: number;
    trailing_stop: number;
    time_stop: number;
    hedge_mode: string;
    position_size_type: string;
    position_size: number;
    mean_reversion_exit?: boolean;
    mean_reversion_threshold?: number;
    reverse_anomaly_exit?: boolean;
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
      stop_loss: 5,
      take_profit: 10,
      max_positions: 5,
      trading_fee: 0.0003,
      trailing_stop: 0,
      time_stop: 0,
      hedge_mode: 'single',
      // 设置新参数的默认值
      polynomial_degree: 3,
      anomaly_threshold: 2.0,
      mean_reversion_exit: true,
      mean_reversion_threshold: 0.5,
      reverse_anomaly_exit: true
    });
  }, [codeA, codeB, minDate, maxDate, form]);

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
                <Divider />
                <Title level={5}>策略设置</Title>
              </Col>

              <Col span={24}>
                <Alert
                  message="基于异常点检测的回测策略"
                  description={
                    <>
                      <p>此回测策略基于价格比值曲线与拟合曲线之间的异常偏离点进行交易。系统检测到的异常点将作为入场信号，出场则基于多种策略组合。</p>
                      <p><strong>入场信号</strong>：当价格比值相对于拟合曲线出现显著偏离（由异常点阈值控制），系统将识别为交易机会。</p>
                      <p><strong>出场策略</strong>：可通过以下方式触发
                        <ul>
                          <li>价格比值回归到拟合曲线附近（回归均值出场）</li>
                          <li>出现反向异常点信号（反向异常点出场）</li>
                          <li>触发止盈止损条件</li>
                          <li>到达最大持仓时间</li>
                        </ul>
                      </p>
                    </>
                  }
                  type="info"
                  showIcon
                  style={{ marginBottom: 24 }}
                />
              </Col>

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
                <Title level={5}>异常点检测参数</Title>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="polynomial_degree"
                  label="多项式拟合次数"
                  tooltip="用于拟合价格比值曲线的多项式次数，数值越高拟合越精确但也可能过拟合"
                  rules={[{ required: true, message: '请选择多项式拟合次数' }]}
                >
                  <Select>
                    <Option value={1}>1次多项式</Option>
                    <Option value={2}>2次多项式</Option>
                    <Option value={3}>3次多项式</Option>
                    <Option value={4}>4次多项式</Option>
                    <Option value={5}>5次多项式</Option>
                    <Option value={6}>6次多项式</Option>
                  </Select>
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="anomaly_threshold"
                  label="异常点阈值"
                  tooltip="检测异常点的阈值倍数，越大检测到的异常点越少但信号可靠性更高"
                  rules={[{ required: true, message: '请输入异常点阈值' }]}
                >
                  <InputNumber min={1.0} max={5.0} step={0.1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Title level={5}>出场策略设置</Title>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="mean_reversion_exit"
                  label="回归均值出场"
                  tooltip="当价格比值回归到拟合曲线附近时平仓"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="mean_reversion_threshold"
                  label="回归阈值倍数"
                  tooltip="回归到拟合线多近时触发平仓（标准差的倍数）"
                >
                  <InputNumber min={0.1} max={1.0} step={0.1} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="reverse_anomaly_exit"
                  label="反向异常点出场"
                  tooltip="当出现与持仓方向相反的异常点信号时平仓"
                  valuePropName="checked"
                >
                  <Switch />
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