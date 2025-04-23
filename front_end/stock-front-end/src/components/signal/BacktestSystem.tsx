import React, { useState } from 'react';
import { Card, Typography, Row, Col, Form, Input, Button, InputNumber, Select, Divider, Spin, Alert, Tabs } from 'antd';
import { QuestionCircleOutlined, SettingOutlined, LineChartOutlined, TableOutlined } from '@ant-design/icons';
import BacktestResults from './BacktestResults';
import BacktestTradesList from './BacktestTradesList';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;

interface BacktestSystemProps {
  codeA: string;
  codeB: string;
  signals: any[];
}

// 回测参数接口
interface BacktestParams {
  code_a: string;
  code_b: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  position_size_type: string; // 'fixed', 'percent'
  position_size: number;
  entry_threshold: number;
  exit_threshold: number;
  stop_loss: number;
  take_profit: number;
  max_positions: number;
  trading_fee: number;
}

// 回测结果接口
interface BacktestResult {
  equity_curve: {
    date: string;
    equity: number;
    drawdown: number;
  }[];
  trades: {
    id: number;
    entry_date: string;
    exit_date: string | null;
    entry_price: number;
    exit_price: number | null;
    position_type: string; // "long" | "short"
    position_size: number;
    pnl: number;
    pnl_percent: number;
    status: string; // "open" | "closed"
  }[];
  metrics: {
    total_return: number;
    annual_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
  };
}

const BacktestSystem: React.FC<BacktestSystemProps> = ({ codeA, codeB, signals }) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [backtestResults, setBacktestResults] = useState<BacktestResult | null>(null);
  const [activeTab, setActiveTab] = useState<string>("1");

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
      entry_threshold: 2.0,
      exit_threshold: 0.5,
      stop_loss: 5,
      take_profit: 10,
      max_positions: 5,
      trading_fee: 0.0003,
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
                  <InputNumber min={1} max={20} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Title level={5}>交易规则</Title>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="entry_threshold"
                  label="入场阈值(Z-score)"
                  tooltip="当价差Z-score超过该阈值时入场"
                  rules={[{ required: true, message: '请输入入场阈值' }]}
                >
                  <InputNumber 
                    min={0.5} 
                    max={5} 
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="exit_threshold"
                  label="出场阈值(Z-score)"
                  tooltip="当价差Z-score低于该阈值时出场"
                  rules={[{ required: true, message: '请输入出场阈值' }]}
                >
                  <InputNumber 
                    min={0.1} 
                    max={3} 
                    step={0.1}
                    style={{ width: '100%' }}
                  />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="trading_fee"
                  label="交易费率"
                  tooltip="交易金额的百分比"
                  rules={[{ required: true, message: '请输入交易费率' }]}
                >
                  <InputNumber 
                    min={0} 
                    max={0.01} 
                    step={0.0001}
                    style={{ width: '100%' }}
                    formatter={value => `${(Number(value) * 100).toFixed(2)} %`}
                  />
                </Form.Item>
              </Col>
              
              <Col span={24}>
                <Divider />
                <Title level={5}>风险管理</Title>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="stop_loss"
                  label="止损比例(%)"
                  tooltip="价格移动超过该百分比时触发止损"
                  rules={[{ required: true, message: '请输入止损比例' }]}
                >
                  <InputNumber min={1} max={20} step={0.5} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={8}>
                <Form.Item
                  name="take_profit"
                  label="止盈比例(%)"
                  tooltip="价格移动超过该百分比时触发止盈"
                  rules={[{ required: true, message: '请输入止盈比例' }]}
                >
                  <InputNumber min={1} max={50} step={0.5} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
              
              <Col span={24} style={{ marginTop: 16 }}>
                <Form.Item>
                  <Button type="primary" htmlType="submit" loading={loading} style={{ marginRight: 16 }}>
                    运行回测
                  </Button>
                  <Button onClick={() => form.resetFields()}>
                    重置参数
                  </Button>
                </Form.Item>
              </Col>
              
              {error && (
                <Col span={24}>
                  <Alert message="回测错误" description={error} type="error" showIcon />
                </Col>
              )}
            </Row>
          </Form>
        </TabPane>
        
        <TabPane 
          tab={<span><LineChartOutlined />回测结果</span>} 
          key="2"
          disabled={!backtestResults}
        >
          {loading ? (
            <div style={{ textAlign: 'center', padding: '30px' }}>
              <Spin tip="回测执行中..." />
            </div>
          ) : backtestResults ? (
            <BacktestResults results={backtestResults} />
          ) : (
            <Alert 
              message="尚未运行回测" 
              description="请在参数设置选项卡中设置参数并运行回测" 
              type="info" 
              showIcon 
            />
          )}
        </TabPane>
        
        <TabPane 
          tab={<span><TableOutlined />交易明细</span>} 
          key="3"
          disabled={!backtestResults}
        >
          {loading ? (
            <div style={{ textAlign: 'center', padding: '30px' }}>
              <Spin tip="回测执行中..." />
            </div>
          ) : backtestResults ? (
            <BacktestTradesList trades={backtestResults.trades} />
          ) : (
            <Alert 
              message="尚未运行回测" 
              description="请在参数设置选项卡中设置参数并运行回测" 
              type="info" 
              showIcon 
            />
          )}
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default BacktestSystem; 