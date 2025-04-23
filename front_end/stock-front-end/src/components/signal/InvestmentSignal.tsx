import React, { useState, useEffect, useCallback } from 'react';
import { Card, Spin, Tabs, message, Space, Typography, Alert, Row, Col, Select, Input } from 'antd';
import { useLocalStorage } from '../../LocalStorageContext';
import SignalList from './SignalList';
import SignalDetail from './SignalDetail';
import CurrentPositionAnalysis from './CurrentPositionAnalysis';
import SignalFilterBar from './SignalFilterBar';
import { SearchOutlined } from '@ant-design/icons';
import axios from 'axios';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

// 定义信号接口
export interface Signal {
  id: number;
  date: string;
  ratio: number;
  z_score: number;
  type: 'positive' | 'negative';
  strength: 'weak' | 'medium' | 'strong';
  description: string;
  recommendation: string;
}

// 股票信息接口
interface StockInfo {
  code: string;
  name: string;
  type?: string;
}

const InvestmentSignal: React.FC = () => {
  // 复用现有的股票选择状态
  const [selectedStockA, setSelectedStockA] = useLocalStorage<string>('ratio-analysis-stockA', '');
  const [selectedStockB, setSelectedStockB] = useLocalStorage<string>('ratio-analysis-stockB', '');
  const [selectedDuration, setSelectedDuration] = useLocalStorage<string>('ratio-analysis-duration', "1y");
  
  // 添加拟合阶数和阈值系数状态
  const [degree, setDegree] = useLocalStorage<number>('signal-analysis-degree', 3);
  const [thresholdArg, setThresholdArg] = useLocalStorage<number>('signal-analysis-threshold', 2.0);
  
  // 股票搜索相关状态
  const [stockAOptions, setStockAOptions] = useState<StockInfo[]>([]);
  const [stockBOptions, setStockBOptions] = useState<StockInfo[]>([]);
  const [stockASearch, setStockASearch] = useState<string>('');
  const [stockBSearch, setStockBSearch] = useState<string>('');
  const [loadingStocksA, setLoadingStocksA] = useState<boolean>(false);
  const [loadingStocksB, setLoadingStocksB] = useState<boolean>(false);
  
  // 信号相关状态
  const [signals, setSignals] = useState<Signal[]>([]);
  const [selectedSignal, setSelectedSignal] = useState<Signal | null>(null);
  const [currentPositionInfo, setCurrentPositionInfo] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(false);
  
  // 过滤条件 - 始终使用数组初始化
  const [signalStrength, setSignalStrength] = useLocalStorage<string[]>('signalStrength', ['weak', 'medium', 'strong']);
  const [signalType, setSignalType] = useLocalStorage<string[]>('signalType', ['positive', 'negative']);
  
  // 初始加载时获取常用股票列表
  useEffect(() => {
    if (selectedStockA) {
      searchStocks(selectedStockA, 'A');
    }
    if (selectedStockB) {
      searchStocks(selectedStockB, 'B');
    }
  }, []);
  
  // 当股票选择或参数变化时获取信号
  useEffect(() => {
    if (selectedStockA && selectedStockB) {
      fetchSignals();
    }
  }, [selectedStockA, selectedStockB, selectedDuration, degree, thresholdArg]);
  
  // 获取信号数据
  const fetchSignals = async () => {
    if (!selectedStockA || !selectedStockB) {
      message.warning('请先选择两只股票');
      return;
    }
    
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/get_investment_signals/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code_a: selectedStockA,
          code_b: selectedStockB,
          duration: selectedDuration,
          degree: degree,
          threshold_arg: thresholdArg
        }),
      });
      
      const data = await response.json();
      
      if (data && data.signals) {
        setSignals(data.signals);
        setCurrentPositionInfo(data.current_position || null);
        
        // 自动选择第一个信号
        if (data.signals.length > 0) {
          setSelectedSignal(data.signals[0]);
        } else {
          setSelectedSignal(null);
        }
      }
    } catch (error) {
      console.error('获取投资信号失败:', error);
      message.error('获取投资信号失败');
    } finally {
      setLoading(false);
    }
  };
  
  // 搜索股票
  const searchStocks = async (keyword: string, type: 'A' | 'B') => {
    if (!keyword) return;
    
    const setLoading = type === 'A' ? setLoadingStocksA : setLoadingStocksB;
    const setOptions = type === 'A' ? setStockAOptions : setStockBOptions;
    
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/search_stocks/${encodeURIComponent(keyword)}`);
      const data = await response.json();
      
      if (data && data.result) {
        setOptions(data.result);
      }
    } catch (error) {
      console.error('搜索股票失败:', error);
      message.error('搜索股票失败');
    } finally {
      setLoading(false);
    }
  };
  
  // 处理搜索输入变化
  const handleSearchChange = (value: string, type: 'A' | 'B') => {
    if (type === 'A') {
      setStockASearch(value);
    } else {
      setStockBSearch(value);
    }
    
    if (value) {
      searchStocks(value, type);
    }
  };
  
  // 处理股票选择
  const handleStockSelect = (value: string, type: 'A' | 'B') => {
    if (type === 'A') {
      setSelectedStockA(value);
    } else {
      setSelectedStockB(value);
    }
  };
  
  const handleSignalSelect = (signal: Signal) => {
    setSelectedSignal(signal);
  };
  
  // 安全检查，确保signals是数组
  const safeSignals = Array.isArray(signals) ? signals : [];
  
  const filteredSignals = safeSignals.filter(signal => 
    signalStrength.includes(signal.strength) && 
    signalType.includes(signal.type)
  );
  
  return (
    <div className="investment-signal-page">
      <Card title="投资信号分析" bordered={false}>
        <Space direction="vertical" style={{ width: '100%' }}>
          {/* 股票选择区域 */}
          <Card 
            type="inner" 
            title="选择股票对"
            style={{ marginBottom: 16 }}
          >
            <Row gutter={16}>
              <Col span={11}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>股票A:</Text>
                  <Select
                    showSearch
                    value={selectedStockA}
                    placeholder="搜索股票代码或名称"
                    suffixIcon={<SearchOutlined />}
                    filterOption={false}
                    onSearch={(value) => handleSearchChange(value, 'A')}
                    onChange={(value) => handleStockSelect(value, 'A')}
                    notFoundContent={loadingStocksA ? <Spin size="small" /> : null}
                    style={{ width: '100%' }}
                  >
                    {stockAOptions.map((option) => (
                      <Option key={option.code} value={option.code}>
                        {option.code} - {option.name} {option.type ? `(${option.type})` : ''}
                      </Option>
                    ))}
                  </Select>
                </Space>
              </Col>
              <Col span={2} style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                <Text strong>/</Text>
              </Col>
              <Col span={11}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text strong>股票B:</Text>
                  <Select
                    showSearch
                    value={selectedStockB}
                    placeholder="搜索股票代码或名称"
                    suffixIcon={<SearchOutlined />}
                    filterOption={false}
                    onSearch={(value) => handleSearchChange(value, 'B')}
                    onChange={(value) => handleStockSelect(value, 'B')}
                    notFoundContent={loadingStocksB ? <Spin size="small" /> : null}
                    style={{ width: '100%' }}
                  >
                    {stockBOptions.map((option) => (
                      <Option key={option.code} value={option.code}>
                        {option.code} - {option.name} {option.type ? `(${option.type})` : ''}
                      </Option>
                    ))}
                  </Select>
                </Space>
              </Col>
            </Row>
          </Card>
          
          {(!selectedStockA || !selectedStockB) && (
            <Alert
              message="请先选择股票"
              description="请使用上方的搜索框选择两只股票后查看投资信号。"
              type="info"
              showIcon
            />
          )}
          
          {selectedStockA && selectedStockB && (
            <>
              <div className="signal-header">
                <Title level={4}>
                  {`${selectedStockA}/${selectedStockB} 投资信号分析`}
                </Title>
                <Text type="secondary">
                  基于历史价格比值异常检测生成的投资信号
                </Text>
              </div>
              
              <SignalFilterBar
                signalStrength={signalStrength}
                setSignalStrength={setSignalStrength}
                signalType={signalType}
                setSignalType={setSignalType}
                selectedDuration={selectedDuration}
                setSelectedDuration={setSelectedDuration}
                degree={degree}
                setDegree={setDegree}
                thresholdArg={thresholdArg}
                setThresholdArg={setThresholdArg}
              />
              
              <Spin spinning={loading}>
                <Tabs defaultActiveKey="1">
                  <TabPane tab="信号列表" key="1">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <SignalList 
                        signals={filteredSignals} 
                        selectedSignal={selectedSignal}
                        onSignalSelect={handleSignalSelect}
                      />
                      
                      {selectedSignal && (
                        <SignalDetail signal={selectedSignal} />
                      )}
                    </Space>
                  </TabPane>
                  <TabPane tab="当前位置分析" key="2">
                    <CurrentPositionAnalysis 
                      currentPositionInfo={currentPositionInfo}
                      signals={safeSignals}
                    />
                  </TabPane>
                </Tabs>
              </Spin>
            </>
          )}
        </Space>
      </Card>
    </div>
  );
};

export default InvestmentSignal; 