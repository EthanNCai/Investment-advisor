import React, { useState, useEffect, useCallback } from 'react';
import { Card, Spin, Tabs, message, Space, Typography, Alert, Row, Col, Select, Input } from 'antd';
import { useLocalStorage } from '../../LocalStorageContext';
import SignalList from './SignalList';
import SignalDetail from './SignalDetail';
import CurrentPositionAnalysis from './CurrentPositionAnalysis';
import SignalFilterBar from './SignalFilterBar';
import BacktestSystem from './BacktestSystem';
import SignalStatsCard from './SignalStatsCard';
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
  quality_evaluation?: any; // 添加质量评分字段
  record_id?: number; // 添加记录ID字段
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
  
  // 添加追踪信号开关
  const [trackSignals, setTrackSignals] = useLocalStorage<boolean>('track-signals', true);
  
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
  
  // 页面布局状态
  const [activeMainTab, setActiveMainTab] = useLocalStorage<string>('signal-main-tab', '1');
  
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
  }, [selectedStockA, selectedStockB, selectedDuration, degree, thresholdArg, trackSignals]);
  
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
          threshold_arg: thresholdArg,
          track_signals: trackSignals
        }),
      });
      
      const data = await response.json();
      
      if (data && data.signals) {
        setSignals(data.signals);
        
        // 确保当前位置信息的处理支持新字段
        if (data.current_position) {
          // 兼容处理：如果是旧格式数据，转换为新格式
          if (data.current_position.nearest_signal_id && !data.current_position.nearest_signals) {
            // 旧格式数据兼容
            const nearestSignal = data.signals.find((s: Signal) => s.id === data.current_position.nearest_signal_id);
            data.current_position.nearest_signals = nearestSignal ? 
              [{
                id: nearestSignal.id,
                date: nearestSignal.date,
                ratio: nearestSignal.ratio,
                similarity: data.current_position.similarity_score || 0,
                type: nearestSignal.type,
                strength: nearestSignal.strength
              }] : [];
          }
          
          // 确保所有新字段存在
          data.current_position.z_score = data.current_position.z_score || null;
          data.current_position.deviation_from_trend = data.current_position.deviation_from_trend || null;
          data.current_position.volatility_level = data.current_position.volatility_level || null;
          data.current_position.historical_signal_pattern = data.current_position.historical_signal_pattern || null;
          
          // 确保新增的分析字段存在
          data.current_position.trend_strength = data.current_position.trend_strength || null;
          data.current_position.support_resistance = data.current_position.support_resistance || null;
          data.current_position.mean_reversion_probability = data.current_position.mean_reversion_probability || null;
          data.current_position.cycle_position = data.current_position.cycle_position || null;
          
          setCurrentPositionInfo(data.current_position);
        } else {
          setCurrentPositionInfo(null);
        }
        
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
          
          {/* 参数设置和过滤器 */}
          <SignalFilterBar
            duration={selectedDuration}
            onDurationChange={setSelectedDuration}
            degree={degree}
            onDegreeChange={setDegree}
            threshold={thresholdArg}
            onThresholdChange={setThresholdArg}
            trackSignals={trackSignals}
            onTrackSignalsChange={setTrackSignals}
            signalStrength={signalStrength}
            onSignalStrengthChange={setSignalStrength}
            signalType={signalType}
            onSignalTypeChange={setSignalType}
          />
          
          {loading && (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <Spin />
              <div style={{ marginTop: 8 }}>加载信号数据...</div>
            </div>
          )}
          
          {/* 主体内容区域 */}
          <Tabs activeKey={activeMainTab} onChange={setActiveMainTab}>
            <TabPane tab="信号列表" key="1">
              <Row gutter={16}>
                <Col span={12}>
                  <SignalList 
                    signals={filteredSignals} 
                    selectedSignal={selectedSignal} 
                    onSignalSelect={handleSignalSelect} 
                  />
                </Col>
                <Col span={12}>
                  {selectedSignal ? (
                    <SignalDetail signal={selectedSignal} />
                  ) : (
                    <Card>
                      <div style={{ textAlign: 'center', padding: '40px 0' }}>
                        <Text type="secondary">请从左侧列表选择一个信号查看详情</Text>
                      </div>
                    </Card>
                  )}
                </Col>
              </Row>
            </TabPane>
            
            <TabPane tab="当前市场分析" key="2">
              <CurrentPositionAnalysis 
                currentPosition={currentPositionInfo} 
                loading={loading} 
              />
            </TabPane>
            
            <TabPane tab="信号统计" key="3">
              <SignalStatsCard />
            </TabPane>
            
            <TabPane tab="回测系统" key="4">
              <BacktestSystem 
                stockA={selectedStockA} 
                stockB={selectedStockB} 
              />
            </TabPane>
          </Tabs>
        </Space>
      </Card>
    </div>
  );
};

export default InvestmentSignal; 