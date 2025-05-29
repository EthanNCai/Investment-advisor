import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Select, Button, Space, Slider, Spin, message, Tabs, Alert, Tooltip, Tag } from 'antd';
import { QuestionCircleOutlined, SettingOutlined, ReloadOutlined, SaveOutlined } from '@ant-design/icons';
import axios from 'axios';
import AssetSelector from './AssetSelector';
import AssetCorrelationMatrix from './AssetCorrelationMatrix';
import AssetPairChart from './AssetPairChart';
import OptimalPairsTable from './OptimalPairsTable';
import AssetStrengthRanking from './AssetStrengthRanking';
import AllPairCharts from './AllPairCharts';

const { Option } = Select;
const { TabPane } = Tabs;

// 可选的K线类型
const klineTypes = [
  { value: 'daily', label: '日K' },
  { value: 'weekly', label: '周K' },
  { value: 'monthly', label: '月K' },
  { value: 'yearly', label: '年K' }
];

// 可选的时间跨度
const durationOptions = [
  { value: 'maximum', label: '全部' },
  { value: '5y', label: '5年' },
  { value: '2y', label: '2年' },
  { value: '1y', label: '1年' },
  { value: '1q', label: '3个月' },
  { value: '1m', label: '1个月' }
];

// 缓存键名
const CACHE_KEY = 'multi_asset_analysis_cache';

const MultiAssetAnalysis: React.FC = () => {
  // 状态管理
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);
  const [duration, setDuration] = useState<string>('2y');
  const [klineType, setKlineType] = useState<string>('daily');
  const [polynomialDegree, setPolynomialDegree] = useState<number>(3);
  const [thresholdMultiplier, setThresholdMultiplier] = useState<number>(2.0);
  const [loading, setLoading] = useState<boolean>(false);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [selectedPair, setSelectedPair] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<string>('1');
  const [isCachedData, setIsCachedData] = useState<boolean>(false);

  // 从缓存加载数据
  useEffect(() => {
    try {
      const cachedData = localStorage.getItem(CACHE_KEY);
      if (cachedData) {
        const parsedData = JSON.parse(cachedData);
        // 恢复参数和分析结果
        if (parsedData.params) {
          setSelectedAssets(parsedData.params.assets || []);
          setDuration(parsedData.params.duration || '2y');
          setKlineType(parsedData.params.kline_type || 'daily');
          setPolynomialDegree(parsedData.params.polynomial_degree || 3);
          setThresholdMultiplier(parsedData.params.threshold_multiplier || 2.0);
        }
        if (parsedData.result) {
          setAnalysisData(parsedData.result);
          setIsCachedData(true);
          // 恢复选中的资产对
          if (parsedData.selectedPair && parsedData.selectedPair.length === 2) {
            setSelectedPair(parsedData.selectedPair);
          } else if (parsedData.params?.assets?.length >= 2) {
            setSelectedPair([parsedData.params.assets[0], parsedData.params.assets[1]]);
          }
        }
        message.info('已加载上次分析的缓存数据');
      }
    } catch (error) {
      console.error('从缓存加载数据失败:', error);
      // 静默处理缓存错误，不显示给用户
    }
  }, []);

  // 资产选择处理
  const handleAssetChange = (assets: string[]) => {
    if (assets.length > 6) {
      message.warning('最多只能选择6个资产进行比较');
      setSelectedAssets(assets.slice(0, 6));
    } else {
      setSelectedAssets(assets);
    }
  };

  // 选择特定资产对进行详细查看
  const handlePairSelect = (assetA: string, assetB: string) => {
    setSelectedPair([assetA, assetB]);
  };

  // 保存到缓存
  const saveToCache = (result: any) => {
    try {
      const cacheData = {
        params: {
          assets: selectedAssets,
          duration,
          kline_type: klineType,
          polynomial_degree: polynomialDegree,
          threshold_multiplier: thresholdMultiplier
        },
        result,
        selectedPair,
        timestamp: new Date().toISOString()
      };
      localStorage.setItem(CACHE_KEY, JSON.stringify(cacheData));
    } catch (error) {
      console.error('保存缓存失败:', error);
      // 静默处理缓存错误
    }
  };

  // 清除缓存
  const clearCache = () => {
    try {
      localStorage.removeItem(CACHE_KEY);
      message.success('已清除缓存数据');
      setIsCachedData(false);
    } catch (error) {
      console.error('清除缓存失败:', error);
    }
  };

  // 运行分析
  const runAnalysis = async () => {
    if (selectedAssets.length < 2) {
      message.warning('请至少选择2个资产进行比较');
      return;
    }

    setLoading(true);
    setIsCachedData(false);
    
    try {
      const response = await axios.post('http://localhost:8000/multi_asset_analysis/', {
        assets: selectedAssets,
        duration,
        kline_type: klineType,
        polynomial_degree: polynomialDegree,
        threshold_multiplier: thresholdMultiplier
      });

      if (response.data) {
        setAnalysisData(response.data);
        // 保存到缓存
        saveToCache(response.data);
        
        // 默认选择前两个资产作为详细查看的对象
        if (selectedAssets.length >= 2 && !selectedPair.length) {
          setSelectedPair([selectedAssets[0], selectedAssets[1]]);
        }
      }
    } catch (error) {
      console.error('分析失败:', error);
      message.error('多资产分析失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 获取所选资产对的详细分析
  const fetchPairAnalysis = async () => {
    if (!selectedPair || selectedPair.length !== 2) return;

    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/asset_pair_analysis/', {
        code_a: selectedPair[0],
        code_b: selectedPair[1],
        duration,
        kline_type: klineType,
        polynomial_degree: polynomialDegree,
        threshold_multiplier: thresholdMultiplier
      });

      if (response.data) {
        const updatedData = {
          ...analysisData,
          pairDetail: response.data
        };
        
        // 更新当前选中资产对的详细数据
        setAnalysisData(updatedData);
        
        // 更新缓存
        saveToCache(updatedData);
      }
    } catch (error) {
      console.error('获取资产对详情失败:', error);
      message.error('获取资产对详情失败，请稍后重试');
    } finally {
      setLoading(false);
    }
  };

  // 当选中资产对变化时获取详细分析
  useEffect(() => {
    if (selectedPair.length === 2) {
      fetchPairAnalysis();
    }
  }, [selectedPair]);

  return (
    <div className="multi-asset-analysis">
      <Card 
        title="多资产联合分析" 
        bordered={false}
        extra={isCachedData ? (
          <Space>
            <Tooltip title="这是缓存的结果，点击重新分析获取最新数据">
              <Tag color="orange">缓存数据</Tag>
            </Tooltip>
            <Button 
              icon={<ReloadOutlined />} 
              size="small" 
              onClick={clearCache}
            >
              清除缓存
            </Button>
          </Space>
        ) : null}
      >
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={24}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="多资产联合分析可以同时比较多个资产的相对价值和投资机会"
                description="选择2-6个资产进行对比分析，系统将计算所有可能的资产对并找出最佳交易机会。"
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              <Card size="small" title="分析设置">
                <Row gutter={[16, 16]}>
                  <Col xs={24} md={12}>
                    <AssetSelector 
                      value={selectedAssets} 
                      onChange={handleAssetChange}
                      maxCount={6}
                    />
                  </Col>
                  <Col xs={24} md={12}>
                    <Space wrap style={{ width: '100%' }}>
                      <span>K线类型:</span>
                      <Select 
                        value={klineType} 
                        onChange={setKlineType}
                        style={{ width: 120 }}
                      >
                        {klineTypes.map(option => (
                          <Option key={option.value} value={option.value}>
                            {option.label}
                          </Option>
                        ))}
                      </Select>
                      
                      <span>时间跨度:</span>
                      <Select 
                        value={duration} 
                        onChange={setDuration}
                        style={{ width: 120 }}
                      >
                        {durationOptions.map(option => (
                          <Option key={option.value} value={option.value}>
                            {option.label}
                          </Option>
                        ))}
                      </Select>
                    </Space>
                    
                    <div style={{ marginTop: 16 }}>
                      <Space style={{ width: '100%' }}>
                        <span>
                          拟合曲线次数: 
                          <Tooltip title="多项式拟合曲线的次数，越高拟合越精确但可能过拟合">
                            <QuestionCircleOutlined style={{ marginLeft: 6 }} />
                          </Tooltip>
                        </span>
                        <Slider
                          min={1}
                          max={6}
                          step={1}
                          value={polynomialDegree}
                          onChange={setPolynomialDegree}
                          style={{ width: 150 }}
                        />
                        <span>{polynomialDegree}</span>
                      </Space>
                    </div>
                    
                    <div style={{ marginTop: 16 }}>
                      <Space style={{ width: '100%' }}>
                        <span>
                          异常值阈值: 
                          <Tooltip title="检测异常值的灵敏度，值越小检测越灵敏">
                            <QuestionCircleOutlined style={{ marginLeft: 5 }} />
                          </Tooltip>
                        </span>
                        <Slider
                          min={1}
                          max={3}
                          step={0.1}
                          value={thresholdMultiplier}
                          onChange={setThresholdMultiplier}
                          style={{ width: 150 }}
                        />
                        <span>{thresholdMultiplier.toFixed(1)}</span>
                      </Space>
                    </div>
                  </Col>
                </Row>
                
                <Row style={{ marginTop: 16 }}>
                  <Col span={24} style={{ textAlign: 'right' }}>
                    <Space>
                      <Button 
                        type="primary" 
                        onClick={runAnalysis}
                        disabled={selectedAssets.length < 2}
                        loading={loading}
                        icon={<SaveOutlined />}
                      >
                        运行分析
                      </Button>
                    </Space>
                  </Col>
                </Row>
              </Card>
            </Space>
          </Col>
        </Row>

        {loading ? (
          <div style={{ textAlign: 'center', padding: 100 }}>
            <Spin size="large" tip="正在分析多个资产间的关系，可能需要一些时间..." />
          </div>
        ) : analysisData ? (
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab="相对强弱关系" key="1">
              <Row gutter={16}>
                <Col span={24} md={12}>
                  <AssetCorrelationMatrix 
                    data={analysisData} 
                    onPairSelect={handlePairSelect}
                    selectedPair={selectedPair}
                  />
                </Col>
                <Col span={24} md={12}>
                  <AssetStrengthRanking data={analysisData} />
                </Col>
              </Row>
            </TabPane>
            
            <TabPane tab="资产对详情" key="2">
              {selectedPair.length === 2 && analysisData.pairDetail ? (
                <AssetPairChart 
                  data={analysisData.pairDetail}
                  assetA={selectedPair[0]}
                  assetB={selectedPair[1]}
                  thresholdMultiplier={thresholdMultiplier}
                />
              ) : (
                <Alert 
                  message="请选择一个资产对"
                  description="在相对强弱关系矩阵中点击一个单元格以查看详细的资产对分析。"
                  type="info" 
                  showIcon 
                />
              )}
            </TabPane>
            
            <TabPane tab="最优交易对" key="3">
              <OptimalPairsTable 
                data={analysisData} 
                onPairSelect={handlePairSelect}
              />
            </TabPane>
            
            <TabPane tab="所有资产对比值图表" key="4">
              <AllPairCharts 
                assets={selectedAssets}
                duration={duration}
                polynomialDegree={parseInt(polynomialDegree.toString())}
                thresholdMultiplier={parseFloat(thresholdMultiplier.toString())}
                klineType={klineType}
              />
            </TabPane>
          </Tabs>
        ) : (
          <div style={{ textAlign: 'center', padding: 100, color: '#999' }}>
            请选择至少2个资产并运行分析
          </div>
        )}
      </Card>
    </div>
  );
};

export default MultiAssetAnalysis; 