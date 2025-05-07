import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Row, Col, Card, Typography, Spin, Select, Button, Table, Statistic, Divider, Empty, Space, Badge, Pagination, Radio } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, ReloadOutlined, LineChartOutlined, DownOutlined, BarsOutlined, AppstoreOutlined } from '@ant-design/icons';
import ReactECharts from 'echarts-for-react';
import { useLocalStorage } from '../../LocalStorageContext';
import { useDashboardData, DashboardData } from '../../hooks/useDashboardData';
import { useClientPagination } from '../../hooks/usePaginatedData';
import { useLoadMore } from '../../hooks/useLoadMore';
import LazyLoadComponent from '../LazyLoad/LazyLoadComponent';
import './DashboardStyles.css'; // 导入样式文件

const { Title, Text } = Typography;
const { Option } = Select;

interface TrendData {
  time: string;
  current_price: number;
  change_percent: number;
}

interface AssetPair {
  code_a: string;
  name_a: string;
  code_b: string;
  name_b: string;
  current_ratio: number;
  change_ratio: number;
  trends_a: TrendData[];
  trends_b: TrendData[];
  signals: any[];
  latest_date: string;
  recommendation?: string;
}

interface Asset {
  code: string;
  name: string;
  current_price: number;
  price_change: number;
  trends: TrendData[];
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [isLoggedIn] = useLocalStorage<boolean>('isLoggedIn', false);
  const [currentUser] = useLocalStorage<any>('currentUser', null);
  
  // 添加加载模式偏好状态
  const [loadingMode, setLoadingMode] = useLocalStorage<'pagination' | 'loadMore'>('loadingMode', 'pagination');
  
  const [selectedPair, setSelectedPair] = useState<string>('');
  const [currentPairData, setCurrentPairData] = useState<AssetPair | null>(null);
  const [ratioTrend, setRatioTrend] = useState<{time: string, ratio: number}[]>([]);

  // 使用自定义Hook获取仪表盘数据
  const { data, isLoading, refetch } = useDashboardData(
    isLoggedIn && !!currentUser,
    {
      onSuccess: (data: DashboardData) => {
        console.log('仪表盘数据获取成功:', data);
      }
    }
  );

  // 提取数据便于使用
  const recentPairs = data?.recent_pairs || [];
  const favoriteAssets = data?.favorite_assets || [];

  // 为资产对和信号添加分页设置
  const ASSETS_PAGE_SIZE = 5;
  const SIGNALS_PAGE_SIZE = 3;
  
  // 使用客户端分页Hook
  const paginatedAssets = useClientPagination(recentPairs, ASSETS_PAGE_SIZE);
  
  // 使用加载更多Hook
  const loadMoreAssets = useLoadMore(recentPairs, ASSETS_PAGE_SIZE, ASSETS_PAGE_SIZE);

  // 检查登录状态
  useEffect(() => {
    if (!isLoggedIn || !currentUser) {
      navigate('/login');
    }
  }, [isLoggedIn, currentUser, navigate]);

  // 初始化选择的资产对
  useEffect(() => {
    if (recentPairs.length > 0 && !selectedPair) {
      const defaultPair = recentPairs[0];
      setSelectedPair(`${defaultPair.code_a}/${defaultPair.code_b}`);
      setCurrentPairData(defaultPair);
      
      // 计算比值趋势
      const trends = calculateRatioTrend(defaultPair);
      setRatioTrend(trends);
    }
  }, [recentPairs, selectedPair]);

  // 计算两只股票的实时比值趋势
  const calculateRatioTrend = (pair: AssetPair) => {
    if (!pair || !pair.trends_a || !pair.trends_b) return [];
    
    const { trends_a, trends_b } = pair;
    const trends: {time: string, ratio: number}[] = [];
    
    // 对齐时间
    const timeMap = new Map<string, number>();
    trends_b.forEach(item => {
      timeMap.set(item.time, item.current_price);
    });
    
    // 计算比值
    trends_a.forEach(itemA => {
      if (timeMap.has(itemA.time)) {
        const priceB = timeMap.get(itemA.time);
        if (priceB && priceB !== 0) {
          trends.push({
            time: itemA.time,
            ratio: itemA.current_price / priceB
          });
        }
      }
    });
    
    return trends;
  };

  // 处理资产对选择变更
  const handlePairChange = (value: string) => {
    setSelectedPair(value);
    
    // 查找对应的资产对数据
    const [codeA, codeB] = value.split('/');
    const pair = recentPairs.find((p: AssetPair) => p.code_a === codeA && p.code_b === codeB);
    
    if (pair) {
      setCurrentPairData(pair);
      
      // 更新比值趋势
      const trends = calculateRatioTrend(pair);
      setRatioTrend(trends);
    }
  };

  // 手动刷新数据
  const handleRefresh = () => {
    refetch();
  };

  // 实时比值趋势图表配置
  const getRatioChartOption = () => {
    if (!currentPairData || !ratioTrend.length) {
      return {};
    }
    
    const times = ratioTrend.map(item => item.time);
    const values = ratioTrend.map(item => item.ratio);
    
    // 获取最新比值
    const lastValue = values[values.length - 1];
    
    return {
      title: {
        text: '实时价格比值趋势',
        left: 'center',
      },
      tooltip: {
        trigger: 'axis',
        formatter: (params: any) => {
          const param = params[0];
          return `${param.axisValue}<br>${param.seriesName}: ${param.value.toFixed(4)}`;
        }
      },
      xAxis: {
        type: 'category',
        data: times,
        axisLabel: {
          rotate: 45
        }
      },
      yAxis: {
        type: 'value',
        scale: true,
        axisLabel: {
          formatter: (value: number) => value.toFixed(4)
        }
      },
      series: [
        {
          name: '价格比值',
          type: 'line',
          data: values,
          showSymbol: false,
          lineStyle: {
            width: 2,
            color: '#1890ff'
          },
          markPoint: {
            data: [
              {
                name: '当前值',
                type: 'max',
                symbolSize: 8,
                itemStyle: {
                  color: '#52c41a'
                },
                label:{
                  show: true,
                  formatter: (params: any) => {
                  return params.value.toFixed(4);
                },
                  position:[10,0],
                  textStyle:{
                    color:'#52c41a'
                  }
                }
              }
            ]
          },
          markLine: {
            silent: true,
            data: [
              {
                yAxis: lastValue,
                lineStyle: {
                  color: '#ff4d4f',
                  type: 'dashed'
                },
                label: {
                  show: true,
                  position: 'end',
                  formatter: (params: any) => {
                    return  params.value.toFixed(2);
                  }
                }
              }
            ]
          }
        }
      ],
      grid: {
        left: '8%',
        right: '5%',
        bottom: '15%',
        top: '15%'
      }
    };
  };

  // 渲染最近查看的资产对表格
  const renderRecentPairsTable = () => {
    const tableData = loadingMode === 'pagination' 
      ? paginatedAssets.currentPageData 
      : loadMoreAssets.visibleData;
      
    return (
      <LazyLoadComponent height={300} placeholder={<div style={{ padding: '40px 0', textAlign: 'center' }}><Spin tip="加载资产数据..." /></div>}>
        <>
          <Table 
            dataSource={tableData}
            columns={recentPairsColumns}
            rowKey={(record) => `${record.code_a}_${record.code_b}`}
            pagination={false}
            size="middle"
            style={{ width: '100%' }}
            className="asset-pairs-table"
          />
        </>
      </LazyLoadComponent>
    );
  };
  
  // 计算投资信号
  const calculateBestSignals = () => {
    if (!recentPairs || recentPairs.length === 0 || !recentPairs.some((pair: AssetPair) => pair.signals && pair.signals.length > 0)) {
      return [];
    }

    // 为每对资产对选择最高相似度的信号
    return recentPairs
      .filter((pair: AssetPair) => pair.signals && pair.signals.length > 0)
      .map((pair: AssetPair) => {
        // 对信号按相似度排序，取最高相似度的信号
        const sortedSignals = [...pair.signals].filter((signal: any) => signal.id).sort((a: any, b: any) => 
          (b.similarity || 0) - (a.similarity || 0)
        );
        
        // 获取当前资产对的recommendation
        // 它在signals数组的最后一个元素中
        let pairRecommendation = '';
        if (pair.signals && pair.signals.length > 0) {
          const lastElement = pair.signals[pair.signals.length - 1];
          if (lastElement && typeof lastElement === 'object' && 'recommendation' in lastElement) {
            pairRecommendation = lastElement.recommendation || '';
          }
        }
        
        // 返回带有资产对信息的最高相似度信号
        if (sortedSignals.length === 0) return null;
        
        return {
          ...sortedSignals[0],
          assetPairName: `${pair.name_a}/${pair.name_b}`,
          code_a: pair.code_a,
          code_b: pair.code_b,
          // 使用从signals最后一个元素中获取的recommendation
          recommendation: pairRecommendation
        };
      })
      .filter((signal: any) => signal !== null); // 过滤掉可能的null值
  };
  
  // 用客户端分页Hook管理投资信号
  const allSignals = calculateBestSignals();
  const paginatedSignals = useClientPagination(allSignals, SIGNALS_PAGE_SIZE);
  const loadMoreSignals = useLoadMore(allSignals, SIGNALS_PAGE_SIZE, SIGNALS_PAGE_SIZE);

  // 渲染投资信号
  const renderSignals = () => {
    if (allSignals.length === 0) {
      return (
        <Empty description="暂无投资信号" image={Empty.PRESENTED_IMAGE_SIMPLE} />
      );
    }
    
    const signalData = loadingMode === 'pagination' 
      ? paginatedSignals.currentPageData 
      : loadMoreSignals.visibleData;

    return (
      <LazyLoadComponent height={400} placeholder={<div style={{ padding: '80px 0', textAlign: 'center' }}><Spin tip="加载投资信号..." /></div>}>
        <>
          <Space direction="vertical" size="middle" style={{ width: '100%' }}>
            {signalData.map((signal: any, index: number) => (
              <Card key={index} size="small" style={{ marginBottom: 8 }}>
                <Row gutter={[8, 16]}>
                  <Col span={24}>
                    <Row align="middle">
                      <Col span={16}>
                        <Text strong>{signal.assetPairName}</Text>
                        <Badge 
                          status={signal.type === 'positive' ? 'success' : 'error'} 
                          text={signal.type === 'positive' ? '超买' : '超卖'} 
                          style={{ marginLeft: 8 }}
                        />
                        <Text strong style={{ marginLeft: 16 }}>
                          相似度: {(signal.similarity * 100).toFixed(1)}%
                        </Text>
                      </Col>
                      <Col span={8} style={{ textAlign: 'right' }}>
                        <Text type="secondary">信号日期: {signal.date}</Text>
                      </Col>
                    </Row>
                  </Col>
                  
                  <Col span={8}>
                    <Text type="secondary">价格比值:</Text>
                    <br />
                    <Text strong>{signal.ratio !== undefined ? signal.ratio.toFixed(4) : '0.0000'}</Text>
                  </Col>
                  
                  <Col span={8}>
                    <Text type="secondary">{signal.type === 'positive' ? '上行' : '下行'}信号</Text>
                    <br />
                    <Text 
                      strong 
                      type={signal.type === 'positive' ? 'success' : 'danger'}
                    >
                      强度: {signal.strength || '中等'}
                    </Text>
                  </Col>
                  
                  <Col span={8} style={{ textAlign: 'right' }}>
                    <Button 
                      type="link" 
                      icon={<LineChartOutlined />}
                      onClick={() => navigate(`/ratio-analysis?codeA=${signal.code_a}&codeB=${signal.code_b}`)}
                    >
                      详细分析
                    </Button>
                  </Col>
                  
                  <Col span={24}>
                    <Divider style={{ margin: '8px 0' }} />
                    <Text type="secondary">建议:</Text>
                    <br />
                    <Text>
                      {signal.recommendation ? signal.recommendation : '当前信号暂无明显异常，建议继续观察市场动态'}
                    </Text>
                  </Col>
                </Row>
              </Card>
            ))}
          </Space>
        </>
      </LazyLoadComponent>
    );
  };

  // 修改调试信息，查看所有资产对的数据结构
  useEffect(() => {
    if (recentPairs && recentPairs.length > 0) {
      // 遍历所有资产对，打印调试信息
      recentPairs.forEach((pair, index) => {
        console.log(`第${index+1}对资产对数据:`, pair);
        
        // 检查signals数组
        if (pair.signals && pair.signals.length > 0) {
          // 打印最后一个元素，查看recommendation
          const lastElement = pair.signals[pair.signals.length - 1];
          console.log(`第${index+1}对资产对的signals最后一个元素:`, lastElement);
          
          // 检查是否包含recommendation
          if (lastElement && typeof lastElement === 'object') {
            console.log(`第${index+1}对资产对的recommendation:`, 'recommendation' in lastElement ? lastElement.recommendation : '不存在');
          }
        }
      });
    }
  }, [recentPairs]);

  // 页面加载状态
  if (isLoading) {
    return (
      <div style={{ textAlign: 'center', padding: '100px 0' }}>
        <Spin size="large" />
        <div style={{ marginTop: 20 }}>正在加载仪表盘数据...</div>
      </div>
    );
  }

  // 渲染最近查看的资产对表格
  const recentPairsColumns = [
    {
      title: '资产对',
      dataIndex: 'name',
      key: 'name',
      width: '40%',
      render: (_: any, record: AssetPair) => (
        <Text strong>{record.name_a}/{record.name_b}</Text>
      )
    },
    {
      title: '当前比值',
      dataIndex: 'current_ratio',
      key: 'current_ratio',
      width: '20%',
      render: (ratio: number) => ratio.toFixed(4)
    },
    {
      title: '变化率',
      dataIndex: 'change_ratio',
      key: 'change_ratio',
      width: '20%',
      render: (value: number) => (
        <Text type={value > 0 ? 'success' : value < 0 ? 'danger' : undefined}>
          {value > 0 ? <ArrowUpOutlined /> : value < 0 ? <ArrowDownOutlined /> : null}
          {value.toFixed(2)}%
        </Text>
      )
    },
    {
      title: '操作',
      key: 'action',
      width: '20%',
      render: (_: any, record: AssetPair) => (
        <Button 
          type="link" 
          icon={<LineChartOutlined />}
          onClick={() => navigate(`/ratio-analysis?codeA=${record.code_a}&codeB=${record.code_b}`)}
        >
          分析
        </Button>
      )
    }
  ];

  // 渲染收藏资产表格
  const favoriteAssetsColumns = [
    {
      title: '资产名称',
      dataIndex: 'name',
      key: 'name',
      render: (_: any, record: Asset) => (
        <Text strong>{record.name} ({record.code})</Text>
      )
    },
    {
      title: '当前价格',
      dataIndex: 'current_price',
      key: 'current_price',
      render: (price: number) => price.toFixed(2)
    },
    {
      title: '变化率',
      dataIndex: 'price_change',
      key: 'price_change',
      render: (value: number) => (
        <Text type={value > 0 ? 'success' : value < 0 ? 'danger' : undefined}>
          {value > 0 ? <ArrowUpOutlined /> : value < 0 ? <ArrowDownOutlined /> : null}
          {value.toFixed(2)}%
        </Text>
      )
    }
  ];

  // 自定义Card标题，包含模式选择
  const renderAssetPairsTitle = () => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
      <span>最近查看的资产对</span>
      {recentPairs.length > ASSETS_PAGE_SIZE && (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Radio.Group
            value={loadingMode}
            onChange={(e) => setLoadingMode(e.target.value)}
            size="small"
            buttonStyle="solid"
            style={{ marginRight: 16 }}
          >
            <Radio.Button value="pagination"><BarsOutlined /> 分页</Radio.Button>
            <Radio.Button value="loadMore"><AppstoreOutlined /> 加载更多</Radio.Button>
          </Radio.Group>
          {loadingMode === 'pagination' && (
            <Pagination
              current={paginatedAssets.currentPage}
              pageSize={paginatedAssets.pageSize}
              total={paginatedAssets.total}
              onChange={paginatedAssets.goToPage}
              size="small"
              showSizeChanger={false}
              simple
            />
          )}
          {loadingMode === 'loadMore' && (
            <Button 
              type="primary" 
              onClick={loadMoreAssets.loadMore} 
              icon={<DownOutlined />} 
              size="small"
              disabled={!loadMoreAssets.hasMore}
            >
              加载更多
            </Button>
          )}
        </div>
      )}
    </div>
  );
  
  // 自定义投资信号标题
  const renderSignalsTitle = () => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', width: '100%' }}>
      <span>最新投资信号</span>
      {allSignals.length > SIGNALS_PAGE_SIZE && (
        <div style={{ display: 'flex', alignItems: 'center' }}>
          <Radio.Group
            value={loadingMode}
            onChange={(e) => setLoadingMode(e.target.value)}
            size="small"
            buttonStyle="solid"
            style={{ marginRight: 16 }}
          >
            <Radio.Button value="pagination"><BarsOutlined /> 分页</Radio.Button>
            <Radio.Button value="loadMore"><AppstoreOutlined /> 加载更多</Radio.Button>
          </Radio.Group>
          {loadingMode === 'pagination' && (
            <Pagination
              current={paginatedSignals.currentPage}
              pageSize={paginatedSignals.pageSize}
              total={paginatedSignals.total}
              onChange={paginatedSignals.goToPage}
              size="small"
              showSizeChanger={false}
              simple
            />
          )}
          {loadingMode === 'loadMore' && (
            <Button 
              type="primary" 
              onClick={loadMoreSignals.loadMore} 
              icon={<DownOutlined />} 
              size="small"
              disabled={!loadMoreSignals.hasMore}
            >
              加载更多
            </Button>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="dashboard">
      <Row gutter={[16, 16]} justify="space-between" align="middle">
        <Col>
          <Title level={2}>资产价值比较分析系统</Title>
        </Col>
        <Col>
          <Space>
            <Button 
              type="primary" 
              icon={<ReloadOutlined />} 
              onClick={handleRefresh}
            >
              刷新数据
            </Button>
          </Space>
        </Col>
      </Row>

      <Divider />

      {/* 实时监控交互区域 */}
      <Card title="实时监控" className="dashboard-card">
        <Row gutter={[16, 16]}>
          <Col xs={24} md={6}>
            <Card size="small" title="选择资产对">
              <Select
                style={{ width: '100%' }}
                value={selectedPair}
                onChange={handlePairChange}
                placeholder="选择资产对"
              >
                {recentPairs.map(pair => (
                  <Option key={`${pair.code_a}/${pair.code_b}`} value={`${pair.code_a}/${pair.code_b}`}>
                    {pair.name_a}/{pair.name_b}
                  </Option>
                ))}
              </Select>
              
              {currentPairData && (
                <div style={{ marginTop: 16 }}>
                  <Statistic 
                    title="当前比值" 
                    value={currentPairData.current_ratio} 
                    precision={4}
                  />
                  <Statistic 
                    title="变化率" 
                    value={currentPairData.change_ratio} 
                    precision={2}
                    valueStyle={{ color: currentPairData.change_ratio >= 0 ? '#3f8600' : '#cf1322' }}
                    prefix={currentPairData.change_ratio >= 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                    suffix="%"
                  />
                </div>
              )}
            </Card>
          </Col>
          
          <Col xs={24} md={18}>
            {ratioTrend.length > 0 ? (
              <ReactECharts 
                option={getRatioChartOption()} 
                style={{ height: 300 }}
              />
            ) : (
              <Empty description="暂无趋势数据" />
            )}
          </Col>
        </Row>
      </Card>

      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} lg={16}>
          {/* 最近查看的资产对区域 */}
          <Card 
            title={renderAssetPairsTitle()} 
            className="dashboard-card asset-pairs-card" 
            bodyStyle={{ padding: '0 0 8px 0' }}
          >
            {renderRecentPairsTable()}
          </Card>
        </Col>

        {/* 收藏的资产区域 - 高度与最近查看的资产对对齐 */}
        <Col xs={24} lg={8}>
          <Card 
            title="收藏的资产" 
            className="dashboard-card favorite-assets-card"
            bodyStyle={{ padding: 0 }}
          >
            <div style={{ maxHeight: '308px', overflow: 'auto' }}>
              <Table 
                dataSource={favoriteAssets}
                columns={favoriteAssetsColumns}
                rowKey="code"
                pagination={false}
                size="small"
                scroll={{ y: 260 }}
                style={{ width: '100%' }}
              />
            </div>
          </Card>
        </Col>
      </Row>
      
      {/* 最新投资信号区域 - 独占一行 */}
      <Row style={{ marginTop: 16 }}>
        <Col span={24}>
          <Card 
            title={renderSignalsTitle()} 
            className="dashboard-card signals-card"
            bodyStyle={{ padding: '0 0 8px 0' }}
          >
            {renderSignals()}
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard; 