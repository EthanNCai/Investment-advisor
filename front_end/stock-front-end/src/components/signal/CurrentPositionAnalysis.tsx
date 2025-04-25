import React from 'react';
import { Card, Row, Col, Typography, Statistic, Divider, Tag, Space, Empty, Spin } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined, LineChartOutlined, AreaChartOutlined, NodeIndexOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

interface CurrentPositionProps {
  currentPosition: any;
  loading: boolean;
}

const CurrentPositionAnalysis: React.FC<CurrentPositionProps> = ({ 
  currentPosition, 
  loading 
}) => {
  if (loading) {
    return (
      <Card bordered={false}>
        <div style={{ textAlign: 'center', padding: '40px 0' }}>
          <Spin />
          <div style={{ marginTop: 8 }}>加载数据中...</div>
        </div>
      </Card>
    );
  }
  
  if (!currentPosition) {
    return (
      <Card bordered={false}>
        <Empty description="无法获取当前市场位置数据" />
      </Card>
    );
  }
  
  const getZScoreColor = (zScore: number | null) => {
    if (zScore === null) return '';
    const absZScore = Math.abs(zScore);
    if (absZScore > 2) return zScore > 0 ? '#ff4d4f' : '#52c41a';
    if (absZScore > 1) return zScore > 0 ? '#faad14' : '#1890ff';
    return '#8c8c8c';
  };
  
  const getPercentileTag = (percentile: number | null) => {
    if (percentile === null) return null;
    
    if (percentile > 0.8) return <Tag color="red">历史高位</Tag>;
    if (percentile > 0.7) return <Tag color="orange">相对高位</Tag>;
    if (percentile < 0.2) return <Tag color="green">历史低位</Tag>;
    if (percentile < 0.3) return <Tag color="blue">相对低位</Tag>;
    return <Tag color="default">中等位置</Tag>;
  };
  
  const getVolatilityTag = (level: string | null) => {
    if (!level) return null;
    
    switch (level) {
      case 'high': return <Tag color="red">高波动</Tag>;
      case 'medium': return <Tag color="orange">中等波动</Tag>;
      case 'low': return <Tag color="green">低波动</Tag>;
      default: return null;
    }
  };
  
  const getDeviationText = (deviation: number | null) => {
    if (deviation === null) return null;
    
    const absDeviation = Math.abs(deviation);
    if (absDeviation < 2) return "接近趋势线";
    if (deviation > 0) return `高于趋势线 ${deviation.toFixed(1)}%`;
    return `低于趋势线 ${Math.abs(deviation).toFixed(1)}%`;
  };
  
  const getTrendStrengthTag = (trendStrength: any) => {
    if (!trendStrength) return null;
    
    const colors: {[key: string]: string} = {
      '无明显趋势': 'default',
      '弱趋势': 'blue',
      '中等趋势': 'orange',
      '强趋势': 'red'
    };
    
    const icons: {[key: string]: React.ReactNode} = {
      '上升': <ArrowUpOutlined />,
      '下降': <ArrowDownOutlined />,
      '平稳': null
    };
    
    const iconElement = trendStrength.direction ? icons[trendStrength.direction as string] : null;
    
    return (
      <Tag color={colors[trendStrength.level] || 'default'}>
        {iconElement && <span style={{ marginRight: 4 }}>{iconElement}</span>}
        {trendStrength.level} {trendStrength.value && `(${trendStrength.value})`}
      </Tag>
    );
  };
  
  const getNearestSignals = () => {
    if (!currentPosition.nearest_signals || currentPosition.nearest_signals.length === 0) {
      return <Text type="secondary">未找到相似的历史信号</Text>;
    }
    
    return (
      <div>
        {currentPosition.nearest_signals.map((signal: any, index: number) => (
          <div key={index} style={{ marginBottom: index === currentPosition.nearest_signals.length - 1 ? 0 : 8 }}>
            <Space>
              <Tag color={signal.type === 'positive' ? 'green' : 'volcano'}>
                {signal.type === 'positive' ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                {signal.date}
              </Tag>
              <Text>相似度: {(signal.similarity * 100).toFixed(0)}%</Text>
            </Space>
          </div>
        ))}
      </div>
    );
  };
  
  const getMeanReversionProbability = (probability: number | null) => {
    if (probability === null) return null;
    
    let color = 'blue';
    if (probability > 0.7) color = 'red';
    else if (probability > 0.5) color = 'orange';
    
    return (
      <Statistic
        title="均值回归概率"
        value={probability}
        precision={2}
        valueStyle={{ color }}
        formatter={(value) => `${(value as number * 100).toFixed(0)}%`}
      />
    );
  };
  
  const getCyclePositionInfo = (cyclePosition: any) => {
    if (!cyclePosition) return null;
    
    const positionColors: {[key: string]: string} = {
      '顶部区域': 'red',
      '上升区域': 'orange',
      '中间区域': 'blue',
      '下降区域': 'cyan',
      '底部区域': 'green'
    };
    
    return (
      <div>
        <div style={{ marginBottom: 8 }}>
          <Tag color={positionColors[cyclePosition.position] || 'default'}>
            {cyclePosition.position}
          </Tag>
        </div>
        <Text>{cyclePosition.status}</Text>
      </div>
    );
  };
  
  return (
    <Card 
      title={
        <Space>
          <InfoCircleOutlined />
          <span>当前市场位置分析</span>
        </Space>
      }
      bordered={false}
    >
      <Row gutter={16}>
        <Col span={8}>
          <Statistic
            title="当前价格比值"
            value={currentPosition.current_ratio}
            precision={4}
          />
        </Col>
        <Col span={8}>
          <Statistic
            title="Z值 (偏离程度)"
            value={currentPosition.z_score !== null ? currentPosition.z_score : '-'}
            precision={2}
            valueStyle={{ color: getZScoreColor(currentPosition.z_score) }}
            prefix={currentPosition.z_score > 0 ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
          />
        </Col>
        <Col span={8}>
          <div style={{ marginBottom: 4 }}>
            <Text type="secondary">位置评估</Text>
          </div>
          <Space>
            {getPercentileTag(currentPosition.percentile)}
            {getVolatilityTag(currentPosition.volatility_level)}
            {currentPosition.is_extreme && <Tag color="red">极端位置</Tag>}
          </Space>
        </Col>
      </Row>
      
      <Divider />
      
      <Row gutter={16}>
        <Col span={12}>
          <Title level={5}><LineChartOutlined /> 趋势分析</Title>
          <Row gutter={16}>
            <Col span={12}>
              {getTrendStrengthTag(currentPosition.trend_strength)}
              <div style={{ marginTop: 8 }}>{getDeviationText(currentPosition.deviation_from_trend)}</div>
            </Col>
            <Col span={12}>
              {currentPosition.cycle_position && (
                <>
                  <Text strong>周期位置:</Text>
                  {getCyclePositionInfo(currentPosition.cycle_position)}
                </>
              )}
            </Col>
          </Row>
        </Col>
        <Col span={12}>
          <Title level={5}><NodeIndexOutlined /> 历史模式</Title>
          <div>{currentPosition.historical_signal_pattern || "无明显模式"}</div>
          <div style={{ marginTop: 12 }}>
            {getMeanReversionProbability(currentPosition.mean_reversion_probability)}
          </div>
        </Col>
      </Row>
      
      <Divider />
      
      {currentPosition.support_resistance && (
        <>
          <Row gutter={16}>
            <Col span={24}>
              <Title level={5}><AreaChartOutlined /> 支撑与阻力位</Title>
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic
                    title="强支撑位"
                    value={currentPosition.support_resistance.strong_support}
                    precision={3}
                    valueStyle={{ color: '#52c41a' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="近期支撑位"
                    value={currentPosition.support_resistance.nearby_support}
                    precision={3}
                    valueStyle={{ color: '#1890ff' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="近期阻力位"
                    value={currentPosition.support_resistance.nearby_resistance}
                    precision={3}
                    valueStyle={{ color: '#faad14' }}
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="强阻力位"
                    value={currentPosition.support_resistance.strong_resistance}
                    precision={3}
                    valueStyle={{ color: '#ff4d4f' }}
                  />
                </Col>
              </Row>
            </Col>
          </Row>
          
          <Divider />
        </>
      )}
      
      <Row gutter={16}>
        <Col span={24}>
          <Title level={5}>相似历史信号</Title>
          {getNearestSignals()}
        </Col>
      </Row>
      
      <Divider />
      
      <Title level={5}>综合建议</Title>
      <Paragraph>
        <Text strong>{currentPosition.recommendation}</Text>
      </Paragraph>
    </Card>
  );
};

export default CurrentPositionAnalysis; 