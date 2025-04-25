import React, { useState } from 'react';
import { Card, Descriptions, Tag, Space, Typography, Divider, Tabs } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, InfoCircleOutlined } from '@ant-design/icons';
import { Signal } from './InvestmentSignal';
import SignalQualityCard from './SignalQualityCard';
import SignalTrackingCard from './SignalTrackingCard';

const { Text, Paragraph } = Typography;
const { TabPane } = Tabs;

interface SignalDetailProps {
  signal: Signal;
}

const SignalDetail: React.FC<SignalDetailProps> = ({ signal }) => {
  const [activeTab, setActiveTab] = useState<string>('1');
  
  if (!signal) {
    return null;
  }

  const getStrengthColor = (strength: string) => {
    switch (strength) {
      case 'weak': return 'blue';
      case 'medium': return 'orange';
      case 'strong': return 'red';
      default: return 'default';
    }
  };
  
  const getStrengthText = (strength: string) => {
    switch (strength) {
      case 'weak': return '弱';
      case 'medium': return '中';
      case 'strong': return '强';
      default: return '未知';
    }
  };
  
  return (
    <Card 
      title={
        <Space>
          <InfoCircleOutlined />
          <span>信号详情</span>
          <Tag color={signal.type === 'positive' ? 'green' : 'volcano'}>
            {signal.type === 'positive' ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
            {signal.type === 'positive' ? '正向信号' : '负向信号'}
          </Tag>
          <Tag color={getStrengthColor(signal.strength)}>
            {getStrengthText(signal.strength)}强度
          </Tag>
        </Space>
      }
      bordered={false}
      className="signal-detail-card"
    >
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="基本信息" key="1">
          <Descriptions bordered size="small" column={2}>
            <Descriptions.Item label="日期">{signal.date}</Descriptions.Item>
            <Descriptions.Item label="价格比值">{signal.ratio.toFixed(4)}</Descriptions.Item>
            <Descriptions.Item label="Z值">{signal.z_score.toFixed(2)}</Descriptions.Item>
            <Descriptions.Item label="信号强度">{getStrengthText(signal.strength)}</Descriptions.Item>
          </Descriptions>
          
          <Divider orientation="left">信号分析</Divider>
          <Paragraph>{signal.description}</Paragraph>
          
          <Divider orientation="left">投资建议</Divider>
          <Paragraph>
            <Text strong>{signal.recommendation}</Text>
          </Paragraph>
        </TabPane>
        
        <TabPane tab="质量评分" key="2">
          <SignalQualityCard signal={signal} />
        </TabPane>
        
        <TabPane tab="信号追踪" key="3">
          <SignalTrackingCard signal={signal} />
        </TabPane>
      </Tabs>
    </Card>
  );
};

export default SignalDetail; 