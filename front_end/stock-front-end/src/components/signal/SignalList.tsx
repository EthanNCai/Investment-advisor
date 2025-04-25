import React from 'react';
import { Table, Tag, Space } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, StarOutlined, StarFilled } from '@ant-design/icons';
import { Signal } from './InvestmentSignal';
import { SignalQualityEvaluation } from './SignalQualityCard';

interface SignalListProps {
  signals: Signal[];
  selectedSignal: Signal | null;
  onSignalSelect: (signal: Signal) => void;
}

const SignalList: React.FC<SignalListProps> = ({ signals, selectedSignal, onSignalSelect }) => {
  
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
  
  const getQualityColor = (quality: SignalQualityEvaluation) => {
    if (!quality) return '';
    const score = quality.quality_score;
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'error';
  };
  
  const getQualityTag = (signal: Signal) => {
    const quality = signal.quality_evaluation as SignalQualityEvaluation;
    if (!quality) return null;
    
    return (
      <Tag color={getQualityColor(quality)}>
        {quality.quality_score.toFixed(0)}分
      </Tag>
    );
  };
  
  const columns = [
    {
      title: '日期',
      dataIndex: 'date',
      key: 'date',
      width: 100,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      width: 90,
      render: (type: string) => (
        <Space>
          {type === 'positive' ? 
            <Tag color="green"><ArrowUpOutlined /> 正向</Tag> : 
            <Tag color="volcano"><ArrowDownOutlined /> 负向</Tag>
          }
        </Space>
      ),
    },
    {
      title: '强度',
      dataIndex: 'strength',
      key: 'strength',
      width: 70,
      render: (strength: string) => (
        <Tag color={getStrengthColor(strength)}>
          {getStrengthText(strength)}
        </Tag>
      ),
    },
    {
      title: 'Z值',
      dataIndex: 'z_score',
      key: 'z_score',
      width: 60,
      render: (z_score: number) => Math.abs(z_score).toFixed(2),
    },
    {
      title: '质量',
      key: 'quality',
      width: 70,
      render: (text: any, record: Signal) => getQualityTag(record),
    },
    {
      title: '信号说明',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
  ];
  
  // 确保signals是数组
  const safeSignals = Array.isArray(signals) ? signals : [];
  
  return (
    <div className="signal-list">
      <Table 
        columns={columns} 
        dataSource={safeSignals} 
        rowKey="id"
        pagination={{ pageSize: 5 }}
        size="middle"
        rowClassName={(record) => 
          selectedSignal && record.id === selectedSignal.id ? 'ant-table-row-selected' : ''
        }
        onRow={(record) => ({
          onClick: () => onSignalSelect(record),
        })}
      />
      {safeSignals.length === 0 && (
        <div style={{ textAlign: 'center', padding: '20px 0' }}>
          当前选择的时间范围内没有检测到投资信号
        </div>
      )}
    </div>
  );
};

export default SignalList; 