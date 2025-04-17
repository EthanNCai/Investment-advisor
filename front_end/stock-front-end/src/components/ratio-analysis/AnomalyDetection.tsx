import React from 'react';
import { Slider, Typography, Card, Divider, Row, Col, Badge, Tag, Table } from 'antd';
import { WarningOutlined, CheckCircleOutlined, ExclamationCircleOutlined } from '@ant-design/icons';

interface AnomalyInfo {
  mean: number;
  std: number;
  anomalies: Array<{
    index: number;
    value: number;
    z_score: number;
    deviation: number;
  }>;
  warning_level: 'normal' | 'medium' | 'high';
  upper_bound: number;
  lower_bound: number;
}

interface AnomalyDetectionProps {
  anomalyInfo: AnomalyInfo;
  dates: string[];
  threshold: number;
  onThresholdChange?: (value: number) => void;
}

const { Text } = Typography;

const AnomalyDetection: React.FC<AnomalyDetectionProps> = ({ 
  anomalyInfo, 
  dates,
  threshold,
  onThresholdChange
}) => {
  const handleThresholdChange = (value: number) => {
    if (onThresholdChange) {
      onThresholdChange(value);
    }
  };

  const upperBound = anomalyInfo.upper_bound;
  const lowerBound = anomalyInfo.lower_bound;

  const getWarningLabel = (level: 'normal' | 'medium' | 'high') => {
    switch (level) {
      case 'high':
        return <Badge status="error" text={<Text strong style={{ color: '#ff4d4f' }}>高风险预警</Text>} />;
      case 'medium':
        return <Badge status="warning" text={<Text strong style={{ color: '#faad14' }}>中等风险预警</Text>} />;
      case 'normal':
        return <Badge status="success" text={<Text style={{ color: '#52c41a' }}>正常</Text>} />;
    }
  };

  const getWarningIcon = (level: 'normal' | 'medium' | 'high') => {
    switch (level) {
      case 'high':
        return <WarningOutlined style={{ color: '#ff4d4f', fontSize: '24px' }} />;
      case 'medium':
        return <ExclamationCircleOutlined style={{ color: '#faad14', fontSize: '24px' }} />;
      case 'normal':
        return <CheckCircleOutlined style={{ color: '#52c41a', fontSize: '24px' }} />;
    }
  };

  const columns = [
    {
      title: '日期',
      dataIndex: 'index',
      key: 'date',
      render: (index: number) => dates[index] || '-',
    },
    {
      title: '比值',
      dataIndex: 'value',
      key: 'value',
      render: (value: number) => value.toFixed(4),
    },
    {
      title: 'Z分数',
      dataIndex: 'z_score',
      key: 'z_score',
      render: (z_score: number) => {
        let color = z_score > 3 ? 'red' : z_score > 2 ? 'orange' : 'green';
        return <Tag color={color}>{z_score.toFixed(2)}</Tag>;
      },
    },
    {
      title: '偏离度',
      dataIndex: 'deviation',
      key: 'deviation',
      render: (deviation: number) => {
        let color = Math.abs(deviation) > 0.1 ? 'red' : Math.abs(deviation) > 0.05 ? 'orange' : 'green';
        return <Tag color={color}>{(deviation * 100).toFixed(2)}%</Tag>;
      },
    },
  ];

  return (
    <Card
      title={
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            {getWarningIcon(anomalyInfo.warning_level)}
            <span>价差异常检测与预警</span>
          </div>
          {getWarningLabel(anomalyInfo.warning_level)}
        </div>
      }
      style={{ marginTop: 16 }}
    >
      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card size="small" title="统计信息">
            <p><Text strong>均值:</Text> {anomalyInfo.mean.toFixed(4)}</p>
            <p><Text strong>标准差:</Text> {anomalyInfo.std.toFixed(4)}</p>
            <p><Text strong>上界值:</Text> {upperBound.toFixed(4)}</p>
            <p><Text strong>下界值:</Text> {lowerBound.toFixed(4)}</p>
          </Card>
        </Col>
        <Col span={12}>
          <Card size="small" title="异常检测阈值设置">
            <div style={{ padding: '0 10px' }}>
              <Slider
                min={1}
                max={5}
                step={0.1}
                value={threshold}
                onChange={handleThresholdChange}
                marks={{
                  1: '1σ',
                  2: '2σ',
                  3: '3σ',
                  4: '4σ',
                  5: '5σ'
                }}
                tooltip={{ formatter: (value) => `${value}σ` }}
              />
              <div style={{ textAlign: 'center', marginTop: 8 }}>
                <Text type="secondary">当前阈值: {threshold}σ (标准差的倍数)</Text>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {anomalyInfo.anomalies.length > 0 ? (
        <div style={{ marginTop: 16 }}>
          <Divider orientation="left">异常值列表 ({anomalyInfo.anomalies.length}个)</Divider>
          <Table 
            dataSource={anomalyInfo.anomalies} 
            columns={columns} 
            rowKey={(record) => `anomaly-${record.index}`}
            pagination={{ pageSize: 5 }}
            size="small"
          />
        </div>
      ) : (
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <Divider orientation="left">异常值列表</Divider>
          <Text type="secondary">未检测到任何异常值</Text>
        </div>
      )}
    </Card>
  );
};

export default AnomalyDetection; 