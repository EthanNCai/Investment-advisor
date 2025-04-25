import React from 'react';
import { Card, Row, Col, Select, Slider, Checkbox, InputNumber, Typography, Space, Tooltip, Switch } from 'antd';
import { FilterOutlined, SettingOutlined, QuestionCircleOutlined } from '@ant-design/icons';

const { Text } = Typography;
const { Option } = Select;

interface SignalFilterBarProps {
  signalStrength: string[];
  onSignalStrengthChange: (values: string[]) => void;
  signalType: string[];
  onSignalTypeChange: (values: string[]) => void;
  duration: string;
  onDurationChange: (value: string) => void;
  degree: number;
  onDegreeChange: (value: number) => void;
  threshold: number;
  onThresholdChange: (value: number) => void;
  trackSignals: boolean;
  onTrackSignalsChange: (value: boolean) => void;
}

const SignalFilterBar: React.FC<SignalFilterBarProps> = ({
  signalStrength,
  onSignalStrengthChange,
  signalType,
  onSignalTypeChange,
  duration,
  onDurationChange,
  degree,
  onDegreeChange,
  threshold,
  onThresholdChange,
  trackSignals,
  onTrackSignalsChange
}) => {
  // 时间跨度选项
  const durationOptions = [
    { label: "1个月", value: "1m" },
    { label: "3个月", value: "1q" },
    { label: "1年", value: "1y" },
    { label: "2年", value: "2y" },
    { label: "5年", value: "5y" },
    { label: "全部", value: "maximum" }
  ];
  
  // 信号强度选项
  const strengthOptions = [
    { label: "弱", value: "weak" },
    { label: "中", value: "medium" },
    { label: "强", value: "strong" }
  ];
  
  // 信号类型选项
  const typeOptions = [
    { label: "正向", value: "positive" },
    { label: "负向", value: "negative" }
  ];
  
  return (
    <Card 
      type="inner" 
      title={
        <Space>
          <SettingOutlined />
          <span>参数设置与过滤</span>
        </Space>
      }
      style={{ marginBottom: 16 }}
    >
      <Row gutter={[16, 16]}>
        {/* 参数设置部分 */}
        <Col span={24}>
          <Text strong>
            <SettingOutlined /> 分析参数:
          </Text>
        </Col>
        
        <Col span={8}>
          <Text>时间跨度:</Text>
          <Select
            value={duration}
            onChange={onDurationChange}
            style={{ width: '100%', marginTop: 8 }}
          >
            {durationOptions.map(option => (
              <Option key={option.value} value={option.value}>
                {option.label}
              </Option>
            ))}
          </Select>
        </Col>
        
        <Col span={8}>
          <Text>
            <Tooltip title="拟合曲线的多项式次数，值越高拟合曲线越复杂">
              <span>拟合阶数: <QuestionCircleOutlined /></span>
            </Tooltip>
          </Text>
          <div style={{ display: 'flex', marginTop: 8 }}>
            <Slider
              min={1}
              max={5}
              value={degree}
              onChange={onDegreeChange}
              style={{ flex: 1, marginRight: 16 }}
            />
            <InputNumber
              min={1}
              max={5}
              value={degree}
              onChange={value => onDegreeChange(value || 3)}
              style={{ width: 60 }}
            />
          </div>
        </Col>
        
        <Col span={8}>
          <Text>
            <Tooltip title="检测异常值的阈值系数，值越高检测标准越严格">
              <span>阈值系数: <QuestionCircleOutlined /></span>
            </Tooltip>
          </Text>
          <div style={{ display: 'flex', marginTop: 8 }}>
            <Slider
              min={1}
              max={3}
              step={0.1}
              value={threshold}
              onChange={onThresholdChange}
              style={{ flex: 1, marginRight: 16 }}
            />
            <InputNumber
              min={1}
              max={3}
              step={0.1}
              value={threshold}
              onChange={value => onThresholdChange(value || 2.0)}
              style={{ width: 60 }}
            />
          </div>
        </Col>
        
        <Col span={24}>
          <Tooltip title="开启后将记录信号并跟踪其后续表现">
            <span>
              <Switch 
                checked={trackSignals}
                onChange={onTrackSignalsChange}
                style={{ marginRight: 8 }}
              />
              <Text>追踪信号表现</Text>
              <QuestionCircleOutlined style={{ marginLeft: 4 }} />
            </span>
          </Tooltip>
        </Col>
        
        {/* 过滤选项部分 */}
        <Col span={24} style={{ marginTop: 8 }}>
          <Text strong>
            <FilterOutlined /> 过滤显示:
          </Text>
        </Col>
        
        <Col span={12}>
          <Text>信号强度:</Text>
          <div style={{ marginTop: 8 }}>
            <Checkbox.Group
              options={strengthOptions}
              value={signalStrength}
              onChange={values => onSignalStrengthChange(values as string[])}
            />
          </div>
        </Col>
        
        <Col span={12}>
          <Text>信号类型:</Text>
          <div style={{ marginTop: 8 }}>
            <Checkbox.Group
              options={typeOptions}
              value={signalType}
              onChange={values => onSignalTypeChange(values as string[])}
            />
          </div>
        </Col>
      </Row>
    </Card>
  );
};

export default SignalFilterBar; 