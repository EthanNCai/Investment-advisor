import React from 'react';
import { Card, Select, Radio, Slider, InputNumber, Row, Col, Space, Typography } from 'antd';
import { FilterOutlined } from '@ant-design/icons';

const { Option } = Select;
const { Text } = Typography;

export interface SignalFilterBarProps {
  signalStrength: string[];
  setSignalStrength: (value: string[]) => void;
  signalType: string[];
  setSignalType: (value: string[]) => void;
  selectedDuration: string;
  setSelectedDuration: (value: string) => void;
  degree: number;
  setDegree: (value: number) => void;
  thresholdArg: number;
  setThresholdArg: (value: number) => void;
}

const SignalFilterBar: React.FC<SignalFilterBarProps> = ({
  signalStrength,
  setSignalStrength,
  signalType,
  setSignalType,
  selectedDuration,
  setSelectedDuration,
  degree,
  setDegree,
  thresholdArg,
  setThresholdArg
}) => {
  const handleDurationChange = (value: string) => {
    setSelectedDuration(value);
  };

  const handleStrengthChange = (value: string[]) => {
    setSignalStrength(value);
  };

  const handleTypeChange = (value: string[]) => {
    setSignalType(value);
  };

  const handleDegreeChange = (value: number | null) => {
    if (value !== null) {
      setDegree(value);
    }
  };

  const handleThresholdChange = (value: number | null) => {
    if (value !== null) {
      setThresholdArg(value);
    }
  };

  return (
    <Card className="signal-filter-bar" size="small" title={<><FilterOutlined /> 信号筛选</>}>
      <Space direction="vertical" style={{ width: '100%' }}>
        <Row gutter={16}>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>时间范围:</Text>
              <Select 
                style={{ width: '100%' }} 
                value={selectedDuration}
                onChange={handleDurationChange}
              >
                <Option value="1m">1个月</Option>
                <Option value="3m">3个月</Option>
                <Option value="6m">6个月</Option>
                <Option value="1y">1年</Option>
                <Option value="2y">2年</Option>
                <Option value="3y">3年</Option>
                <Option value="5y">5年</Option>
                <Option value="maximum">全部</Option>
              </Select>
            </Space>
          </Col>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>信号强度:</Text>
              <Select
                mode="multiple"
                style={{ width: '100%' }}
                placeholder="选择信号强度"
                value={signalStrength}
                onChange={handleStrengthChange}
              >
                <Option value="weak">弱信号</Option>
                <Option value="medium">中等信号</Option>
                <Option value="strong">强信号</Option>
              </Select>
            </Space>
          </Col>
          <Col span={8}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>信号类型:</Text>
              <Select
                mode="multiple"
                style={{ width: '100%' }}
                placeholder="选择信号类型"
                value={signalType}
                onChange={handleTypeChange}
              >
                <Option value="positive">正向信号</Option>
                <Option value="negative">负向信号</Option>
              </Select>
            </Space>
          </Col>
        </Row>
        
        <Row gutter={16} style={{ marginTop: 16 }}>
          <Col span={12}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>拟合多项式阶数: {degree}</Text>
              <Row>
                <Col span={18}>
                  <Slider
                    min={1}
                    max={5}
                    step={1}
                    value={degree}
                    onChange={handleDegreeChange}
                  />
                </Col>
                <Col span={6} style={{ paddingLeft: 12 }}>
                  <InputNumber
                    min={1}
                    max={5}
                    value={degree}
                    onChange={handleDegreeChange}
                    style={{ width: '100%' }}
                  />
                </Col>
              </Row>
              <Text type="secondary">阶数越高，拟合曲线越复杂，拟合精度越高</Text>
            </Space>
          </Col>
          <Col span={12}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text strong>阈值系数: {thresholdArg}</Text>
              <Row>
                <Col span={18}>
                  <Slider
                    min={1}
                    max={3}
                    step={0.1}
                    value={thresholdArg}
                    onChange={handleThresholdChange}
                  />
                </Col>
                <Col span={6} style={{ paddingLeft: 12 }}>
                  <InputNumber
                    min={1}
                    max={3}
                    step={0.1}
                    value={thresholdArg}
                    onChange={handleThresholdChange}
                    style={{ width: '100%' }}
                  />
                </Col>
              </Row>
              <Text type="secondary">系数越高，筛选出的信号越少且更可靠</Text>
            </Space>
          </Col>
        </Row>
      </Space>
    </Card>
  );
};

export default SignalFilterBar; 