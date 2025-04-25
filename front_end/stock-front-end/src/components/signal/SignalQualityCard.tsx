import React from 'react';
import { Card, Progress, Descriptions, Typography, Row, Col, Tag, Space, Tooltip, Divider } from 'antd';
import { InfoCircleOutlined, CheckCircleOutlined, CloseCircleOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { Signal } from './InvestmentSignal';

const { Text, Title } = Typography;

// 定义信号质量评估接口
export interface SignalQualityEvaluation {
  quality_score: number;
  confidence_level: 'low' | 'medium' | 'high';
  historical_accuracy: number | null;
  expected_return: number;
  risk_ratio: number;
  risk_return_ratio: number;
  factors: {
    strength: number;
    market_condition: number;
    consistency: number;
    technical_support: number;
  };
}

interface SignalQualityCardProps {
  signal: Signal;
}

const SignalQualityCard: React.FC<SignalQualityCardProps> = ({ signal }) => {
  // 安全检查，确保质量评估数据存在
  const evaluation = signal?.quality_evaluation as SignalQualityEvaluation;
  
  if (!evaluation) {
    return (
      <Card title="信号质量评分" bordered={false}>
        <Text type="secondary">无可用的质量评分数据</Text>
      </Card>
    );
  }
  
  // 获取评分等级颜色
  const getScoreColor = (score: number) => {
    if (score >= 80) return '#52c41a'; // 绿色
    if (score >= 60) return '#faad14'; // 黄色
    return '#ff4d4f'; // 红色
  };
  
  // 获取可信度标签颜色
  const getConfidenceColor = (level: string) => {
    switch (level) {
      case 'high': return 'success';
      case 'medium': return 'warning';
      case 'low': return 'error';
      default: return 'default';
    }
  };
  
  // 获取可信度中文显示
  const getConfidenceText = (level: string) => {
    switch (level) {
      case 'high': return '高';
      case 'medium': return '中';
      case 'low': return '低';
      default: return '未知';
    }
  };
  
  // 计算风险收益比评级
  const getRiskReturnRating = (ratio: number) => {
    if (ratio >= 2) return { text: '优秀', color: 'success' };
    if (ratio >= 1) return { text: '良好', color: 'processing' };
    if (ratio >= 0.5) return { text: '一般', color: 'warning' };
    return { text: '较差', color: 'error' };
  };
  
  const riskReturnRating = getRiskReturnRating(evaluation.risk_return_ratio);
  
  return (
    <Card 
      title={
        <Space>
          <InfoCircleOutlined />
          <span>信号质量评分</span>
        </Space>
      }
      bordered={false}
      className="signal-quality-card"
    >
      <Row gutter={[16, 16]}>
        <Col span={24} md={8}>
          <div style={{ textAlign: 'center' }}>
            <Progress
              type="dashboard"
              percent={evaluation.quality_score}
              strokeColor={getScoreColor(evaluation.quality_score)}
              format={(percent) => (
                <span style={{ fontSize: '18px', fontWeight: 'bold' }}>
                  {percent?.toFixed(0)}
                </span>
              )}
            />
            <div style={{ marginTop: 8 }}>
              <Tag color={getConfidenceColor(evaluation.confidence_level)}>
                可信度: {getConfidenceText(evaluation.confidence_level)}
              </Tag>
            </div>
          </div>
        </Col>
        
        <Col span={24} md={16}>
          <Descriptions column={1} size="small">
            <Descriptions.Item label={
              <Tooltip title="基于历史相似信号的盈利情况计算">
                历史准确率 <QuestionCircleOutlined />
              </Tooltip>
            }>
              {evaluation.historical_accuracy !== null ? 
                `${evaluation.historical_accuracy.toFixed(1)}%` : 
                <Text type="secondary">数据不足</Text>
              }
            </Descriptions.Item>
            
            <Descriptions.Item label={
              <Tooltip title="预计该信号可能带来的收益百分比">
                预期收益 <QuestionCircleOutlined />
              </Tooltip>
            }>
              <Text type="success">+{evaluation.expected_return.toFixed(1)}%</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label={
              <Tooltip title="预计该信号可能带来的最大风险(回撤)百分比">
                风险比率 <QuestionCircleOutlined />
              </Tooltip>
            }>
              <Text type="danger">-{evaluation.risk_ratio.toFixed(1)}%</Text>
            </Descriptions.Item>
            
            <Descriptions.Item label={
              <Tooltip title="收益与风险的比值，越高越好">
                风险收益比 <QuestionCircleOutlined />
              </Tooltip>
            }>
              <Space>
                <Text strong>{evaluation.risk_return_ratio.toFixed(2)}</Text>
                <Tag color={riskReturnRating.color}>{riskReturnRating.text}</Tag>
              </Space>
            </Descriptions.Item>
          </Descriptions>
        </Col>
      </Row>
      
      <Divider orientation="left">评分因素</Divider>
      
      <Row gutter={[16, 8]}>
        <Col span={12}>
          <Tooltip title="基于信号Z得分和强度的评分">
            <div>信号强度: {evaluation.factors.strength.toFixed(1)}/30</div>
            <Progress percent={evaluation.factors.strength / 30 * 100} size="small" />
          </Tooltip>
        </Col>
        
        <Col span={12}>
          <Tooltip title="基于市场条件和时机分析的评分">
            <div>市场条件: {evaluation.factors.market_condition.toFixed(1)}/25</div>
            <Progress percent={evaluation.factors.market_condition / 25 * 100} size="small" />
          </Tooltip>
        </Col>
        
        <Col span={12}>
          <Tooltip title="基于与历史信号的一致性分析">
            <div>信号一致性: {evaluation.factors.consistency.toFixed(1)}/25</div>
            <Progress percent={evaluation.factors.consistency / 25 * 100} size="small" />
          </Tooltip>
        </Col>
        
        <Col span={12}>
          <Tooltip title="基于技术指标支持程度的评分">
            <div>技术指标支持: {evaluation.factors.technical_support.toFixed(1)}/20</div>
            <Progress percent={evaluation.factors.technical_support / 20 * 100} size="small" />
          </Tooltip>
        </Col>
      </Row>
    </Card>
  );
};

export default SignalQualityCard; 