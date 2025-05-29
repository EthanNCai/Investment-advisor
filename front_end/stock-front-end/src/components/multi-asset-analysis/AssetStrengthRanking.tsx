import React from 'react';
import { Card, Table, Tag, Tooltip, Progress } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, MinusOutlined } from '@ant-design/icons';

interface AssetStrengthRankingProps {
  data: any;
}

const AssetStrengthRanking: React.FC<AssetStrengthRankingProps> = ({ data }) => {
  if (!data || !data.assetStrengthRanking) {
    return (
      <Card title="资产强弱排名" style={{ height: '100%', minHeight: 400 }}>
        <div style={{ textAlign: 'center', padding: 100, color: '#999' }}>
          暂无排名数据
        </div>
      </Card>
    );
  }

  const { assetStrengthRanking, assets, assetNames } = data;

  // 创建表格数据
  const tableData = assetStrengthRanking.map((item: any, index: number) => {
    const code = assets[item.assetIndex];
    const name = assetNames[item.assetIndex];
    
    // 计算进度条显示
    const progressPercent = (item.score + 100) / 2; // 转换-100~100的得分到0~100的百分比
    
    // 确定强弱状态
    let status = 'normal';
    let statusIcon = <MinusOutlined />;
    let statusText = '中性';
    let statusColor = 'orange';
    
    if (item.score >= 60) {
      status = 'strong';
      statusIcon = <ArrowUpOutlined />;
      statusText = '强势';
      statusColor = 'green';
    } else if (item.score > 40) {
      status = 'moderately-strong';
      statusIcon = <ArrowUpOutlined />;
      statusText = '较强势';
      statusColor = 'lime';
    } else if (item.score <= -40) {
      status = 'weak';
      statusIcon = <ArrowDownOutlined />;
      statusText = '弱势';
      statusColor = 'red';
    } else if (item.score < -25 && item.score > -40) {
      status = 'moderately-weak';
      statusIcon = <ArrowDownOutlined />;
      statusText = '较弱势';
      statusColor = 'pink';
    }
    
    return {
      key: index,
      rank: index + 1,
      code,
      name,
      score: item.score,
      progress: progressPercent,
      status,
      statusIcon,
      statusText,
      statusColor
    };
  });

  const columns = [
    {
      title: '排名',
      dataIndex: 'rank',
      key: 'rank',
      width: 60,
      render: (rank: number) => (
        <div style={{ 
          textAlign: 'center', 
          fontWeight: 'bold',
          color: rank <= 1 ? '#f5222d' : rank <= 2 ? '#fa8c16' : '#8c8c8c'
        }}>
          {rank}
        </div>
      )
    },
    {
      title: '资产',
      dataIndex: 'code',
      key: 'code',
      render: (code: string, record: any) => (
        <Tooltip title={`${record.name} (${code})`}>
          <div style={{ maxWidth: 120, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {record.name} ({code})
          </div>
        </Tooltip>
      )
    },
    {
      title: '强弱状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (_: any, record: any) => (
        <Tag color={record.statusColor} icon={record.statusIcon}>
          {record.statusText}
        </Tag>
      )
    },
    {
      title: '综合得分',
      dataIndex: 'score',
      key: 'score',
      width: 200,
      render: (score: number, record: any) => (
        <Tooltip title={`综合得分: ${score.toFixed(1)}`}>
          <Progress
            percent={record.progress}
            strokeColor={
              score > 60 ? '#52c41a' :
              score > 40 ? '#a0d911' :
              score > -25 ? '#faad14' :
              score > -40 ? '#ff7875' :
              '#f5222d'
            }
            size="small"
            format={() => score.toFixed(1)}
          />
        </Tooltip>
      )
    }
  ];

  return (
    <Card title="资产强弱排名" style={{ height: '100%', minHeight: 400 }}>
      <Table 
        dataSource={tableData} 
        columns={columns} 
        pagination={false}
        size="small"
      />
      <div style={{ marginTop: 16, fontSize: '0.9em', color: '#666' }}>
        <p>
          <Tag color="green" icon={<ArrowUpOutlined />}>强势</Tag> 
          表示该资产相对其他资产整体处于强势地位，可考虑做多
        </p>
        <p>
          <Tag color="lime" icon={<ArrowUpOutlined />}>较强势</Tag> 
          表示该资产相对其他资产整体处于较强势地位，可考虑做多
        </p>
        <p>
          <Tag color="pink" icon={<ArrowDownOutlined />}>较弱势</Tag> 
          表示该资产相对其他资产整体处于较弱势地位，可考虑做空
        </p>
        <p>
          <Tag color="red" icon={<ArrowDownOutlined />}>弱势</Tag> 
          表示该资产相对其他资产整体处于弱势地位，可考虑做空
        </p>
      </div>
    </Card>
  );
};

export default AssetStrengthRanking; 