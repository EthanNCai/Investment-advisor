import React from 'react';
import { Card, Table, Tag, Button, Tooltip, Progress } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, SearchOutlined } from '@ant-design/icons';

interface OptimalPairsTableProps {
  data: any;
  onPairSelect: (assetA: string, assetB: string) => void;
}

const OptimalPairsTable: React.FC<OptimalPairsTableProps> = ({ data, onPairSelect }) => {
  if (!data || !data.optimalPairs) {
    return (
      <Card title="最优交易对" style={{ height: '100%', minHeight: 500 }}>
        <div style={{ textAlign: 'center', padding: 100, color: '#999' }}>
          暂无最优交易对数据
        </div>
      </Card>
    );
  }

  const { optimalPairs, assets, assetNames } = data;

  // 创建表格数据
  const tableData = optimalPairs.map((pair: any, index: number) => {
    const assetACode = assets[pair.assetAIndex];
    const assetBCode = assets[pair.assetBIndex];
    const assetAName = assetNames[pair.assetAIndex];
    const assetBName = assetNames[pair.assetBIndex];
    
    // 强度百分比转换
    const strengthPercent = Math.min(Math.abs(pair.signalStrength), 100);
    
    // 交易方向和类型
    const direction = pair.direction; // 1表示做多A做空B，-1表示做空A做多B
    const tradeSide = direction === 1 ? 
      { long: assetACode, short: assetBCode, longName: assetAName, shortName: assetBName } : 
      { long: assetBCode, short: assetACode, longName: assetBName, shortName: assetAName };
    
    // 根据信号强度确定颜色
    const strengthColor = Math.abs(pair.signalStrength) >= 70 ? '#f5222d' : 
                          Math.abs(pair.signalStrength) >= 50 ? '#fa8c16' : 
                          Math.abs(pair.signalStrength) >= 30 ? '#52c41a' : '#8c8c8c';
    
    // 根据信号强度确定交易建议
    let recommendation = '观望';
    if (Math.abs(pair.signalStrength) >= 70) recommendation = '强烈推荐';
    else if (Math.abs(pair.signalStrength) >= 50) recommendation = '建议';
    else if (Math.abs(pair.signalStrength) >= 30) recommendation = '可考虑';
    
    return {
      key: index,
      rank: index + 1,
      assetACode,
      assetBCode,
      assetAName,
      assetBName,
      signalStrength: pair.signalStrength,
      direction,
      tradeSide,
      strengthPercent,
      strengthColor,
      recommendation,
      expectedReturn: pair.expectedReturn
    };
  });

  // 处理点击操作按钮跳转到资产对详情页
  const handleViewPairDetail = (assetACode: string, assetBCode: string) => {
    onPairSelect(assetACode, assetBCode);
    
    // 自动跳转到"资产对详情"页面
    const detailTabPane = document.querySelector('.ant-tabs-tab[data-node-key="2"]') as HTMLElement;
    if (detailTabPane) {
      detailTabPane.click();
    }
  };

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
          color: rank <= 3 ? '#f5222d' : '#8c8c8c'
        }}>
          {rank}
        </div>
      )
    },
    {
      title: '交易对',
      dataIndex: 'pair',
      key: 'pair',
      render: (_: any, record: any) => (
        <div>
          <div>{record.assetAName} ({record.assetACode})</div>
          <div>vs</div>
          <div>{record.assetBName} ({record.assetBCode})</div>
        </div>
      )
    },
    {
      title: '交易方向',
      dataIndex: 'direction',
      key: 'direction',
      width: 180,
      render: (_: any, record: any) => (
        <div>
          {record.direction === 1 ? (
            <>
              <div>
                <Tag color="#EE6666" icon={<ArrowUpOutlined />}>
                  做多 {record.assetAName} ({record.assetACode})
                </Tag>
              </div>
              <div style={{ marginTop: 5 }}>
                <Tag color="#EE6666" icon={<ArrowDownOutlined />}>
                  做空 {record.assetBName} ({record.assetBCode})
                </Tag>
              </div>
            </>
          ) : (
            <>
              <div>
                <Tag color="#91CC75" icon={<ArrowUpOutlined />}>
                  做多 {record.assetBName} ({record.assetBCode})
                </Tag>
              </div>
              <div style={{ marginTop: 5 }}>
                <Tag color="#91CC75" icon={<ArrowDownOutlined />}>
                  做空 {record.assetAName} ({record.assetACode})
                </Tag>
              </div>
            </>
          )}
        </div>
      )
    },
    {
      title: '信号强度',
      dataIndex: 'signalStrength',
      key: 'signalStrength',
      width: 120,
      render: (strength: number, record: any) => (
        <Tooltip title={`信号强度: ${Math.abs(strength).toFixed(1)}`}>
          <Progress
            percent={record.strengthPercent}
            strokeColor={record.direction === 1 ? '#EE6666' : '#91CC75'}
            size="small"
            format={() => `${Math.abs(strength).toFixed(1)}`}
          />
        </Tooltip>
      )
    },
    {
      title: '建议',
      dataIndex: 'recommendation',
      key: 'recommendation',
      width: 100,
      render: (text: string, record: any) => (
        <Tag color={record.direction === 1 ? '#EE6666' : '#91CC75'}>
          {text}
        </Tag>
      )
    },
    {
      title: '预期收益',
      dataIndex: 'expectedReturn',
      key: 'expectedReturn',
      width: 100,
      render: (value: number) => (
        <span style={{ color: value > 0 ? '#52c41a' : '#f5222d' }}>
          {value > 0 ? '+' : ''}{value.toFixed(2)}%
        </span>
      )
    },
    {
      title: '操作',
      key: 'action',
      width: 80,
      render: (_: any, record: any) => (
        <Tooltip title="查看详情">
          <Button 
            type="primary" 
            icon={<SearchOutlined />} 
            size="small"
            onClick={() => handleViewPairDetail(record.assetACode, record.assetBCode)}
          />
        </Tooltip>
      )
    }
  ];

  return (
    <Card title="最优交易对" style={{ minHeight: 500 }}>
      <Table 
        dataSource={tableData} 
        columns={columns}
        pagination={{ pageSize: 5 }}
      />
      <div style={{ marginTop: 16, fontSize: '0.9em', color: '#666' }}>
        <p>最优交易对根据当前价格比值异常程度排序</p>
        <p>点击"查看详情"可以查看该资产对的详细比值分析图表</p>
        <p>说明: 红色表示做多第一个资产(A)做空第二个资产(B)，绿色表示做空第一个资产(A)做多第二个资产(B)</p>
      </div>
    </Card>
  );
};

export default OptimalPairsTable; 