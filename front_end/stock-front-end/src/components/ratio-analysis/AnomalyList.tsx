import React from 'react';
import { Table } from 'antd';

interface Anomaly {
  index: number;
  value: number;
  z_score: number;
  deviation: number;
}

interface AnomalyListProps {
  anomalies: Anomaly[];
}

const AnomalyList: React.FC<AnomalyListProps> = ({ anomalies }) => {
  return (
    <div style={{ marginTop: '16px' }}>
      <h4>异常值列表</h4>
      <Table 
        dataSource={anomalies}
        columns={[
          {
            title: '日期',
            dataIndex: 'index',
            key: 'index',
            render: (index) => chartData?.dates[index] || '-'
          },
          {
            title: '比值',
            dataIndex: 'value',
            key: 'value',
            render: (value) => value.toFixed(4)
          },
          {
            title: '偏离度',
            dataIndex: 'deviation',
            key: 'deviation',
            render: (deviation) => `${(deviation * 100).toFixed(2)}%`
          },
          {
            title: 'Z分数',
            dataIndex: 'z_score',
            key: 'z_score',
            render: (z_score) => z_score.toFixed(2)
          }
        ]}
        size="small"
        pagination={false}
      />
    </div>
  );
};

export default AnomalyList; 