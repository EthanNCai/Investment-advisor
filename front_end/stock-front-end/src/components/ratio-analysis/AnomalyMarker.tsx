import React from 'react';
import { Popover, Tag } from 'antd';

interface Anomaly {
  index: number;
  value: number;
  z_score: number;
  deviation: number;
}

interface AnomalyMarkerProps {
  anomaly: Anomaly;
  x: number;
  y: number;
  date: string;
}

const AnomalyMarker: React.FC<AnomalyMarkerProps> = ({ anomaly, x, y, date }) => {
  const size = anomaly.z_score > 3 ? 8 : anomaly.z_score > 2.5 ? 7 : 6;
  const color = anomaly.z_score > 3 ? '#ff4d4f' : anomaly.z_score > 2.5 ? '#faad14' : '#1890ff';

  const content = (
    <div>
      <p><strong>日期:</strong> {date}</p>
      <p><strong>比值:</strong> {anomaly.value.toFixed(4)}</p>
      <p><strong>Z分数:</strong> <Tag color={anomaly.z_score > 3 ? 'red' : anomaly.z_score > 2.5 ? 'orange' : 'blue'}>{anomaly.z_score.toFixed(2)}</Tag></p>
      <p><strong>偏离度:</strong> {(anomaly.deviation * 100).toFixed(2)}%</p>
    </div>
  );

  return (
    <Popover content={content} title="异常点详情">
      <circle
        cx={x}
        cy={y}
        r={size}
        fill={color}
        stroke="#fff"
        strokeWidth={1}
        style={{ cursor: 'pointer' }}
      />
    </Popover>
  );
};

export default AnomalyMarker; 