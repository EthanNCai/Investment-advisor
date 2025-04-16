import React from 'react';

interface WarningIndicatorProps {
  level: 'normal' | 'medium' | 'high';
}

const WarningIndicator: React.FC<WarningIndicatorProps> = ({ level }) => {
  const colors = {
    normal: '#52c41a',
    medium: '#faad14',
    high: '#f5222d'
  };
  
  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center',
      padding: '4px 8px',
      borderRadius: '4px',
      backgroundColor: colors[level],
      color: 'white',
      fontSize: '12px'
    }}>
      {level === 'normal' ? '正常' : level === 'medium' ? '中等风险' : '高风险'}
    </div>
  );
};

export default WarningIndicator; 