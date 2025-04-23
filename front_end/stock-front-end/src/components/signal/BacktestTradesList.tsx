import React, { useState } from 'react';
import { Table, Tag, Typography, Badge, Statistic, Space, Button, Row, Col } from 'antd';
import { ArrowUpOutlined, ArrowDownOutlined, DownloadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import type { Key } from 'react';

interface Trade {
  id: number;
  entry_date: string;
  exit_date: string | null;
  entry_price: number;
  exit_price: number | null;
  position_type: string;
  position_size: number;
  pnl: number;
  pnl_percent: number;
  status: 'open' | 'closed';
}

interface BacktestTradesListProps {
  trades: Trade[];
}

const BacktestTradesList: React.FC<BacktestTradesListProps> = ({ trades }) => {
  const [filteredInfo, setFilteredInfo] = useState<any>({});
  const [sortedInfo, setSortedInfo] = useState<any>({
    columnKey: 'entry_date',
    order: 'descend',
  });

  // 计算统计信息
  const profitableTrades = trades.filter(trade => trade.pnl > 0);
  const lossTrades = trades.filter(trade => trade.pnl < 0);
  
  const totalProfit = profitableTrades.reduce((sum, trade) => sum + trade.pnl, 0);
  const totalLoss = lossTrades.reduce((sum, trade) => sum + trade.pnl, 0);
  
  const avgProfit = profitableTrades.length > 0 ? totalProfit / profitableTrades.length : 0;
  const avgLoss = lossTrades.length > 0 ? totalLoss / lossTrades.length : 0;
  
  const handleChange = (pagination: any, filters: any, sorter: any) => {
    setFilteredInfo(filters);
    setSortedInfo(sorter);
  };

  const handleExportCSV = () => {
    // 构建CSV内容
    const headers = ['ID', '入场日期', '出场日期', '仓位类型', '入场价格', '出场价格', '仓位大小', '盈亏', '盈亏百分比', '状态'];
    const csvContent = [
      headers.join(','),
      ...trades.map(trade => [
        trade.id,
        trade.entry_date,
        trade.exit_date || '-',
        trade.position_type.includes('long') ? '做多' : '做空',
        trade.entry_price,
        trade.exit_price || '-',
        trade.position_size,
        trade.pnl.toFixed(2),
        (trade.pnl_percent * 100).toFixed(2) + '%',
        trade.status === 'open' ? '持仓中' : '已平仓'
      ].join(','))
    ].join('\n');
    
    // 创建Blob并下载
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', '回测交易记录.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const resetFilters = () => {
    setFilteredInfo({});
    setSortedInfo({
      columnKey: 'entry_date',
      order: 'descend',
    });
  };

  const columns: ColumnsType<Trade> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 60,
    },
    {
      title: '入场日期',
      dataIndex: 'entry_date',
      key: 'entry_date',
      sorter: (a: Trade, b: Trade) => new Date(a.entry_date).getTime() - new Date(b.entry_date).getTime(),
      sortOrder: sortedInfo.columnKey === 'entry_date' && sortedInfo.order,
    },
    {
      title: '出场日期',
      dataIndex: 'exit_date',
      key: 'exit_date',
      render: (exit_date: string | null) => exit_date || '-',
      sorter: (a: Trade, b: Trade) => {
        if (!a.exit_date) return 1;
        if (!b.exit_date) return -1;
        return new Date(a.exit_date).getTime() - new Date(b.exit_date).getTime();
      },
      sortOrder: sortedInfo.columnKey === 'exit_date' && sortedInfo.order,
    },
    {
      title: '仓位',
      dataIndex: 'position_type',
      key: 'position_type',
      render: (type: string) => (
        <Tag color={type.includes('long') ? '#52c41a' : '#f5222d'}>
          {type.includes('long') ? '做多' : '做空'}
        </Tag>
      ),
      filters: [
        { text: '做多', value: 'long' },
        { text: '做空', value: 'short' },
      ],
      filteredValue: filteredInfo.position_type || null,
      onFilter: (value: Key | boolean, record: Trade) => 
        record.position_type.includes(String(value)),
    },
    {
      title: '入场价格',
      dataIndex: 'entry_price',
      key: 'entry_price',
      render: (price: number) => price.toFixed(4),
    },
    {
      title: '出场价格',
      dataIndex: 'exit_price',
      key: 'exit_price',
      render: (price: number | null) => price ? price.toFixed(4) : '-',
    },
    {
      title: '仓位大小',
      dataIndex: 'position_size',
      key: 'position_size',
      render: (size: number) => `￥${size.toLocaleString()}`,
      sorter: (a: Trade, b: Trade) => a.position_size - b.position_size,
      sortOrder: sortedInfo.columnKey === 'position_size' && sortedInfo.order,
    },
    {
      title: '盈亏',
      dataIndex: 'pnl',
      key: 'pnl',
      render: (pnl: number) => (
        <span style={{ color: pnl > 0 ? '#52c41a' : pnl < 0 ? '#f5222d' : 'inherit' }}>
          {pnl > 0 ? '+' : ''}{pnl.toFixed(2)}
        </span>
      ),
      sorter: (a: Trade, b: Trade) => a.pnl - b.pnl,
      sortOrder: sortedInfo.columnKey === 'pnl' && sortedInfo.order,
    },
    {
      title: '盈亏百分比',
      dataIndex: 'pnl_percent',
      key: 'pnl_percent',
      render: (percent: number) => (
        <span style={{ color: percent > 0 ? '#52c41a' : percent < 0 ? '#f5222d' : 'inherit' }}>
          {percent > 0 ? '+' : ''}{(percent * 100).toFixed(2)}%
        </span>
      ),
      sorter: (a: Trade, b: Trade) => a.pnl_percent - b.pnl_percent,
      sortOrder: sortedInfo.columnKey === 'pnl_percent' && sortedInfo.order,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge 
          status={status === 'open' ? 'processing' : 'success'} 
          text={status === 'open' ? '持仓中' : '已平仓'} 
        />
      ),
      filters: [
        { text: '持仓中', value: 'open' },
        { text: '已平仓', value: 'closed' },
      ],
      filteredValue: filteredInfo.status || null,
      onFilter: (value: Key | boolean, record: Trade) => 
        record.status === String(value),
    },
  ];

  return (
    <div className="backtest-trades-list">
      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="交易总数"
            value={trades.length}
            valueStyle={{ color: '#1890ff' }}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="盈利交易数"
            value={profitableTrades.length}
            suffix={`/ ${trades.length}`}
            valueStyle={{ color: '#52c41a' }}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="平均盈利"
            value={avgProfit}
            precision={2}
            valueStyle={{ color: '#52c41a' }}
            prefix={<ArrowUpOutlined />}
            suffix="￥"
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Statistic
            title="平均亏损"
            value={Math.abs(avgLoss)}
            precision={2}
            valueStyle={{ color: '#f5222d' }}
            prefix={<ArrowDownOutlined />}
            suffix="￥"
          />
        </Col>
      </Row>

      <Space style={{ marginBottom: 16 }}>
        <Button onClick={resetFilters}>重置筛选</Button>
        <Button 
          type="primary" 
          icon={<DownloadOutlined />} 
          onClick={handleExportCSV}
        >
          导出CSV
        </Button>
      </Space>
      
      <Table 
        columns={columns} 
        dataSource={trades}
        rowKey="id"
        onChange={handleChange}
        scroll={{ x: 'max-content' }}
        pagination={{ 
          pageSize: 10, 
          showSizeChanger: true, 
          pageSizeOptions: ['10', '20', '50', '100'],
          showTotal: (total, range) => `${range[0]}-${range[1]} 共 ${total} 条记录`
        }}
        summary={pageData => {
          let totalPnl = 0;
          let totalPnlPercent = 0;
          
          pageData.forEach(({ pnl, pnl_percent }) => {
            totalPnl += pnl;
            totalPnlPercent += pnl_percent;
          });
          
          const avgPnlPercent = pageData.length > 0 ? totalPnlPercent / pageData.length : 0;
          
          return (
            <>
              <Table.Summary.Row>
                <Table.Summary.Cell index={0} colSpan={6} align="right">
                  <Typography.Text strong>当前页总计:</Typography.Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={1} colSpan={1}>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={2} colSpan={1}>
                  <Typography.Text strong style={{ color: totalPnl > 0 ? '#52c41a' : '#f5222d' }}>
                    {totalPnl.toFixed(2)}
                  </Typography.Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={3} colSpan={1}>
                  <Typography.Text strong style={{ color: avgPnlPercent > 0 ? '#52c41a' : '#f5222d' }}>
                    {(avgPnlPercent * 100).toFixed(2)}%
                  </Typography.Text>
                </Table.Summary.Cell>
                <Table.Summary.Cell index={4} colSpan={1}>
                </Table.Summary.Cell>
              </Table.Summary.Row>
            </>
          );
        }}
      />
    </div>
  );
};

export default BacktestTradesList; 