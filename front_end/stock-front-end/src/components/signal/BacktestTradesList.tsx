import React, {useState} from 'react';
import {Table, Tag, Typography, Badge, Statistic, Space, Button, Row, Col, Card} from 'antd';
import {ArrowUpOutlined, ArrowDownOutlined, DownloadOutlined, ClockCircleOutlined} from '@ant-design/icons';
import type {ColumnsType} from 'antd/es/table';
import type {Key} from 'react';

interface Trade {
    id: number;
    entry_date: string;
    exit_date: string | null;
    holding_days: number;
    entry_price: number;
    exit_price: number | null;
    position_type: string;
    position_size: number;
    pnl: number;
    pnl_percent: number;
    status: 'open' | 'closed';
    exit_reason?: string;
    signal_value?: number;
}

interface BacktestTradesListProps {
    trades: Trade[];
}

const BacktestTradesList: React.FC<BacktestTradesListProps> = ({trades}) => {
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
    
    // 计算平均持仓天数
    const avgHoldingDays = trades.length > 0 
        ? trades.reduce((sum, trade) => sum + (trade.holding_days || 0), 0) / trades.length 
        : 0;
    
    // 计算出场原因的分布
    const exitReasonCounts: Record<string, number> = {};
    trades.forEach(trade => {
        if (trade.exit_reason) {
            exitReasonCounts[trade.exit_reason] = (exitReasonCounts[trade.exit_reason] || 0) + 1;
        }
    });

    const handleChange = (pagination: any, filters: any, sorter: any) => {
        setFilteredInfo(filters);
        setSortedInfo(sorter);
    };

    const handleExportCSV = () => {
        // 构建CSV内容
        const headers = ['ID', '入场日期', '出场日期', '持仓天数', '仓位类型', '入场价格', '出场价格', '仓位大小', '盈亏', '盈亏百分比', '状态', '出场原因', '信号值'];
        const csvContent = [
            headers.join(','),
            ...trades.map(trade => [
                trade.id,
                trade.entry_date,
                trade.exit_date || '-',
                trade.holding_days || '-',
                trade.position_type,
                trade.entry_price,
                trade.exit_price || '-',
                trade.position_size,
                trade.pnl.toFixed(2),
                (trade.pnl_percent * 100).toFixed(2) + '%',
                trade.status,
                trade.exit_reason || '-',
                trade.signal_value?.toFixed(2) || '-'
            ].join(','))
        ].join('\n');

        // 创建Blob并下载
        const blob = new Blob([csvContent], {type: 'text/csv;charset=utf-8;'});
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
    
    // 获取出场原因的标签颜色
    const getExitReasonColor = (reason: string) => {
        switch(reason) {
            case 'signal':
                return 'blue';
            case 'stop_loss':
                return 'red';
            case 'take_profit':
                return 'green';
            case 'trailing_stop':
                return 'purple';
            case 'breakeven_stop':
                return 'orange';
            case 'time_stop':
                return 'cyan';
            case 'end_of_backtest':
                return 'default';
            default:
                return 'default';
        }
    };
    
    // 获取出场原因的显示文本
    const getExitReasonText = (reason: string) => {
        switch(reason) {
            case 'signal':
                return '信号平仓';
            case 'stop_loss':
                return '止损';
            case 'take_profit':
                return '止盈';
            case 'trailing_stop':
                return '追踪止损';
            case 'breakeven_stop':
                return '保本止损';
            case 'time_stop':
                return '时间止损';
            case 'end_of_backtest':
                return '回测结束';
            default:
                return reason;
        }
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
            title: '持仓天数',
            dataIndex: 'holding_days',
            key: 'holding_days',
            render: (days: number) => days || '-',
            sorter: (a: Trade, b: Trade) => (a.holding_days || 0) - (b.holding_days || 0),
            sortOrder: sortedInfo.columnKey === 'holding_days' && sortedInfo.order,
        },
        {
            title: '仓位',
            dataIndex: 'position_type',
            key: 'position_type',
            render: (type: string) => {
                if (type.includes('long_') && type.includes('_A')) {
                    return <Tag color="#52c41a">做多A</Tag>;
                } else if (type.includes('short_') && type.includes('_A')) {
                    return <Tag color="#f5222d">做空A</Tag>;
                } else if (type.includes('long_') && type.includes('_B')) {
                    return <Tag color="#1890ff">做多B</Tag>;
                } else if (type.includes('short_') && type.includes('_B')) {
                    return <Tag color="#fa8c16">做空B</Tag>;
                } else if (type.includes('long')) {
                    return <Tag color="#52c41a">做多</Tag>;
                } else if (type.includes('short')) {
                    return <Tag color="#f5222d">做空</Tag>;
                }
                return <Tag>{type}</Tag>;
            },
            filters: [
                {text: '做多', value: 'long'},
                {text: '做空', value: 'short'},
                {text: '做多A', value: 'long_A'},
                {text: '做空A', value: 'short_A'},
                {text: '做多B', value: 'long_B'},
                {text: '做空B', value: 'short_B'},
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
                <span style={{color: pnl > 0 ? '#52c41a' : pnl < 0 ? '#f5222d' : 'inherit'}}>
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
                <span style={{color: percent > 0 ? '#52c41a' : percent < 0 ? '#f5222d' : 'inherit'}}>
                    {percent > 0 ? '+' : ''}{(percent * 100).toFixed(2)}%
                </span>
            ),
            sorter: (a: Trade, b: Trade) => a.pnl_percent - b.pnl_percent,
            sortOrder: sortedInfo.columnKey === 'pnl_percent' && sortedInfo.order,
        },
        {
            title: '出场原因',
            dataIndex: 'exit_reason',
            key: 'exit_reason',
            render: (reason: string) => reason ? (
                <Tag color={getExitReasonColor(reason)}>
                    {getExitReasonText(reason)}
                </Tag>
            ) : '-',
            filters: [
                {text: '信号平仓', value: 'signal'},
                {text: '止损', value: 'stop_loss'},
                {text: '止盈', value: 'take_profit'},
                {text: '追踪止损', value: 'trailing_stop'},
                {text: '保本止损', value: 'breakeven_stop'},
                {text: '时间止损', value: 'time_stop'},
                {text: '回测结束', value: 'end_of_backtest'},
            ],
            filteredValue: filteredInfo.exit_reason || null,
            onFilter: (value: Key | boolean, record: Trade) =>
                record.exit_reason === String(value),
        },
        {
            title: '信号值',
            dataIndex: 'signal_value',
            key: 'signal_value',
            render: (signal: number | undefined) => signal !== undefined ? signal.toFixed(2) : '-',
            sorter: (a: Trade, b: Trade) => (a.signal_value || 0) - (b.signal_value || 0),
            sortOrder: sortedInfo.columnKey === 'signal_value' && sortedInfo.order,
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
                {text: '持仓中', value: 'open'},
                {text: '已平仓', value: 'closed'},
            ],
            filteredValue: filteredInfo.status || null,
            onFilter: (value: Key | boolean, record: Trade) =>
                record.status === String(value),
        },
    ];

    return (
        <div className="backtest-trades-list">
            <Row gutter={[16, 16]} style={{marginBottom: 16}}>
                <Col xs={24} sm={12} md={6}>
                    <Card>
                        <Statistic
                            title="交易总数"
                            value={trades.length}
                            valueStyle={{color: '#1890ff'}}
                        />
                    </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                    <Card>
                        <Statistic
                            title="盈利交易数"
                            value={profitableTrades.length}
                            suffix={`/ ${trades.length}`}
                            valueStyle={{color: '#52c41a'}}
                        />
                    </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                    <Card>
                        <Statistic
                            title="平均盈利"
                            value={avgProfit}
                            precision={2}
                            valueStyle={{color: '#52c41a'}}
                            prefix={<ArrowUpOutlined/>}
                            suffix="￥"
                        />
                    </Card>
                </Col>
                <Col xs={24} sm={12} md={6}>
                    <Card>
                        <Statistic
                            title="平均亏损"
                            value={Math.abs(avgLoss)}
                            precision={2}
                            valueStyle={{color: '#f5222d'}}
                            prefix={<ArrowDownOutlined/>}
                            suffix="￥"
                        />
                    </Card>
                </Col>
                
                <Col xs={24} sm={12} md={6}>
                    <Card>
                        <Statistic
                            title="平均持仓天数"
                            value={avgHoldingDays}
                            precision={1}
                            valueStyle={{color: '#1890ff'}}
                            prefix={<ClockCircleOutlined/>}
                            suffix="天"
                        />
                    </Card>
                </Col>
                
                {Object.entries(exitReasonCounts).map(([reason, count], index) => (
                    <Col xs={24} sm={12} md={6} key={index}>
                        <Card>
                            <Statistic
                                title={`${getExitReasonText(reason)}次数`}
                                value={count}
                                valueStyle={{color: getExitReasonColor(reason) === 'default' ? '#1890ff' : ''}}
                                suffix={`/ ${trades.length}`}
                            />
                        </Card>
                    </Col>
                ))}
            </Row>

            <Space style={{marginBottom: 16}}>
                <Button onClick={resetFilters}>重置筛选</Button>
                <Button
                    type="primary"
                    icon={<DownloadOutlined/>}
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
                scroll={{x: 'max-content'}}
                pagination={{
                    pageSize: 10,
                    showSizeChanger: true,
                    pageSizeOptions: ['10', '20', '50', '100'],
                }}
                summary={pageData => {
                    let totalPnl = 0;
                    let totalPositionSize = 0;
                    
                    pageData.forEach(({pnl, position_size}) => {
                        totalPnl += pnl;
                        totalPositionSize += position_size;
                    });
                    
                    return (
                        <>
                            <Table.Summary.Row>
                                <Table.Summary.Cell index={0} colSpan={7}>
                                    <strong>当前页合计</strong>
                                </Table.Summary.Cell>
                                <Table.Summary.Cell index={7}>
                                    <strong>{`￥${totalPositionSize.toLocaleString()}`}</strong>
                                </Table.Summary.Cell>
                                <Table.Summary.Cell index={8}>
                                    <strong style={{color: totalPnl > 0 ? '#52c41a' : totalPnl < 0 ? '#f5222d' : 'inherit'}}>
                                        {totalPnl > 0 ? '+' : ''}{totalPnl.toFixed(2)}
                                    </strong>
                                </Table.Summary.Cell>
                                <Table.Summary.Cell index={9} colSpan={4}></Table.Summary.Cell>
                            </Table.Summary.Row>
                        </>
                    );
                }}
            />
        </div>
    );
};

export default BacktestTradesList; 