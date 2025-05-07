import { useContext } from 'react';
import { 
    Paper, 
    Typography, 
    Box, 
    Skeleton,
    Tabs,
    Tab
} from '@mui/material';
import { 
    LineChart, 
    Line, 
    XAxis, 
    YAxis, 
    CartesianGrid, 
    Tooltip, 
    Legend, 
    ResponsiveContainer,
    BarChart,
    Bar,
    ComposedChart,
    Cell
} from 'recharts';
import { AssetSelectionContext } from './AssetInterfaces';
import { useState } from 'react';

// 添加recharts依赖
// npm install recharts

const StockKlineChart = () => {
    const context = useContext(AssetSelectionContext);
    const [activeTab, setActiveTab] = useState(0);
    
    if (!context) {
        return null;
    }
    
    const { 
        klineData, 
        selectedIndicators, 
        selectedAsset,
        isLoading,
        klineType,
        // 添加细粒度指标选择的状态，如果不存在则使用默认值
        selectedMALines = ['ma5', 'ma10', 'ma20', 'ma60'],
        selectedMACDLines = ['dif', 'dea', 'macd'],
        selectedRSILines = ['rsi6', 'rsi12', 'rsi24'],
        selectedKDJLines = ['k', 'd', 'j']
    } = context;
    
    const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
        setActiveTab(newValue);
    };
    
    if (isLoading) {
        return (
            <Paper elevation={3} sx={{ p: 2, height: '500px' }}>
                <Skeleton animation="wave" variant="rectangular" height={40} sx={{ mb: 2 }} />
                <Skeleton animation="wave" variant="rectangular" height={400} />
            </Paper>
        );
    }
    
    if (!selectedAsset || !klineData) {
        return (
            <Paper elevation={3} sx={{ p: 2, height: '500px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Typography variant="subtitle1" color="text.secondary">
                    请选择一个资产查看K线图
                </Typography>
            </Paper>
        );
    }
    
    // 构建图表数据
    const chartData = klineData.kline_data.map((item, index) => ({
        date: item.date,
        open: item.open,
        close: item.close,
        high: item.high,
        low: item.low,
        volume: item.volume,
        ma5: selectedIndicators.includes('ma') ? klineData.indicators.ma5[index] : null,
        ma10: selectedIndicators.includes('ma') ? klineData.indicators.ma10[index] : null,
        ma20: selectedIndicators.includes('ma') ? klineData.indicators.ma20[index] : null,
        ma60: selectedIndicators.includes('ma') ? klineData.indicators.ma60[index] : null,
        dif: selectedIndicators.includes('macd') ? klineData.indicators.macd.dif[index] : null,
        dea: selectedIndicators.includes('macd') ? klineData.indicators.macd.dea[index] : null,
        macd: selectedIndicators.includes('macd') ? klineData.indicators.macd.macd[index] : null,
        rsi6: selectedIndicators.includes('rsi') ? klineData.indicators.rsi.rsi6[index] : null,
        rsi12: selectedIndicators.includes('rsi') ? klineData.indicators.rsi.rsi12[index] : null,
        rsi24: selectedIndicators.includes('rsi') ? klineData.indicators.rsi.rsi24[index] : null,
        k: selectedIndicators.includes('kdj') && klineData.indicators.kdj ? klineData.indicators.kdj.k[index] : null,
        d: selectedIndicators.includes('kdj') && klineData.indicators.kdj ? klineData.indicators.kdj.d[index] : null,
        j: selectedIndicators.includes('kdj') && klineData.indicators.kdj ? klineData.indicators.kdj.j[index] : null,
    }));
    
    // 构建实时价格趋势数据
    const trendData = klineData.trends ? klineData.trends.map(item => ({
        date: item.date,
        price: item.current_price,
        volume: item.volume
    })) : [];
    
    // 处理MACD数据，仅保留DIF和DEA都不为0的数据
    const validMacdData = chartData.filter(item => {
        return item.dif !== null && item.dif !== 0 && item.dea !== null && item.dea !== 0;
    });
    
    // 自定义工具提示，显示字段名称和数据值
    const CustomTooltip = ({ active, payload, label }: any) => {
        if (active && payload && payload.length) {
            return (
                <Paper sx={{ p: 1, boxShadow: 3, backgroundColor: 'rgba(255, 255, 255, 0.9)' }}>
                    <Typography sx={{ fontWeight: 'bold', mb: 1 }}>
                        {klineType === 'realtime' ? label : `日期: ${label}`}
                    </Typography>
                    {payload.map((entry: any, index: number) => (
                        <Box key={`item-${index}`} sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
                            <Box 
                                component="span" 
                                sx={{ width: 10, height: 10, borderRadius: '50%', backgroundColor: entry.color, mr: 1 }}
                            />
                            <Typography variant="body2">
                                {entry.name}: {entry.value !== null ? Number(entry.value).toFixed(2) : '--'}
                            </Typography>
                        </Box>
                    ))}
                </Paper>
            );
        }
        return null;
    };
    
    const renderPriceChart = () => {
        // 实时价格趋势图
        if (klineType === 'realtime' && klineData.trends && klineData.trends.length > 0) {
            return (
                <ResponsiveContainer width="100%" height={400}>
                    <LineChart
                        data={trendData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            scale="band"
                            tick={{ fontSize: 12 }}
                            tickFormatter={(value) => {
                                // 显示时间部分 "HH:MM"
                                const parts = value.split(' ');
                                if (parts.length >= 2) {
                                    const timeParts = parts[1].split(':');
                                    return `${timeParts[0]}:${timeParts[1]}`;
                                }
                                return value;
                            }}
                        />
                        <YAxis 
                            yAxisId="left"
                            domain={['auto', 'auto']}
                            tick={{ fontSize: 12 }}
                            width={60}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Line 
                            type="monotone" 
                            dataKey="price" 
                            stroke="#ff7300" 
                            yAxisId="left" 
                            dot={false}
                            name="当前价格"
                        />
                    </LineChart>
                </ResponsiveContainer>
            );
        }
        
        // 标准K线图
        return (
            <ResponsiveContainer width="100%" height={400}>
                <LineChart
                    data={chartData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                        dataKey="date" 
                        scale="band"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => value.substring(5)} // 只显示月-日
                    />
                    <YAxis 
                        yAxisId="left"
                        domain={['auto', 'auto']}
                        tick={{ fontSize: 12 }}
                        width={60}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Line 
                        type="monotone" 
                        dataKey="close" 
                        stroke="#ff7300" 
                        yAxisId="left" 
                        dot={false}
                        name="收盘价"
                    />
                    
                    {selectedIndicators.includes('ma') && (
                        <>
                            {selectedMALines.includes('ma5') && (
                                <Line 
                                    type="monotone" 
                                    dataKey="ma5" 
                                    stroke="#8884d8" 
                                    yAxisId="left" 
                                    dot={false}
                                    name="MA5"
                                />
                            )}
                            {selectedMALines.includes('ma10') && (
                                <Line 
                                    type="monotone" 
                                    dataKey="ma10" 
                                    stroke="#82ca9d" 
                                    yAxisId="left" 
                                    dot={false}
                                    name="MA10"
                                />
                            )}
                            {selectedMALines.includes('ma20') && (
                                <Line 
                                    type="monotone" 
                                    dataKey="ma20" 
                                    stroke="#ffc658" 
                                    yAxisId="left" 
                                    dot={false}
                                    name="MA20"
                                />
                            )}
                            {selectedMALines.includes('ma60') && (
                                <Line 
                                    type="monotone" 
                                    dataKey="ma60" 
                                    stroke="#9467bd" 
                                    yAxisId="left" 
                                    dot={false}
                                    name="MA60"
                                />
                            )}
                        </>
                    )}
                </LineChart>
            </ResponsiveContainer>
        );
    };
    
    const renderVolumeChart = () => {
        // 实时成交量图
        if (klineType === 'realtime' && klineData.trends && klineData.trends.length > 0) {
            return (
                <ResponsiveContainer width="100%" height={400}>
                    <BarChart
                        data={trendData}
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                    >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis 
                            dataKey="date" 
                            scale="band"
                            tick={{ fontSize: 12 }}
                            tickFormatter={(value) => {
                                // 显示时间部分 "HH:MM"
                                const parts = value.split(' ');
                                if (parts.length >= 2) {
                                    const timeParts = parts[1].split(':');
                                    return `${timeParts[0]}:${timeParts[1]}`;
                                }
                                return value;
                            }}
                        />
                        <YAxis 
                            tick={{ fontSize: 12 }}
                            width={80}
                            tickFormatter={(value) => value >= 10000 ? `${(value / 10000).toFixed(1)}万` : value}
                        />
                        <Tooltip content={<CustomTooltip />} />
                        <Legend />
                        <Bar dataKey="volume" fill="#8884d8" name="成交量" />
                    </BarChart>
                </ResponsiveContainer>
            );
        }
        
        // 标准成交量图
        return (
            <ResponsiveContainer width="100%" height={400}>
                <BarChart
                    data={chartData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                        dataKey="date" 
                        scale="band"
                        tick={{ fontSize: 12 }}
                        tickFormatter={(value) => value.substring(5)} // 只显示月-日
                    />
                    <YAxis 
                        tick={{ fontSize: 12 }}
                        width={80}
                        tickFormatter={(value) => value >= 10000 ? `${(value / 10000).toFixed(1)}万` : value}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend />
                    <Bar dataKey="volume" fill="#8884d8" name="成交量" />
                </BarChart>
            </ResponsiveContainer>
        );
    };
    
    const renderMACDChart = () => (
        <ResponsiveContainer width="100%" height={400}>
            <ComposedChart
                data={validMacdData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                    dataKey="date" 
                    scale="band"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => value.substring(5)} // 只显示月-日
                />
                <YAxis 
                    tick={{ fontSize: 12 }}
                    width={60}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                
                {selectedMACDLines.includes('dif') && (
                    <Line type="monotone" dataKey="dif" stroke="#ff7300" dot={false} name="DIF线" />
                )}
                
                {selectedMACDLines.includes('dea') && (
                    <Line type="monotone" dataKey="dea" stroke="#82ca9d" dot={false} name="DEA线" />
                )}
                
                {selectedMACDLines.includes('macd') && (
                    <Bar dataKey="macd" name="MACD柱">
                        {validMacdData.map((entry, index) => (
                            <Cell 
                                key={`cell-${index}`} 
                                fill={(entry.macd === null ? '#8884d8' : (entry.macd >= 0 ? '#e74c3c' : '#2ecc71'))} 
                            />
                        ))}
                    </Bar>
                )}
            </ComposedChart>
        </ResponsiveContainer>
    );
    
    const renderRSIChart = () => (
        <ResponsiveContainer width="100%" height={400}>
            <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                    dataKey="date" 
                    scale="band"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => value.substring(5)} // 只显示月-日
                />
                <YAxis 
                    domain={[0, 100]}
                    tick={{ fontSize: 12 }}
                    width={60}
                    ticks={[0, 20, 30, 50, 70, 80, 100]}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                
                {selectedRSILines.includes('rsi6') && (
                    <Line type="monotone" dataKey="rsi6" stroke="#ff7300" dot={false} name="RSI6" />
                )}
                
                {selectedRSILines.includes('rsi12') && (
                    <Line type="monotone" dataKey="rsi12" stroke="#82ca9d" dot={false} name="RSI12" />
                )}
                
                {selectedRSILines.includes('rsi24') && (
                    <Line type="monotone" dataKey="rsi24" stroke="#8884d8" dot={false} name="RSI24" />
                )}
            </LineChart>
        </ResponsiveContainer>
    );
    
    const renderKDJChart = () => (
        <ResponsiveContainer width="100%" height={400}>
            <LineChart
                data={chartData}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
            >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                    dataKey="date" 
                    scale="band"
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => value.substring(5)} // 只显示月-日
                />
                <YAxis 
                    domain={['auto', 'auto']}
                    tick={{ fontSize: 12 }}
                    width={60}
                />
                <Tooltip content={<CustomTooltip />} />
                <Legend />
                
                {selectedKDJLines.includes('k') && (
                    <Line type="monotone" dataKey="k" stroke="#ff7300" dot={false} name="K值" />
                )}
                
                {selectedKDJLines.includes('d') && (
                    <Line type="monotone" dataKey="d" stroke="#82ca9d" dot={false} name="D值" />
                )}
                
                {selectedKDJLines.includes('j') && (
                    <Line type="monotone" dataKey="j" stroke="#8884d8" dot={false} name="J值" />
                )}
            </LineChart>
        </ResponsiveContainer>
    );
    
    // 检查KDJ指标是否存在
    const hasKdjIndicator = selectedIndicators.includes('kdj') && klineData.indicators.kdj !== undefined;
    
    return (
        <Paper elevation={3} sx={{ p: 2 }}>
            <Box sx={{ mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                    {selectedAsset.name} ({selectedAsset.code}) - {klineData.type}
                </Typography>
                <Tabs value={activeTab} onChange={handleTabChange} aria-label="indicator tabs">
                    <Tab label="价格" />
                    <Tab label="成交量" />
                    {selectedIndicators.includes('macd') && <Tab label="MACD" />}
                    {selectedIndicators.includes('rsi') && <Tab label="RSI" />}
                    {hasKdjIndicator && <Tab label="KDJ" />}
                </Tabs>
            </Box>
            
            {activeTab === 0 && renderPriceChart()}
            {activeTab === 1 && renderVolumeChart()}
            {selectedIndicators.includes('macd') && activeTab === 2 && renderMACDChart()}
            {selectedIndicators.includes('rsi') && activeTab === (selectedIndicators.includes('macd') ? 3 : 2) && renderRSIChart()}
            {hasKdjIndicator && activeTab === (2 + 
                (selectedIndicators.includes('macd') ? 1 : 0) + 
                (selectedIndicators.includes('rsi') ? 1 : 0)) && renderKDJChart()}
        </Paper>
    );
};

export default StockKlineChart;