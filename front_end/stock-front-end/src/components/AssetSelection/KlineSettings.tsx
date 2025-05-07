import { useContext } from 'react';
import { 
    Paper, 
    Typography, 
    ToggleButton, 
    ToggleButtonGroup, 
    Box,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Divider,
    FormGroup,
    FormControlLabel,
    Checkbox,
    SelectChangeEvent,
    Collapse
} from '@mui/material';
import { AssetSelectionContext, KlineType } from './AssetInterfaces';

const KlineSettings = () => {
    const context = useContext(AssetSelectionContext);
    
    if (!context) {
        return null;
    }
    
    const { 
        klineType, 
        setKlineType, 
        duration, 
        setDuration,
        selectedIndicators,
        setSelectedIndicators,
        selectedAsset,
        selectedMALines = ['ma5', 'ma10', 'ma20'],
        setSelectedMALines = (lines) => console.log('MA lines not supported', lines),
        selectedMACDLines = ['dif', 'dea', 'macd'],
        setSelectedMACDLines = (lines) => console.log('MACD lines not supported', lines),
        selectedRSILines = ['rsi6', 'rsi12', 'rsi24'],
        setSelectedRSILines = (lines) => console.log('RSI lines not supported', lines),
        selectedKDJLines = ['k', 'd', 'j'],
        setSelectedKDJLines = (lines) => console.log('KDJ lines not supported', lines)
    } = context;
    
    const handleKlineTypeChange = (_event: React.MouseEvent<HTMLElement>, newKlineType: KlineType | null) => {
        if (newKlineType !== null) {
            setKlineType(newKlineType);
        }
    };
    
    const handleDurationChange = (event: SelectChangeEvent) => {
        setDuration(event.target.value);
    };
    
    const handleIndicatorChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const indicator = event.target.name;
        
        if (event.target.checked) {
            setSelectedIndicators([...selectedIndicators, indicator]);
        } else {
            setSelectedIndicators(selectedIndicators.filter(item => item !== indicator));
        }
    };
    
    const handleMALineChange = (checkedValues: string[]) => {
        if (checkedValues.length === 0) {
            alert('请至少选择一条均线');
            return;
        }
        setSelectedMALines(checkedValues);
    };

    const handleMACDLineChange = (checkedValues: string[]) => {
        if (checkedValues.length === 0) {
            alert('请至少选择一条MACD线');
            return;
        }
        setSelectedMACDLines(checkedValues);
    };

    const handleRSILineChange = (checkedValues: string[]) => {
        if (checkedValues.length === 0) {
            alert('请至少选择一条RSI线');
            return;
        }
        setSelectedRSILines(checkedValues);
    };

    const handleKDJLineChange = (checkedValues: string[]) => {
        if (checkedValues.length === 0) {
            alert('请至少选择一条KDJ线');
            return;
        }
        setSelectedKDJLines(checkedValues);
    };
    
    if (!selectedAsset) {
        return null;
    }
    
    return (
        <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
            <Typography variant="h6" gutterBottom>
                图表设置
            </Typography>
            
            <Box sx={{ mb: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                    K线类型
                </Typography>
                <ToggleButtonGroup
                    color="primary"
                    value={klineType}
                    exclusive
                    onChange={handleKlineTypeChange}
                    aria-label="K线类型"
                    size="small"
                    fullWidth
                >
                    <ToggleButton value="realtime">实时</ToggleButton>
                    <ToggleButton value="daily">日K</ToggleButton>
                    <ToggleButton value="weekly">周K</ToggleButton>
                    <ToggleButton value="monthly">月K</ToggleButton>
                    <ToggleButton value="yearly">年K</ToggleButton>

                </ToggleButtonGroup>
            </Box>
            
            <Box sx={{ mb: 2 }}>
                <FormControl fullWidth size="small">
                    <InputLabel>时间范围</InputLabel>
                    <Select
                        value={duration}
                        label="时间范围"
                        onChange={handleDurationChange}
                    >
                        <MenuItem value="maximum">全部</MenuItem>
                        <MenuItem value="5y">5年</MenuItem>
                        <MenuItem value="2y">2年</MenuItem>
                        <MenuItem value="1y">1年</MenuItem>
                        <MenuItem value="1q">3个月</MenuItem>
                        <MenuItem value="1m">1个月</MenuItem>
                    </Select>
                </FormControl>
            </Box>
            
            <Divider sx={{ my: 2 }} />
            
            <Typography variant="subtitle2" gutterBottom>
                技术指标
            </Typography>
            
            <FormGroup>
                <Box sx={{ display: 'flex', flexWrap: 'wrap' }}>
                    <FormControlLabel
                        control={
                            <Checkbox 
                                checked={selectedIndicators.includes('ma')}
                                onChange={handleIndicatorChange}
                                name="ma"
                                size="small"
                            />
                        }
                        label="移动平均线"
                    />
                    <FormControlLabel
                        control={
                            <Checkbox 
                                checked={selectedIndicators.includes('macd')}
                                onChange={handleIndicatorChange}
                                name="macd"
                                size="small"
                            />
                        }
                        label="MACD"
                    />
                    <FormControlLabel
                        control={
                            <Checkbox 
                                checked={selectedIndicators.includes('rsi')}
                                onChange={handleIndicatorChange}
                                name="rsi"
                                size="small"
                            />
                        }
                        label="RSI"
                    />
                    <FormControlLabel
                        control={
                            <Checkbox 
                                checked={selectedIndicators.includes('kdj')}
                                onChange={handleIndicatorChange}
                                name="kdj"
                                size="small"
                            />
                        }
                        label="KDJ"
                    />
                </Box>
            </FormGroup>
            
            <Collapse in={selectedIndicators.includes('ma')}>
                <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" gutterBottom>
                        选择显示的均线:
                    </Typography>
                    <FormGroup row>
                        {['MA5', 'MA10', 'MA20', 'MA60'].map((ma, index) => (
                            <FormControlLabel
                                key={index}
                                control={
                                    <Checkbox 
                                        checked={selectedMALines.includes(ma.toLowerCase())}
                                        onChange={(e) => {
                                            const value = ma.toLowerCase();
                                            if (e.target.checked) {
                                                handleMALineChange([...selectedMALines, value]);
                                            } else {
                                                handleMALineChange(selectedMALines.filter(item => item !== value));
                                            }
                                        }}
                                        size="small"
                                    />
                                }
                                label={ma}
                            />
                        ))}
                    </FormGroup>
                </Box>
            </Collapse>
            
            <Collapse in={selectedIndicators.includes('macd')}>
                <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" gutterBottom>
                        选择显示的MACD指标:
                    </Typography>
                    <FormGroup row>
                        {[
                            { label: 'DIF', value: 'dif' },
                            { label: 'DEA', value: 'dea' },
                            { label: 'MACD柱', value: 'macd' }
                        ].map((item, index) => (
                            <FormControlLabel
                                key={index}
                                control={
                                    <Checkbox 
                                        checked={selectedMACDLines.includes(item.value)}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                handleMACDLineChange([...selectedMACDLines, item.value]);
                                            } else {
                                                handleMACDLineChange(selectedMACDLines.filter(line => line !== item.value));
                                            }
                                        }}
                                        size="small"
                                    />
                                }
                                label={item.label}
                            />
                        ))}
                    </FormGroup>
                </Box>
            </Collapse>
            
            <Collapse in={selectedIndicators.includes('rsi')}>
                <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" gutterBottom>
                        选择显示的RSI指标:
                    </Typography>
                    <FormGroup row>
                        {[
                            { label: 'RSI6', value: 'rsi6' },
                            { label: 'RSI12', value: 'rsi12' },
                            { label: 'RSI24', value: 'rsi24' }
                        ].map((item, index) => (
                            <FormControlLabel
                                key={index}
                                control={
                                    <Checkbox 
                                        checked={selectedRSILines.includes(item.value)}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                handleRSILineChange([...selectedRSILines, item.value]);
                                            } else {
                                                handleRSILineChange(selectedRSILines.filter(line => line !== item.value));
                                            }
                                        }}
                                        size="small"
                                    />
                                }
                                label={item.label}
                            />
                        ))}
                    </FormGroup>
                </Box>
            </Collapse>
            
            <Collapse in={selectedIndicators.includes('kdj')}>
                <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" gutterBottom>
                        选择显示的KDJ指标:
                    </Typography>
                    <FormGroup row>
                        {[
                            { label: 'K线', value: 'k' },
                            { label: 'D线', value: 'd' },
                            { label: 'J线', value: 'j' }
                        ].map((item, index) => (
                            <FormControlLabel
                                key={index}
                                control={
                                    <Checkbox 
                                        checked={selectedKDJLines.includes(item.value)}
                                        onChange={(e) => {
                                            if (e.target.checked) {
                                                handleKDJLineChange([...selectedKDJLines, item.value]);
                                            } else {
                                                handleKDJLineChange(selectedKDJLines.filter(line => line !== item.value));
                                            }
                                        }}
                                        size="small"
                                    />
                                }
                                label={item.label}
                            />
                        ))}
                    </FormGroup>
                </Box>
            </Collapse>
        </Paper>
    );
};

export default KlineSettings; 