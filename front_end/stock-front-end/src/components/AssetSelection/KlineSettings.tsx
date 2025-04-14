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
    SelectChangeEvent
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
        selectedAsset
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
        </Paper>
    );
};

export default KlineSettings; 