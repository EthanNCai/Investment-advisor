import { useState, useEffect } from 'react';
import { Container, Grid, Typography, Box, Alert } from '@mui/material';
import { 
    AssetSelectionContext, 
    AssetInfo, 
    StockKlineData,
    KlineType
} from './AssetInterfaces';
import AssetTypeFilter from './AssetTypeFilter';
import AssetSearch from './AssetSearch';
import AssetList from './AssetList';
import KlineSettings from './KlineSettings';
import StockKlineChart from './StockKlineChart';

const AssetSelection = () => {
    // 状态管理
    const [selectedAsset, setSelectedAsset] = useState<AssetInfo | null>(null);
    const [assetList, setAssetList] = useState<AssetInfo[]>([]);
    const [selectedType, setSelectedType] = useState<string>('全部');
    const [searchKeyword, setSearchKeyword] = useState<string>('');
    const [klineData, setKlineData] = useState<StockKlineData | null>(null);
    const [klineType, setKlineType] = useState<KlineType>('daily');
    const [duration, setDuration] = useState<string>('1y');
    const [selectedIndicators, setSelectedIndicators] = useState<string[]>(['ma']);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    
    // 获取所有资产或按类型筛选
    const fetchAllAssets = async () => {
        setIsLoading(true);
        setError(null);
        
        try {
            let url = '';
            
            if (selectedType === '全部') {
                url = 'http://localhost:8000/get_all_assets/';
            } else {
                url = `http://localhost:8000/get_assets_by_type/${selectedType}`;
            }
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error('获取资产列表失败');
            }
            
            const data = await response.json();
            setAssetList(data.assets);
        } catch (err) {
            console.error('获取资产列表错误:', err);
            setError('获取资产列表失败，请稍后重试');
            setAssetList([]);
        } finally {
            setIsLoading(false);
        }
    };

    // 初始加载和类型变更时获取资产列表
    useEffect(() => {
        fetchAllAssets();
    }, [selectedType]);
    
    // 搜索资产
    useEffect(() => {
        // 如果关键词为空，显示所有资产
        if (searchKeyword.trim() === '') {
            fetchAllAssets();
            return;
        }
        
        setIsLoading(true);
        setError(null);
        
        const searchAssets = async () => {
            try {
                // 先获取搜索结果
                const response = await fetch(`http://localhost:8000/search_top_assets/${searchKeyword}`);
                
                if (!response.ok) {
                    throw new Error('搜索资产失败');
                }
                
                const data = await response.json();
                
                // 如果选择了特定类型（非"全部"），则过滤结果
                if (selectedType !== '全部') {
                    const filteredAssets = data.assets.filter((asset: AssetInfo) => asset.type === selectedType);
                    setAssetList(filteredAssets);
                } else {
                    setAssetList(data.assets);
                }
            } catch (err) {
                console.error('搜索资产错误:', err);
                setError('搜索资产失败，请稍后重试');
            } finally {
                setIsLoading(false);
            }
        };
        
        searchAssets();
    }, [searchKeyword, selectedType]);
    
    // 获取单只股票K线数据
    useEffect(() => {
        if (!selectedAsset) {
            setKlineData(null);
            return;
        }
        
        setIsLoading(true);
        setError(null);
        
        const fetchKlineData = async () => {
            try {
                const response = await fetch('http://localhost:8000/get_stock_kline/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        code: selectedAsset.code,
                        kline_type: klineType,
                        duration: duration
                    }),
                });
                
                if (!response.ok) {
                    throw new Error('获取K线数据失败');
                }
                
                const data = await response.json();
                setKlineData(data);
            } catch (err) {
                console.error('获取K线数据错误:', err);
                setError('获取K线数据失败，请稍后重试');
                setKlineData(null);
            } finally {
                setIsLoading(false);
            }
        };
        
        fetchKlineData();
    }, [selectedAsset, klineType, duration]);
    
    return (
        <AssetSelectionContext.Provider
            value={{
                selectedAsset,
                setSelectedAsset,
                assetList,
                setAssetList,
                selectedType,
                setSelectedType,
                searchKeyword,
                setSearchKeyword,
                klineData,
                setKlineData,
                klineType,
                setKlineType,
                duration,
                setDuration,
                selectedIndicators,
                setSelectedIndicators,
                isLoading,
                setIsLoading
            }}
        >
            <Container maxWidth="lg" sx={{ mt: 2 }}>
                <Typography variant="h4" gutterBottom>
                    资产选择
                </Typography>
                
                {error && (
                    <Alert severity="error" sx={{ mb: 2 }}>
                        {error}
                    </Alert>
                )}
                
                <Grid container spacing={3}>
                    {/* 左侧：资产选择 */}
                    <Grid item xs={12} md={4}>
                        <Box sx={{ mb: 2 }}>
                            <AssetTypeFilter />
                        </Box>
                        <AssetSearch />
                        <AssetList />
                    </Grid>
                    
                    {/* 右侧：K线图 */}
                    <Grid item xs={12} md={8}>
                        {selectedAsset && (
                            <KlineSettings />
                        )}
                        <StockKlineChart />
                    </Grid>
                </Grid>
            </Container>
        </AssetSelectionContext.Provider>
    );
};

export default AssetSelection; 