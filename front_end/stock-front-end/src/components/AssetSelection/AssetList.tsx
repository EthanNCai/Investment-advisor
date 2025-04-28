import { useContext, useState, useEffect } from 'react';
import { 
    Paper, 
    List, 
    ListItem, 
    ListItemButton, 
    ListItemText, 
    Typography, 
    Divider,
    Box,
    Chip,
    Skeleton,
    IconButton,
    Tooltip
} from '@mui/material';
import { AssetSelectionContext, AssetInfo } from './AssetInterfaces';
import StarIcon from '@mui/icons-material/Star';
import StarBorderIcon from '@mui/icons-material/StarBorder';
import { message } from 'antd';
import { useLocalStorage } from '../../LocalStorageContext';

const AssetList = () => {
    const context = useContext(AssetSelectionContext);
    const [isLoggedIn] = useLocalStorage<boolean>('isLoggedIn', false);
    const [favorites, setFavorites] = useState<Record<string, boolean>>({});
    const [loadingFavorites, setLoadingFavorites] = useState<boolean>(false);
    
    if (!context) {
        return null;
    }
    
    const { assetList, selectedAsset, setSelectedAsset, isLoading } = context;
    
    // 获取用户收藏列表
    useEffect(() => {
        if (isLoggedIn) {
            fetchFavorites();
        }
    }, [isLoggedIn]);
    
    // 获取用户收藏列表
    const fetchFavorites = async () => {
        if (!isLoggedIn) return;
        
        setLoadingFavorites(true);
        try {
            const response = await fetch('http://localhost:8000/api/favorites', {
                credentials: 'include'
            });
            
            if (response.ok) {
                const data = await response.json();
                const favMap: Record<string, boolean> = {};
                
                data.favorites.forEach((fav: any) => {
                    favMap[fav.stock_code] = true;
                });
                
                setFavorites(favMap);
            }
        } catch (error) {
            console.error('获取收藏列表失败:', error);
        } finally {
            setLoadingFavorites(false);
        }
    };
    
    // 检查资产是否已收藏
    const isFavorite = (stockCode: string): boolean => {
        return favorites[stockCode] || false;
    };
    
    // 切换收藏状态
    const toggleFavorite = async (asset: AssetInfo, e: React.MouseEvent) => {
        e.stopPropagation(); // 防止触发列表项点击事件
        
        if (!isLoggedIn) {
            message.warning('请先登录后再收藏资产');
            return;
        }
        
        const stockCode = asset.code;
        const isFav = isFavorite(stockCode);
        
        try {
            if (isFav) {
                // 取消收藏
                const response = await fetch(`http://localhost:8000/api/favorites/${stockCode}`, {
                    method: 'DELETE',
                    credentials: 'include'
                });
                
                if (response.ok) {
                    setFavorites(prev => {
                        const newFavorites = { ...prev };
                        delete newFavorites[stockCode];
                        return newFavorites;
                    });
                    message.success('已取消收藏');
                } else {
                    message.error('取消收藏失败');
                }
            } else {
                // 添加收藏
                const response = await fetch('http://localhost:8000/api/favorites', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        stock_code: asset.code,
                        stock_name: asset.name,
                        stock_type: asset.type
                    }),
                    credentials: 'include'
                });
                
                if (response.ok) {
                    setFavorites(prev => ({
                        ...prev,
                        [stockCode]: true
                    }));
                    message.success('已添加到收藏');
                } else {
                    message.error('添加收藏失败');
                }
            }
        } catch (error) {
            console.error('操作收藏失败:', error);
            message.error('操作失败，请稍后重试');
        }
    };
    
    const handleAssetClick = (asset: AssetInfo) => {
        setSelectedAsset(asset);
    };
    
    if (isLoading) {
        return (
            <Paper elevation={3} sx={{ p: 2, height: '400px', overflow: 'auto' }}>
                {[...Array(10)].map((_, index) => (
                    <Box key={index} sx={{ mb: 2 }}>
                        <Skeleton animation="wave" height={40} />
                        <Skeleton animation="wave" height={20} width="60%" />
                        <Divider sx={{ my: 1 }} />
                    </Box>
                ))}
            </Paper>
        );
    }
    
    return (
        <Paper elevation={3} sx={{ height: '400px', overflow: 'auto' }}>
            {assetList.length === 0 ? (
                <Box sx={{ p: 3, textAlign: 'center' }}>
                    <Typography variant="body1" color="text.secondary">
                        没有找到匹配的资产
                    </Typography>
                </Box>
            ) : (
                <List sx={{ width: '100%', p: 0 }}>
                    {assetList.map((asset, index) => (
                        <Box key={asset.code}>
                            <ListItem 
                                disablePadding
                                secondaryAction={
                                    <Tooltip title={isFavorite(asset.code) ? "取消收藏" : "添加到收藏"}>
                                        <IconButton 
                                            edge="end" 
                                            aria-label="favorite"
                                            onClick={(e) => toggleFavorite(asset, e)}
                                            disabled={loadingFavorites}
                                        >
                                            {isFavorite(asset.code) ? 
                                                <StarIcon color="warning" /> : 
                                                <StarBorderIcon />
                                            }
                                        </IconButton>
                                    </Tooltip>
                                }
                            >
                                <ListItemButton 
                                    selected={selectedAsset?.code === asset.code}
                                    onClick={() => handleAssetClick(asset)}
                                >
                                    <Box sx={{ width: '100%' }}>
                                        <Box sx={{ 
                                            display: 'flex', 
                                            justifyContent: 'space-between',
                                            alignItems: 'center',
                                            mb: 0.5
                                        }}>
                                            <Typography variant="subtitle1" component="span">
                                                {asset.name}
                                            </Typography>
                                            <Chip 
                                                label={asset.type} 
                                                size="small" 
                                                color={
                                                    asset.type === 'A股' ? 'error' : 
                                                    asset.type === '港股' ? 'primary' : 
                                                    asset.type === '黄金' ? 'warning' : 'success'
                                                }
                                            />
                                        </Box>
                                        <Typography variant="body2" color="text.secondary">
                                            {asset.code}
                                        </Typography>
                                    </Box>
                                </ListItemButton>
                            </ListItem>
                            {index < assetList.length - 1 && <Divider />}
                        </Box>
                    ))}
                </List>
            )}
        </Paper>
    );
};

export default AssetList; 