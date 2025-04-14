import { useContext } from 'react';
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
    Skeleton
} from '@mui/material';
import { AssetSelectionContext, AssetInfo } from './AssetInterfaces';

const AssetList = () => {
    const context = useContext(AssetSelectionContext);
    
    if (!context) {
        return null;
    }
    
    const { assetList, selectedAsset, setSelectedAsset, isLoading } = context;
    
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
                            <ListItem disablePadding>
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
                                                    asset.type === '港股' ? 'primary' : 'success'
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