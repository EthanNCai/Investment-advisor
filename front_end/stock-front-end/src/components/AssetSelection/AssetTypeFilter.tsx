import { useContext } from 'react';
import { Tabs, Tab, Box, Paper } from '@mui/material';
import { AssetSelectionContext } from './AssetInterfaces';

const AssetTypeFilter = () => {
    const context = useContext(AssetSelectionContext);
    
    if (!context) {
        return null;
    }
    
    const { selectedType, setSelectedType } = context;
    
    const handleChange = (_event: React.SyntheticEvent, newValue: string) => {
        setSelectedType(newValue);
    };
    
    return (
        <Paper elevation={2}>
            <Box sx={{ width: '100%', bgcolor: 'background.paper' }}>
                <Tabs
                    value={selectedType}
                    onChange={handleChange}
                    centered
                    variant="standard"
                    textColor="primary"
                    indicatorColor="primary"
                    sx={{
                        '& .MuiTabs-indicator': {
                            transition: 'all 0.3s ease',
                            height: 3
                        },
                        '& .MuiTab-root': {
                            minWidth: 0,
                            padding: '12px 16px',
                            '&:hover': {
                                opacity: 1
                            }
                        }
                    }}
                >
                    <Tab label="全部" value="全部" />
                    <Tab label="A股" value="A股" />
                    <Tab label="港股" value="港股" />
                    <Tab label="美股" value="美股" />
                    <Tab label="黄金" value="黄金" />
                </Tabs>
            </Box>
        </Paper>
    );
};

export default AssetTypeFilter; 