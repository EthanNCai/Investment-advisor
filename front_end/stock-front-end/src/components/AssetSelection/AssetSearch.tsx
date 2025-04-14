import { useContext, useState, useEffect } from 'react';
import { Paper, InputBase, IconButton, Box, CircularProgress } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import { AssetSelectionContext } from './AssetInterfaces';

const AssetSearch = () => {
    const context = useContext(AssetSelectionContext);
    const [inputValue, setInputValue] = useState('');
    const [isSearching, setIsSearching] = useState(false);
    
    if (!context) {
        return null;
    }
    
    const { setSearchKeyword, isLoading } = context;
    
    // 使用useEffect实现防抖搜索
    useEffect(() => {
        // 取消上一次的计时器
        const debounceTimeout = setTimeout(() => {
            // 直接传递当前输入值，包括空字符串
            // 当为空字符串时，在AssetSelection组件中会加载所有资产
            setIsSearching(true);
            setSearchKeyword(inputValue.trim());
        }, 300); // 300ms防抖延迟
        
        return () => clearTimeout(debounceTimeout);
    }, [inputValue, setSearchKeyword]);
    
    // 当加载状态变化时更新搜索状态
    useEffect(() => {
        if (!isLoading && isSearching) {
            setIsSearching(false);
        }
    }, [isLoading, isSearching]);
    
    return (
        <Paper
            elevation={3}
            sx={{
                p: '2px 4px',
                display: 'flex',
                alignItems: 'center',
                width: '100%',
                my: 2
            }}
        >
            <InputBase
                sx={{ ml: 1, flex: 1 }}
                placeholder="输入股票代码或名称自动搜索"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
            />
            {isLoading ? (
                <CircularProgress size={20} sx={{ mr: 1, opacity: 0.7 }} />
            ) : (
                <IconButton 
                    type="button" 
                    sx={{ p: '10px', opacity: 0.7 }} 
                    aria-label="search"
                    disabled={true}
                >
                    <SearchIcon />
                </IconButton>
            )}
        </Paper>
    );
};

export default AssetSearch; 