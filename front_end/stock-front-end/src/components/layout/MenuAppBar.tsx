import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import { Button, Avatar, Drawer, List, ListItem, ListItemButton, ListItemIcon, ListItemText, Divider } from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import { useLocalStorage } from '../../LocalStorageContext';
import DashboardIcon from '@mui/icons-material/Dashboard';
import SearchIcon from '@mui/icons-material/Search';
import BarChartIcon from '@mui/icons-material/BarChart';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import LoginIcon from '@mui/icons-material/Login';
import LogoutIcon from '@mui/icons-material/Logout';
import AccountCircleIcon from '@mui/icons-material/AccountCircle';
import { message } from 'antd';

export default function MenuAppBar() {
    const navigate = useNavigate();
    const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
    const [isLoggedIn, setIsLoggedIn] = useLocalStorage<boolean>('isLoggedIn', false);
    const [currentUser, setCurrentUser] = useLocalStorage<any>('currentUser', null);

    const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    const handleLogout = async () => {
        try {
            const response = await fetch('http://localhost:8000/api/logout', {
                method: 'POST',
                credentials: 'include',
            });

            if (response.ok) {
                // 清除本地存储的用户信息
                setIsLoggedIn(false);
                setCurrentUser(null);
                
                // 关闭菜单
                setAnchorEl(null);
                
                // 显示成功消息
                message.success('已成功退出登录');
                
                // 跳转到登录页面
                navigate('/login');
            } else {
                message.error('退出登录失败，请重试');
            }
        } catch (error) {
            console.error('退出登录请求失败:', error);
            message.error('退出登录请求失败，请稍后重试');
        }
    };

    // 主要菜单项列表
    const menuItems = [
        { 
            text: '仪表盘', 
            icon: <DashboardIcon />, 
            path: '/dashboard',
            requireAuth: true
        },
        { 
            text: '资产选择', 
            icon: <SearchIcon />, 
            path: '/assets',
            requireAuth: true
        },
        { 
            text: '比值分析', 
            icon: <BarChartIcon />, 
            path: '/ratio-analysis',
            requireAuth: true
        },
        { 
            text: '投资信号', 
            icon: <SignalCellularAltIcon />, 
            path: '/investment-signals',
            requireAuth: true
        }
    ];

    return (
        <Box sx={{ flexGrow: 1 }}>
            <AppBar position="static">
                <Toolbar>
                    {/* 系统名称不可点击 */}
                    <Typography variant="h6" sx={{ 
                        flexGrow: 0, 
                        color: 'inherit',
                        marginRight: 4
                    }}>
                        资产价值比较分析系统
                    </Typography>
                    
                    {/* 水平导航菜单 */}
                    <Box sx={{ flexGrow: 1, display: 'flex' }}>
                        {menuItems.map((item) => (
                            // 控制菜单项的显示，未登录用户不显示需要授权的菜单
                            (!item.requireAuth || isLoggedIn) && (
                                <Button
                                    key={item.text}
                                    component={Link}
                                    to={item.path}
                                    color="inherit"
                                    startIcon={item.icon}
                                    sx={{ mr: 2 }}
                                >
                                    {item.text}
                                </Button>
                            )
                        ))}
                    </Box>

                    {isLoggedIn ? (
                        <div>
                            <IconButton
                                size="large"
                                aria-label="account of current user"
                                aria-controls="menu-appbar"
                                aria-haspopup="true"
                                onClick={handleMenu}
                                color="inherit"
                            >
                                <Avatar sx={{ bgcolor: '#1890ff' }}>
                                    {currentUser?.username ? currentUser.username.charAt(0).toUpperCase() : 'U'}
                                </Avatar>
                            </IconButton>
                            <Menu
                                id="menu-appbar"
                                anchorEl={anchorEl}
                                anchorOrigin={{
                                    vertical: 'bottom',
                                    horizontal: 'right',
                                }}
                                keepMounted
                                transformOrigin={{
                                    vertical: 'top',
                                    horizontal: 'right',
                                }}
                                open={Boolean(anchorEl)}
                                onClose={handleClose}
                            >
                                <MenuItem disabled>
                                    <Typography variant="body2">
                                        {currentUser?.username || '用户'}
                                    </Typography>
                                </MenuItem>
                                <Divider />
                                <MenuItem onClick={handleLogout}>
                                    <ListItemIcon>
                                        <LogoutIcon fontSize="small" />
                                    </ListItemIcon>
                                    退出登录
                                </MenuItem>
                            </Menu>
                        </div>
                    ) : (
                        <Button
                            color="inherit"
                            startIcon={<LoginIcon />}
                            component={Link}
                            to="/login"
                        >
                            登录
                        </Button>
                    )}
                </Toolbar>
            </AppBar>
        </Box>
    );
}