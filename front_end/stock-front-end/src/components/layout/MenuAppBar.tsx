import * as React from 'react';
import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import MenuIcon from '@mui/icons-material/Menu';
import AccountCircle from '@mui/icons-material/AccountCircle';
import MenuItem from '@mui/material/MenuItem';
import Menu from '@mui/material/Menu';
import Button from '@mui/material/Button';
import {Link as RouterLink} from 'react-router-dom';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

export default function MenuAppBar() {
    const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);

    const handleMenu = (event: React.MouseEvent<HTMLElement>) => {
        setAnchorEl(event.currentTarget);
    };

    const handleClose = () => {
        setAnchorEl(null);
    };

    return (
        <Box sx={{flexGrow: 1}}>
            <AppBar position="static">
                <Toolbar>
                    <IconButton
                        size="large"
                        edge="start"
                        color="inherit"
                        aria-label="menu"
                        sx={{mr: 2}}
                    >
                        <MenuIcon/>
                    </IconButton>
                    <Typography variant="h6" component="div" sx={{flexGrow: 1}}>
                        资产价值比较分析
                    </Typography>

                    <Button
                        color="inherit"
                        component={RouterLink}
                        to="/"
                        sx={{mx: 1}}
                    >
                        首页
                    </Button>
                    <Button
                        color="inherit"
                        component={RouterLink}
                        to="/assets"
                        sx={{mx: 1}}
                    >
                        资产选择
                    </Button>
                    <Button
                        color="inherit"
                        component={RouterLink}
                        to="/ratio-analysis"
                        sx={{mx: 1}}
                        startIcon={<TrendingUpIcon/>}
                    >
                        比值分析
                    </Button>
                    <Button
                        color="inherit"
                        component={RouterLink}
                        to="/investment-signals"
                        sx={{mx: 1}}
                        startIcon={<TrendingUpIcon/>}
                    >
                        投资信号
                    </Button>
                    <div>
                        <IconButton
                            size="large"
                            aria-label="account of current user"
                            aria-controls="menu-appbar"
                            aria-haspopup="true"
                            onClick={handleMenu}
                            color="inherit"
                        >
                            <AccountCircle/>
                        </IconButton>
                        <Menu
                            id="menu-appbar"
                            anchorEl={anchorEl}
                            anchorOrigin={{
                                vertical: 'top',
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
                            <MenuItem onClick={handleClose}>我的账户</MenuItem>
                            <MenuItem onClick={handleClose}>登出</MenuItem>
                        </Menu>
                    </div>
                </Toolbar>
            </AppBar>
        </Box>
    );
}