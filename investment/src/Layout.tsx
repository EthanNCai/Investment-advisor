import React from 'react';
import { AppBar, Toolbar, Typography, Drawer, List, ListItem, ListItemIcon, ListItemText } from '@mui/material';
import { Inbox as InboxIcon, Mail as MailIcon } from '@mui/icons-material';

const Layout = () => {
  return (
    <div style={{ display: 'flex' }}>
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h6">顶部菜单</Typography>
        </Toolbar>
      </AppBar>
      <Drawer variant="permanent" anchor="left">
        <List>
          <ListItem button>
            <ListItemIcon>
              <InboxIcon />
            </ListItemIcon>
            <ListItemText primary="菜单项1" />
          </ListItem>
          <ListItem button>
            <ListItemIcon>
              <MailIcon />
            </ListItemIcon>
            <ListItemText primary="菜单项2" />
          </ListItem>
        </List>
      </Drawer>
      <main style={{ marginLeft: '240px', marginTop: '64px' }}>
        {/* 主要内容区域 */}
        <h1>欢迎来到我的网站</h1>
        {/* 其他内容 */}
      </main>
    </div>
  );
};

export default Layout;