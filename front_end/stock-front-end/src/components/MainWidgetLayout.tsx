import {Box} from '@mui/system';

import {Paper} from "@mui/material";


const MainWidgetLayout = () => {



    return (
        <Box>
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: {
                        'xs':'column',
                        'sm': 'row',
                        'lg': 'row'

                    }
                }}
            >
                <Box sx={{flex: 2}}>
                    <Paper>左侧内容</Paper>

                </Box>

                <Box sx={{flex: 1}}>
                    <Paper>R侧内容</Paper>
                </Box>

            </Box>
        </Box>
    );
};

export default MainWidgetLayout;