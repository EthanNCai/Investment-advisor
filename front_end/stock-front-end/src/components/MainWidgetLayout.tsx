
import { makeStyles } from '@mui/styles';
import { Box } from '@mui/system';
import { Paper } from '@mui/material';

const useStyles = makeStyles(() => ({
  container: {
    display: 'flex',
    height: '100vh',
  },
  left: {
    flex: '3',
    height: '100%',
    backgroundColor: '#e0e0e0',
  },
  right: {
    flex: '1',
    height: '100%',
    backgroundColor: '#f5f5f5',
  },
}));

const MainWidgetLayout = () => {
  const classes = useStyles();

  return (
    <Box className={classes.container}>
      <Box className={classes.left}>
        <Paper>Left Content</Paper>
      </Box>
      <Box className={classes.right}>
        <Paper>Right Content</Paper>
      </Box>
    </Box>
  );
};

export default MainWidgetLayout;