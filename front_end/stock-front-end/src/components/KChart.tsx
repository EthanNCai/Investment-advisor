import { StockContext } from "./interfaces.tsx";
import { LineChart } from "@mui/x-charts/LineChart";
import  { useContext } from "react";
import { Stack, Box } from "@mui/material";
import { RadioSelector } from "./RadioSelector.tsx";
import Skeleton from "@mui/material/Skeleton";
import { ChartsReferenceLine } from '@mui/x-charts/ChartsReferenceLine';
import {CheckSelector} from "./CheckSelector.tsx";
import Typography from "@mui/material/Typography";


export const KChart = () => {
  const { kChartInfo ,duration,setDuration,threshold_arg,showRatio,showDelta,setShowDelta,setShowRatio} = useContext(StockContext)!;
  function formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, "0");
    const day = date.getDate().toString().padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  return (
    <Stack spacing={0} >

      <Box justifyContent="center"display="flex"
          gap={0}
           alignItems="center">

        <RadioSelector
            title={"时间跨度"}
            option_labels={["最大", "五年", "两年", "一年", "一季"]}
            available_options={["maximum", "5y","2y", "1y","1q"]}
            current_option={duration}
            set_current_option={setDuration}
        />

      </Box>
      <Box justifyContent="center"display="flex"

           alignItems="center">


          <Stack direction={'row'}><CheckSelector title={'显示比例图'} value={showRatio} set_value={setShowRatio}/>
            <CheckSelector title={'显示差异图'} value={showDelta} set_value={setShowDelta}/></Stack>
      </Box>

      {kChartInfo ? (
          <Stack direction={{
            xs: "column",
            sm: "column",
            md: "row",
            lg: "row",
          }} spacing={0}>

            { showRatio && (<LineChart

                skipAnimation = {false}
                series={[
                  {
                    showMark: false,
                    data: kChartInfo ? kChartInfo?.ratio : [] , label:'A和B的收盘价比',id:"ratio",
                    color:"#2196f3"


                    // area: true,
                  },{
                    showMark: false,
                    curve: "natural",
                    data: kChartInfo?.fitting_line, label:'多项式拟合线',id:"fitting",
                    color:'red'

                    // area: true,
                  },
                ]}
                xAxis={[
                  {
                    scaleType: "time",
                    data: kChartInfo?.dates.map((dateString) => new Date(dateString)),
                    valueFormatter: (value) => formatDate(value),
                    disableTicks: true,
                    // colorMap: {
                    //   type: "piecewise",
                    //   thresholds: kChartInfo.outlier_date_splitters.map((dateString) => new Date(dateString)),
                    //   colors: kChartInfo.colors,
                    // },
                  },
                ]}
                height={300}
                sx={{

                  '.MuiLineElement-series-ratio': {
                    strokeWidth: duration === 'maximum' || duration === '5y' || duration === '2y' ? 1 : 2
                  }
                  ,
                  '.MuiLineElement-series-fitting': {
                    strokeDasharray: '5 5',
                    strokeWidth:3


                  },
                  margin:0

                }}
            >
              {/*{kChartInfo?.outlier_date_splitters.map((date)=>(<ChartsReferenceLine x={new Date(date)} lineStyle={{ stroke: 'red' }} />))}*/}

            </LineChart>)}
            {showDelta &&(<LineChart

                skipAnimation = {true}
                series={[
                  {
                    showMark: false,
                    data: kChartInfo?.delta, label:'收盘价比与拟合线的差值',id:"ratio",
                    area: true
                    ,color:'#bbdefb'

                  },
                ]}
                xAxis={[
                  {
                    scaleType: "time",
                    data: kChartInfo?.dates.map((dateString) => new Date(dateString)),
                    valueFormatter: (value) => formatDate(value),
                    disableTicks: true,

                  },
                ]}
                yAxis={[
                  {
                    colorMap:
                        {
                          type: 'piecewise',
                          thresholds: [- threshold_arg * kChartInfo?.threshold,  threshold_arg * kChartInfo?.threshold],
                          colors: ['#C96868','#bbdefb', '#15B392'],
                        }
                  },
                ]}
                height={300}
            >
              <ChartsReferenceLine y={0} lineStyle={{ stroke: 'red' ,strokeWidth:3,strokeDasharray: '5 5',}} />

            </LineChart>)}

            </Stack>
      ) : (

          <Stack  spacing={1}>

          <Skeleton height={300}/>
          </Stack>

      )}
    </Stack>
  );
};
