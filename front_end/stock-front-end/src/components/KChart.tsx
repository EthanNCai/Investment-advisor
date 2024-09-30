import { StockContext } from "./interfaces.tsx";
import { LineChart ,lineElementClasses,
  markElementClasses,} from "@mui/x-charts/LineChart";
import React, { useContext } from "react";
import { Stack } from "@mui/material";
import { RadioSelector } from "./RadioSelector.tsx";
import Skeleton from "@mui/material/Skeleton";
import { ChartsReferenceLine } from '@mui/x-charts/ChartsReferenceLine';



export const KChart = () => {
  const { kChartInfo ,duration,setDuration} = useContext(StockContext)!;
  function formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, "0");
    const day = date.getDate().toString().padStart(2, "0");
    return `${year}-${month}-${day}`;
  }

  return (
    <Stack>
      <RadioSelector
          title={"时间跨度"}
          option_labels={["最大", "五年", "两年", "一年", "一季", "一月"]}
          available_options={["maximum", "5y","2y", "1y","1q", "1m"]}
          current_option={duration}
          set_current_option={setDuration}
      />
      {kChartInfo ? (
          <Stack direction="row" spacing={0}>
        <LineChart
            skipAnimation = {false}
          series={[
            {
              showMark: false,
              data: kChartInfo?.ratio, label:'ratio',id:"ratio"


              // area: true,
            },{
              showMark: false,
              data: kChartInfo?.fitting_line, label:'fitting',id:"fitting"

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
              [`.${lineElementClasses.root}, .${markElementClasses.root}`]: {
                strokeWidth: 1,
              },
              '.MuiLineElement-series-fitting': {
                strokeDasharray: '5 5',
              },


            }}
        >
          {/*{kChartInfo?.outlier_date_splitters.map((date)=>(<ChartsReferenceLine x={new Date(date)} lineStyle={{ stroke: 'red' }} />))}*/}

        </LineChart>
            <LineChart
                skipAnimation = {true}
                series={[
                  {
                    showMark: false,
                    data: kChartInfo?.delta, label:'ratio',id:"ratio",
                    area: true
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
                        thresholds: [- kChartInfo?.thres,  kChartInfo?.thres],
                        colors: ['red','#E2E2E2', 'green'],
                      }
                  },
                ]}
                height={300}
            >
              {kChartInfo?.outlier_date_splitters.map(()=>(<ChartsReferenceLine y={0} lineStyle={{ stroke: 'red' }} />))}

            </LineChart>
            </Stack>
      ) : (
        <Stack>
          <Skeleton />
          <Skeleton />
          <Skeleton />
          <Skeleton />
          <Skeleton />
          <Skeleton />
          <Skeleton />
        </Stack>
      )}
    </Stack>
  );
};
