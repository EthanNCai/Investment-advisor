import {StockContext} from './interfaces.tsx';
import {LineChart} from "@mui/x-charts/LineChart";
import {useContext} from "react";
import {Stack} from "@mui/material";
import {RadioSelector} from "./RadioSelector.tsx";
import Skeleton from '@mui/material/Skeleton';
export const KChart = () => {
    const {kChartInfo} = useContext(StockContext)!;
    function formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = (date.getMonth() + 1).toString().padStart(2, '0');
    const day = date.getDate().toString().padStart(2, '0');
    return `${year}-${month}-${day}`;
}

    return (
        <Stack>
        <RadioSelector title={"时间跨度"} option_labels={['一年','五年']} option_values={['nihao','buhao']}/>
        {(kChartInfo ? (
            <LineChart

                series={[
                    {
                    showMark:false,
                    data: kChartInfo?.ratio,

                        // area: true,
                    },
                 ]}

                 xAxis={[{
                     scaleType: 'time',
                     data: kChartInfo?.dates.map(dateString => new Date(dateString)) ,
                     valueFormatter: (value) => formatDate(value),
                     disableTicks: true,
                     colorMap: {
                     type: 'piecewise',
                     thresholds: [ new Date(2015, 4, 2),new Date(2023, 1, 1)],
                     colors: ['orange','green','orange'],

                 }

                 }]}
                width={500}
                height={300}

            />
        ) : (
            <Stack>

    <Skeleton/>
    <Skeleton/>
    <Skeleton/>
    <Skeleton/>
    <Skeleton/>
    <Skeleton/>
    <Skeleton/>

                </Stack>
))}

        </Stack>
    )

}