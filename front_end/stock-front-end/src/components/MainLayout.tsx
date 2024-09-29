import {Box} from '@mui/system';
import {Paper} from "@mui/material";
import SearchBar from "./Search.tsx";
import React, {useEffect, useState} from "react";
import {StockContext, KChartInfo, UserOptions} from './interfaces.tsx'

import {KChart} from "./KChart.tsx";
const MainLayout = () => {

    const [codeA,setCodeA] = useState('');
    const [codeB,setCodeB] = useState('');
    const [kChartInfo,setKChartInfo] = useState<KChartInfo|undefined>(undefined);
    const [userOptions,SetUserOptions] = useState<UserOptions|undefined>(undefined);



    useEffect(() => {
    const fetchKChartInfo = (codeA:string, codeB:string,userOptions:UserOptions, set_stock:React.Dispatch<React.SetStateAction<KChartInfo | undefined>>) => {
        if (codeA === '') {
            return;
        }
        return fetch(`http://localhost:8000/fetch_stock_info/${codeA}`)
            .then(response => {
                if (!response.ok) {throw new Error('Network ERROR');}
                return response.json();
            })
            .then(data => {set_stock(data);})
            .catch(error => {console.error('Error fetching data:', error);});
    };
    if (codeA !== '') {
        fetchKChartInfo(codeA,codeB, userOptions!,setKChartInfo);
        console.log(kChartInfo);
    }
}, [codeA,codeB]);



    return (
        <StockContext.Provider value={{ codeA: codeA, setCodeA, codeB: codeB, setCodeB,kChartInfo:kChartInfo,userOptions:userOptions!}}>
        <Box>
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: {
                        'xs': 'column',
                        'sm': 'row',
                        'lg': 'row'

                    }
                }}
            >
                <Box sx={{flex: 2}}>
                    <Paper>
                        <KChart/>
                    </Paper>
                </Box>
                <Box sx={{flex: 1}}>
                    <Paper>
                        <SearchBar />
                    </Paper>
                </Box>

            </Box>
        </Box>

    </StockContext.Provider>
    );
};

export default MainLayout;