import React from "react";

import {Chip,Stack} from "@mui/material";

import {StockInfo} from "./interfaces.tsx";
import Typography from "@mui/material/Typography";

interface StockDivisionPorps {
    stockA: StockInfo;
    stockB: StockInfo;
}

export const StockDivision: React.FC<StockDivisionPorps> = ({stockA, stockB}) => {
    return (
        // <Paper elevation={3} sx={{ padding: "10px"}}>
        <Stack direction={'row'} alignItems={'center'} justifyContent={'space-between'}>
            <Typography fontSize={25}> 当前对象: </Typography>
            <Stack direction="column" alignItems={'center'} spacing={1} >
                <Stack direction={'row'}  alignItems={'center'} spacing ={1}> <Typography fontSize={15}>A :</Typography><Chip label={stockA.type ?  stockA.name:"尚未选择证券A"} /><Chip label={stockA.type? stockA.code:"-"} /></Stack>
                <Stack direction={'row'}  alignItems={'center'} spacing ={1}> <Typography fontSize={15}>B :</Typography><Chip label={stockB.type? stockB.name:"尚未选择证券B"} /><Chip label={stockB.type? stockB.code:"-"} /></Stack>

            </Stack>
        </Stack>
        // </Paper>
    );
};