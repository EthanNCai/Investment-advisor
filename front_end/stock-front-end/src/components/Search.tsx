import React, {useEffect, useState} from "react";
import {Button, Container, Stack} from "@mui/material";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import {Box} from "@mui/system";
import {useContext } from 'react';
import {StockContext} from './interfaces.tsx';
import {RadioSelector} from "./RadioSelector.tsx";
interface Search_Result{
    code:string;
    name:string;
    type:string;
}
export default function SearchBar(){
    const { codeA, setCodeA, codeB, setCodeB} = useContext(StockContext)!;
    const [search_keyword, setSearch_keyword] = useState('');
    const [search_result, setSearchResult] = useState<Search_Result[]>([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch(`http://localhost:8000/search_stocks/${search_keyword}`);
                const data = await response.json();
                setSearchResult(data.result);
            } catch (error) {
                console.error('Error fetching data:', error);
            }
        };

        if (search_keyword !== '') {
            fetchData();
        } else {
            setSearchResult([]); // Clear search results if search keyword is empty
        }
    }, [search_keyword]);


    return (
        <Container>
            <Stack>

             <RadioSelector title={"时间跨度"} option_labels={['一年','五年']} option_values={['nihao','buhao']}/>
             <RadioSelector title={"时间跨度"} option_labels={['一年','五年']} option_values={['nihao','buhao']}/>

        <TextField
            onChange={(event) => {
                const newValue = event.target.value;
                if (newValue !== null) {
                    setSearch_keyword(newValue);
                }
            }}
            label="Search"
            variant="outlined"
        />

           </Stack>

            {search_result.map((item:Search_Result) => (
                <Box key={item.code}>
                    <Stack direction={'row'}>
                        <Typography>{item.code}-</Typography>
                        <Typography>{item.name}</Typography>
                        <Typography>-{item.type}</Typography>
                        <Button onClick={
                            () => {if (codeB === item.code) {
                                alert('A和B不能相同');
                            } else {
                                setCodeA(item.code);
                            }}}>
                            作为A
                        </Button>
                        <Button onClick={
                            () => {if (codeA === item.code) {
                                alert('A和B不能相同');
                            } else {
                                setCodeB(item.code);
                            }}}>
                            作为B
                        </Button>
                    </Stack>
                </Box>
        ))}
        <Typography>
            A:<br/>{codeA ? codeA: 'none'}
        </Typography>
        <Typography>
            B:<br/>{codeB ? codeB: 'none'}
        </Typography>


        </Container>
    )
}