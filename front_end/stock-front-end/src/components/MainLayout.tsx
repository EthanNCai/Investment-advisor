import { Box } from "@mui/system";
import { Paper } from "@mui/material";
import SearchBar from "./Search.tsx";
import  { useEffect, useState } from "react";
import {StockContext, KChartInfo, UserOptionInfo, StockInfo} from "./interfaces.tsx";

import { KChart } from "./KChart.tsx";
const MainLayout = () => {

  const [degree, setDegree] = useState(2);
  const [duration, setDuration] = useState('1y');
  const [threshold_arg, setThreshold_arg] = useState(1.5);
  const [kChartInfo, setKChartInfo] = useState<KChartInfo | undefined>(
    undefined
  );
    const [stockInfoA, setStockInfoA] = useState<StockInfo>({name:"",code:"",type:""});
    const [stockInfoB, setStockInfoB] = useState<StockInfo>({name:"",code:"",type:""});

  useEffect(() => {
    const fetchKChartInfo = () => {
      if (stockInfoA.code === "" || stockInfoB.code === "") {
        return;
      }
      const user_option_info: UserOptionInfo = {
          code_a:stockInfoA.code,
          code_b:stockInfoB.code,
          duration:duration,
          // threshold_arg:threshold,
          degree:degree
      }
      return fetch(`http://localhost:8000/get_k_chart_info/`,{
            method: "POST",
                headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(user_option_info),
        })

        .then((response) => {
          if (!response.ok) {
            throw new Error("Network ERROR");
          }
          return response.json();
        })
        .then((data) => {
          setKChartInfo(data);

        })
        .catch((error) => {
          console.error("Error fetching data:", error);
        });
    };
    fetchKChartInfo();

  }, [stockInfoA, stockInfoB, duration,degree]);

  return (
    <StockContext.Provider
      value={{

        kChartInfo: kChartInfo,
        degree:degree,
        setDegree:setDegree,
        duration:duration,
        setDuration:setDuration,
        threshold_arg:threshold_arg,
        setThreshold_arg:setThreshold_arg,
          stockInfoA:stockInfoA,
          setStockInfoA:setStockInfoA,
          stockInfoB:stockInfoB,
          setStockInfoB:setStockInfoB ,
      }}
    >
      <Box>
        <Box
          sx={{
            display: "flex",
            flexDirection: {
              xs: "column",
              sm: "row",
              lg: "row",
            },
              gap: 2,
          }}
        >
          <Box sx={{ flex: 2 }}>
            <Paper elevation={3}>
              <KChart />
            </Paper>
          </Box>
          <Box sx={{ flex: 1 }}>
            <Paper elevation={3}>
              <SearchBar />
            </Paper>
          </Box>
        </Box>
      </Box>
    </StockContext.Provider>
  );
};

export default MainLayout;
