import { Box } from "@mui/system";

import SearchBar from "../common/Search.tsx";
import  { useEffect, useState } from "react";
import { StockContext, KChartInfo, UserOptionInfo, StockInfo } from "../common/interfaces.tsx";

import { KChart } from "../charts/KChart.tsx";
import { Paper } from "@mui/material";

const MainLayout = () => {
  const [degree, setDegree] = useState(2);
  const [duration, setDuration] = useState('1y');
  const [threshold_arg, setThreshold_arg] = useState(1.5);
  const [kChartInfo, setKChartInfo] = useState<KChartInfo | undefined>(
    undefined
  );
  const [stockInfoA, setStockInfoA] = useState<StockInfo>({name:"",code:"",type:""});
  const [stockInfoB, setStockInfoB] = useState<StockInfo>({name:"",code:"",type:""});
  const [showRatio, setShowRatio] = useState(true);
  const [showDelta, setShowDelta] = useState(true);

  useEffect(() => {
    const fetchKChartInfo = () => {
      if (stockInfoA.code === "" || stockInfoB.code === "") {
        return;
      }
      const user_option_info: UserOptionInfo = {
          code_a:stockInfoA.code,
          code_b:stockInfoB.code,
          duration:duration,
          threshold_arg:threshold_arg,
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

  }, [stockInfoA, stockInfoB, duration, degree, threshold_arg]);

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
          showDelta:showDelta,
          setShowDelta:setShowDelta ,
          showRatio:showRatio,
          setShowRatio:setShowRatio,
      }}
    >
      <Box>

        <Box
          sx={{
            display: "flex",
            flexDirection: {
              xs: "column",
              sm: "column",
              lg: "row",
            },
              gap: 2,
          }}
        >

          <Box sx={{ flex: 1 }}>
              <Paper>
            <Box sx={{padding:"10px"}}>
              <SearchBar />
            </Box>
              </Paper>
          </Box>

            <Box sx={{ flex: 2 }}>
                <Box >
                    <KChart />
                </Box>
            </Box>
        </Box>
      </Box>
    </StockContext.Provider>
  );
};

export default MainLayout;
