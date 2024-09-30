import { Box } from "@mui/system";
import { Paper } from "@mui/material";
import SearchBar from "./Search.tsx";
import React, { useEffect, useState } from "react";
import {StockContext, KChartInfo, UserOptionInfo} from "./interfaces.tsx";

import { KChart } from "./KChart.tsx";
const MainLayout = () => {
  const [code_a, setCode_a] = useState("");
  const [code_b, setCode_b] = useState("");
  const [degree, setDegree] = useState(3);
  const [duration, setDuration] = useState('1m');
  const [threshold, setThreshold] = useState(1.5);
  const [kChartInfo, setKChartInfo] = useState<KChartInfo | undefined>(
    undefined
  );

  useEffect(() => {
    const fetchKChartInfo = () => {
      if (code_a === "" || code_b === "") {
        return;
      }
      const user_option_info: UserOptionInfo = {
          code_a:code_a,
          code_b:code_b,
          duration:duration,
          threshold:threshold,
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
    if (code_a !== "") {
      fetchKChartInfo();
      console.log(kChartInfo);
    }
  }, [code_a, code_b,duration,degree,threshold]);

  return (
    <StockContext.Provider
      value={{
        codeA: code_a,
        setCodeA: setCode_a,
        codeB: code_b,
        setCodeB: setCode_b,
        kChartInfo: kChartInfo,
        degree:degree,
        setDegree:setDegree,
        duration:duration,
        setDuration:setDuration,
        threshold:threshold,
        setThreshold:setThreshold,
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
          }}
        >
          <Box sx={{ flex: 2 }}>
            <Paper>
              <KChart />
            </Paper>
          </Box>
          <Box sx={{ flex: 1 }}>
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
