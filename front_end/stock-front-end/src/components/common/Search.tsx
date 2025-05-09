import { useEffect, useState } from "react";
import { Button, Container, Stack } from "@mui/material";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import { Box } from "@mui/system";
import { useContext } from "react";
import { StockContext, StockInfo } from "./interfaces.tsx";
import { SliderSelector } from "../selectors/SliderSelector.tsx";
import { StockDivision } from "./StockDivision.tsx";

export default function SearchBar() {
  const {
    stockInfoA,
    setStockInfoA,
    stockInfoB,
    setStockInfoB,
    degree,
    setDegree,
    threshold_arg,
    setThreshold_arg,
  } = useContext(StockContext)!;
  const [search_keyword, setSearch_keyword] = useState("");
  const [stockInfos, setStockInfos] = useState<StockInfo[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // 临时状态用于SliderSelector内部处理
  const [tempDegree, setTempDegree] = useState<number>(degree);
  const [tempThreshold, setTempThreshold] = useState<number>(threshold_arg);
  
  // 当上下文中的值变化时，更新临时状态
  useEffect(() => {
    setTempDegree(degree);
    setTempThreshold(threshold_arg);
  }, [degree, threshold_arg]);

  // 使用防抖函数优化搜索，避免频繁请求
  useEffect(() => {
    const debounceTimeout = setTimeout(() => {
      if (search_keyword !== "") {
        setIsSearching(true);
        fetchData();
      } else {
        setStockInfos([]); // 清空搜索结果
      }
    }, 300); // 300ms延迟，减少请求频率

    return () => clearTimeout(debounceTimeout);
  }, [search_keyword]);

  const fetchData = async () => {
    try {
      const response = await fetch(
        `http://localhost:8000/search_stocks/${search_keyword}`
      );
      const data = await response.json();
      setStockInfos(data.result);
      setIsSearching(false);
    } catch (error) {
      console.error("Error fetching data:", error);
      setIsSearching(false);
    }
  };
  
  // 处理拟合曲线阶数变化
  const handleDegreeChange = (value: number) => {
    setTempDegree(value);
    setDegree(value);
  };
  
  // 处理异常值门限变化
  const handleThresholdChange = (value: number) => {
    setTempThreshold(value);
    setThreshold_arg(value);
  };

  return (
    <Container>
      <Stack>
        <StockDivision stockA={stockInfoA} stockB={stockInfoB}/>
        <SliderSelector
            title={"拟合曲线阶"}
            value={tempDegree}
            onChange={handleDegreeChange}
            max={8}
            min={1}
            step={1}
        />

        <SliderSelector
            title={"异常值门限"}
            value={tempThreshold}
            onChange={handleThresholdChange}
            max={3}
            min={1}
            step={0.05}
        />

        <TextField
            sx={{marginY:"15px"}}
            onChange={(event) => {
              const newValue = event.target.value;
              setSearch_keyword(newValue);
            }}
            label="搜索股票代码/名称"
            variant="outlined"
            helperText={isSearching ? "搜索中..." : "实时搜索，输入即可显示结果"}
        />
      </Stack>

      {stockInfos.map((item: StockInfo) => (
        <Box key={item.code}>
          <Stack direction={"row"} justifyContent={'space-between'}>
            <Box>
            <Typography>{item.code}-{item.name}-{item.type}</Typography>
            </Box>
            <Box>
            <Button
              onClick={() => {
                if (stockInfoB.code === item.code) {
                  alert("A和B不能相同");
                } else {
                  setStockInfoA({...item})
                }
              }}
            >
              作为A
            </Button>
            <Button
              onClick={() => {
                if (stockInfoA.code === item.code) {
                  alert("A和B不能相同");
                } else {
                  setStockInfoB({...item})
                }
              }}
            >
              作为B
            </Button>
            </Box>
          </Stack>
        </Box>
      ))}
    </Container>
  );
}
