import { useEffect, useState } from "react";
import { Button, Container, Stack } from "@mui/material";
import Typography from "@mui/material/Typography";
import TextField from "@mui/material/TextField";
import { Box } from "@mui/system";
import { useContext } from "react";
import {StockContext, StockInfo} from "./interfaces.tsx";
import {SliderSelector} from "./SliderSelector.tsx";

export default function SearchBar() {
  const {
    stockInfoA,
    setStockInfoA,
    stockInfoB,
    setStockInfoB ,
    degree,
    setDegree,
    threshold_arg,
    setThreshold_arg,



  } = useContext(StockContext)!;
  const [search_keyword, setSearch_keyword] = useState("");
  const [stockInfos, setStockInfos] = useState<StockInfo[]>([]);



  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(
          `http://localhost:8000/search_stocks/${search_keyword}`
        );
        const data = await response.json();
        setStockInfos(data.result);
      } catch (error) {
        console.error("Error fetching data:", error);
      }
    };

    if (search_keyword !== "") {
      fetchData();
    } else {
      setStockInfos([]); // Clear search results if search keyword is empty
    }
  }, [search_keyword]);

  return (
    <Container>
      <Stack>
        <Typography>
          A:

          {stockInfoA.name ? stockInfoA.name : "未选定"}
        </Typography>
        <Typography>
          B:

          {stockInfoB.name ? stockInfoB.name : "未选定"}
        </Typography>
        <SliderSelector
            title={"拟合曲线阶"}
            current_value={degree}
            set_current_value={setDegree}
            max={8}
            min={1}
            step={1}
        />


        <SliderSelector
            title={"异常值门限"}
            current_value={threshold_arg}
            set_current_value={setThreshold_arg}
            max={3}
            min={1}
            step={0.05}
        />



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

      {stockInfos.map((item: StockInfo) => (
        <Box key={item.code}>
          <Stack direction={"row"}>
            <Typography>{item.code}-</Typography>
            <Typography>{item.name}</Typography>
            <Typography>-{item.type}</Typography>
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
          </Stack>
        </Box>
      ))}

    </Container>
  );
}
