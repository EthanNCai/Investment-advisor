import MenuAppBar from "./components/layout/MenuAppBar.tsx";
import {Container, CssBaseline} from "@mui/material";
import MainLayout from "./components/layout/MainLayout.tsx";
import { Routes, Route } from "react-router-dom";
import AssetSelection from "./components/AssetSelection/AssetSelection.tsx";
import RatioAnalysis from "./components/ratio-analysis/RatioAnalysis.tsx";

function App() {
  return (
      <>
          <CssBaseline/>

          <>
          <div
              style={{
                  position: "fixed",
                  top: 0,
                  left: 0,
                  right: 0,
                  zIndex: 9,
              }}>
              <MenuAppBar/>
          </div>

          <Container maxWidth="lg" sx={{ marginTop: "100px" }}>
              <Routes>
                  <Route path="/" element={<MainLayout />} />
                  <Route path="/assets" element={<AssetSelection />} />
                  <Route path="/ratio-analysis" element={<RatioAnalysis />} />
              </Routes>
          </Container>
          </>
      </>
  )
}

export default App