import MenuAppBar from "./components/MenuAppBar.tsx";
import {Container, CssBaseline} from "@mui/material";
import MainLayout from "./components/MainLayout.tsx";



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

          <Container  maxWidth="lg" sx={{  marginTop: "100px" }}>

              <MainLayout/>
          </Container>
</>
      </>
  )
}

export default App
