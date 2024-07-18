import MenuAppBar from "./components/MenuAppBar.tsx";
import {Container, CssBaseline} from "@mui/material";
import MainWidgetLayout from "./components/MainWidgetLayout.tsx";



function App() {

  return (
      <><CssBaseline/>
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

          <Container maxWidth="lg" sx={{ padding: "0px", marginTop: "66px" }}>

              <MainWidgetLayout/>
          </Container>

      </>
  )
}

export default App
