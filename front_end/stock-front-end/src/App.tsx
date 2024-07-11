import MenuAppBar from "./components/MenuAppBar.tsx";
import {Container} from "@mui/material";
import MainWidgetLayout from "./components/MainWidgetLayout.tsx";


function App() {

  return (
    <>
        <MenuAppBar/>
        <Container className="App">


            <MainWidgetLayout/>
        </Container>

    </>
  )
}

export default App
