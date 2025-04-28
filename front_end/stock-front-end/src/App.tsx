import MenuAppBar from "./components/layout/MenuAppBar.tsx";
import {Container, CssBaseline} from "@mui/material";
import MainLayout from "./components/layout/MainLayout.tsx";
import {Routes, Route, Navigate} from "react-router-dom";
import AssetSelection from "./components/AssetSelection/AssetSelection.tsx";
import RatioAnalysis from "./components/ratio-analysis/RatioAnalysis.tsx";
import InvestmentSignal from "./components/signal/InvestmentSignal.tsx";
import Login from "./components/auth/Login.tsx";
import Register from "./components/auth/Register.tsx";
import ForgotPassword from "./components/auth/ForgotPassword.tsx";
import Dashboard from "./components/Dashboard/Dashboard.tsx";
import { useLocalStorage } from "./LocalStorageContext.tsx";

// 保护路由组件
const ProtectedRoute = ({ children, isAuthenticated }) => {
    if (!isAuthenticated) {
        // 如果未登录，重定向到登录页面
        return <Navigate to="/login" replace />;
    }
    
    return children;
};

function App() {
    const [isLoggedIn] = useLocalStorage<boolean>('isLoggedIn', false);
    
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

                <Container maxWidth="lg" sx={{marginTop: "100px"}}>
                    <Routes>
                        {/* 认证页面 */}
                        <Route path="/login" element={<Login/>}/>
                        <Route path="/register" element={<Register/>}/>
                        <Route path="/forgot-password" element={<ForgotPassword/>}/>
                        
                        {/* 主页路由 */}
                        <Route path="/" element={isLoggedIn ? <Dashboard/> : <Login/>}/>
                        
                        {/* 需要保护的路由 */}
                        <Route path="/dashboard" element={
                            <ProtectedRoute isAuthenticated={isLoggedIn}>
                                <Dashboard/>
                            </ProtectedRoute>
                        }/>
                        
                        <Route path="/main" element={
                            <ProtectedRoute isAuthenticated={isLoggedIn}>
                                <MainLayout/>
                            </ProtectedRoute>
                        }/>
                        
                        {/* 功能页面 */}
                        <Route path="/assets" element={
                            <ProtectedRoute isAuthenticated={isLoggedIn}>
                                <AssetSelection/>
                            </ProtectedRoute>
                        }/>
                        
                        <Route path="/ratio-analysis" element={
                            <ProtectedRoute isAuthenticated={isLoggedIn}>
                                <RatioAnalysis/>
                            </ProtectedRoute>
                        }/>
                        
                        <Route path="/investment-signals" element={
                            <ProtectedRoute isAuthenticated={isLoggedIn}>
                                <InvestmentSignal />
                            </ProtectedRoute>
                        }/>
                    </Routes>
                </Container>
            </>
        </>
    )
}

export default App