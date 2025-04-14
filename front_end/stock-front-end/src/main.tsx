import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import { BrowserRouter } from 'react-router-dom'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <BrowserRouter>
            <App/>
        </BrowserRouter>
    </React.StrictMode>,
)
