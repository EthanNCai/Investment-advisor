import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import { BrowserRouter } from 'react-router-dom'
import { LocalStorageProvider } from './LocalStorageContext.tsx'
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './queryClient'

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <BrowserRouter>
            <QueryClientProvider client={queryClient}>
                <LocalStorageProvider>
                    <App/>
                </LocalStorageProvider>
            </QueryClientProvider>
        </BrowserRouter>
    </React.StrictMode>,
)