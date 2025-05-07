import { QueryClient } from '@tanstack/react-query';

// 创建并配置QueryClient实例
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // 缓存数据保留时间（1分钟）
      staleTime: 60 * 1000,
      
      // 数据过期后，组件重新挂载或窗口获得焦点时自动重新获取
      refetchOnWindowFocus: true,
      refetchOnMount: true,
      
      // 请求失败时自动重试
      retry: 1,
      
      // 保持缓存5分钟，即使组件卸载
      gcTime: 5 * 60 * 1000,
    },
  },
}); 