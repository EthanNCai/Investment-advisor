import { useState, useMemo } from 'react';

/**
 * 实现"加载更多"功能的Hook
 * @param data 完整的数据数组
 * @param initialLimit 初始显示的条目数
 * @param increment 每次加载更多的增量
 * @returns 当前显示的数据和加载控制
 */
export function useLoadMore<T>(data: T[], initialLimit: number = 5, increment: number = 5) {
  const [limit, setLimit] = useState(initialLimit);
  
  // 当前显示的数据
  const visibleData = useMemo(() => {
    return data.slice(0, limit);
  }, [data, limit]);
  
  // 是否还有更多数据可以加载
  const hasMore = useMemo(() => {
    return limit < data.length;
  }, [data.length, limit]);
  
  // 加载更多数据
  const loadMore = () => {
    setLimit(prevLimit => {
      const newLimit = prevLimit + increment;
      // 确保不超过数据总量
      return Math.min(newLimit, data.length);
    });
  };
  
  // 重置到初始状态
  const reset = () => {
    setLimit(initialLimit);
  };
  
  return {
    visibleData,
    hasMore,
    loadMore,
    reset,
    total: data.length,
    loaded: visibleData.length
  };
} 