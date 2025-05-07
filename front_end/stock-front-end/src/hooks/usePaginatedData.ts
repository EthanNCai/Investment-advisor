import { useState, useMemo } from 'react';

/**
 * 客户端分页Hook，用于大数据集的分页显示
 * @param data 需要分页的完整数据数组
 * @param pageSize 每页显示的条目数
 * @returns 分页数据和分页控制
 */
export function useClientPagination<T>(data: T[], pageSize: number = 10) {
  const [currentPage, setCurrentPage] = useState(1);
  
  // 计算总页数
  const totalPages = useMemo(() => {
    return Math.ceil(data.length / pageSize);
  }, [data.length, pageSize]);
  
  // 获取当前页数据
  const currentPageData = useMemo(() => {
    const startIndex = (currentPage - 1) * pageSize;
    const endIndex = startIndex + pageSize;
    return data.slice(startIndex, endIndex);
  }, [data, currentPage, pageSize]);
  
  // 切换到指定页
  const goToPage = (page: number) => {
    // 保证页码在有效范围内
    const validPage = Math.max(1, Math.min(page, totalPages));
    setCurrentPage(validPage);
  };
  
  // 下一页
  const nextPage = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    }
  };
  
  // 上一页
  const prevPage = () => {
    if (currentPage > 1) {
      setCurrentPage(currentPage - 1);
    }
  };
  
  return {
    currentPage,
    totalPages,
    pageSize,
    currentPageData,
    goToPage,
    nextPage,
    prevPage,
    total: data.length
  };
} 