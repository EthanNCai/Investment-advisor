import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// 定义持久化存储上下文类型
interface LocalStorageContextType {
  getItem: <T>(key: string, defaultValue: T) => T;
  setItem: <T>(key: string, value: T) => void;
  removeItem: (key: string) => void;
  clear: () => void;
}

// 创建上下文
const LocalStorageContext = createContext<LocalStorageContextType | undefined>(undefined);

// 提供者组件
export const LocalStorageProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  // 定义本地存储操作方法
  const getItem = <T,>(key: string, defaultValue: T): T => {
    if (typeof window === 'undefined') {
      return defaultValue;
    }
    
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
      console.error(`获取localStorage项目时出错: ${key}`, error);
      return defaultValue;
    }
  };

  const setItem = <T,>(key: string, value: T): void => {
    if (typeof window === 'undefined') {
      return;
    }
    
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      console.error(`设置localStorage项目时出错: ${key}`, error);
    }
  };

  const removeItem = (key: string): void => {
    if (typeof window === 'undefined') {
      return;
    }
    
    try {
      window.localStorage.removeItem(key);
    } catch (error) {
      console.error(`删除localStorage项目时出错: ${key}`, error);
    }
  };

  const clear = (): void => {
    if (typeof window === 'undefined') {
      return;
    }
    
    try {
      window.localStorage.clear();
    } catch (error) {
      console.error('清除localStorage时出错', error);
    }
  };

  // 暴露给上下文的值
  const contextValue: LocalStorageContextType = {
    getItem,
    setItem,
    removeItem,
    clear
  };

  return (
    <LocalStorageContext.Provider value={contextValue}>
      {children}
    </LocalStorageContext.Provider>
  );
};

// 自定义Hook，用于在组件中访问LocalStorage上下文
export const useLocalStorage = <T,>(key: string, defaultValue: T): [T, (value: T) => void] => {
  const context = useContext(LocalStorageContext);
  
  if (context === undefined) {
    throw new Error('useLocalStorage必须在LocalStorageProvider内部使用');
  }
  
  // 获取保存的值
  const [storedValue, setStoredValue] = useState<T>(() => {
    return context.getItem(key, defaultValue);
  });
  
  // 监听其他组件或标签页对localStorage的更改
  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === key && event.newValue !== null) {
        try {
          const newValue = JSON.parse(event.newValue);
          setStoredValue(newValue);
        } catch (e) {
          console.error(`解析localStorage变更事件出错: ${key}`, e);
        }
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    return () => {
      window.removeEventListener('storage', handleStorageChange);
    };
  }, [key]);
  
  // 定义更新值的方法
  const setValue = (value: T) => {
    setStoredValue(value);
    context.setItem(key, value);
  };
  
  return [storedValue, setValue];
};

// 使用示例
// const [userData, setUserData] = useLocalStorage('userData', { name: '', preferences: {} }); 