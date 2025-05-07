import React, { useState, useEffect, useRef, ReactNode } from 'react';
import { Spin } from 'antd';

interface LazyLoadComponentProps {
  children: ReactNode;
  placeholder?: ReactNode;
  height?: number | string;
  threshold?: number; // 触发加载的阈值，表示元素可见比例
  once?: boolean; // 是否只触发一次
}

/**
 * 懒加载组件，当组件进入视口时才加载内容
 */
const LazyLoadComponent: React.FC<LazyLoadComponentProps> = ({
  children,
  placeholder = <Spin />,
  height = 200,
  threshold = 0.1,
  once = true
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // 如果已经加载过且设置为只加载一次，则不再监听
    if (once && hasLoaded) return;

    const currentElement = containerRef.current;
    if (!currentElement) return;

    // 创建Intersection Observer实例
    const observer = new IntersectionObserver(
      (entries) => {
        // 当容器进入视口
        if (entries[0].isIntersecting) {
          setIsVisible(true);
          if (once) {
            setHasLoaded(true);
            // 如果只需要加载一次，加载后取消监听
            observer.unobserve(currentElement);
          }
        } else if (!once) {
          // 如果需要多次触发，则在离开视口时设置为不可见
          setIsVisible(false);
        }
      },
      {
        root: null, // 使用视口作为root
        rootMargin: '0px',
        threshold // 当元素有10%可见时触发
      }
    );

    // 开始监听容器元素
    observer.observe(currentElement);

    // 组件卸载时取消监听
    return () => {
      if (currentElement) {
        observer.unobserve(currentElement);
      }
    };
  }, [once, hasLoaded, threshold]);

  return (
    <div 
      ref={containerRef}
      style={{ 
        minHeight: isVisible ? 'auto' : height,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center'
      }}
    >
      {isVisible ? children : placeholder}
    </div>
  );
};

export default LazyLoadComponent; 