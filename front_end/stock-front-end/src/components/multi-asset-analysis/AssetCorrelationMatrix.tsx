import React, { useEffect, useRef } from 'react';
import { Card, Empty, Spin } from 'antd';
import * as echarts from 'echarts';
import { debounce } from 'lodash';

interface AssetCorrelationMatrixProps {
  data: any;
  onPairSelect: (assetA: string, assetB: string) => void;
  selectedPair: string[];
}

const AssetCorrelationMatrix: React.FC<AssetCorrelationMatrixProps> = ({ 
  data, 
  onPairSelect,
  selectedPair 
}) => {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  // 生成图表数据
  const generateChartData = () => {
    if (!data || !data.relationsMatrix) return null;

    const { assets, assetNames, relationsMatrix } = data;

    // 准备数据系列
    const series = [];
    const categories = assets.map((code: string, index: number) => ({
      name: `${assetNames[index] || code}`
    }));

    // 图表点击回调
    const handleChartClick = (params: any) => {
      if (params.data && params.data.source !== params.data.target) {
        const sourceAsset = assets[params.data.source];
        const targetAsset = assets[params.data.target];
        onPairSelect(sourceAsset, targetAsset);
        
        // 自动跳转到"资产对详情"页面
        const detailTabPane = document.querySelector('.ant-tabs-tab[data-node-key="2"]') as HTMLElement;
        if (detailTabPane) {
          detailTabPane.click();
        }
      }
    };

    // 计算单元格颜色
    const getCellColor = (value: number) => {
      if (value === 0) return '#e0e0e0';  // 中性/无信号
      if (value > 0) {
        // 正值，做多第一个资产、做空第二个资产
        const intensity = Math.min(Math.abs(value) / 100, 1);
        return `rgba(238, 102, 102, ${intensity})`; // 红色，强度根据值变化
      } else {
        // 负值，做空第一个资产、做多第二个资产
        const intensity = Math.min(Math.abs(value) / 100, 1);
        return `rgba(145, 204, 117, ${intensity})`; // 绿色，强度根据值变化
      }
    };

    // 生成热力图数据
    const heatmapData = [];
    for (let i = 0; i < assets.length; i++) {
      for (let j = 0; j < assets.length; j++) {
        if (i !== j) {  // 不显示自己与自己的比较
          const value = relationsMatrix[i][j];
          const isSelected = (
            (selectedPair[0] === assets[i] && selectedPair[1] === assets[j]) ||
            (selectedPair[0] === assets[j] && selectedPair[1] === assets[i])
          );
          
          heatmapData.push({
            value: [j, i, value],
            itemStyle: {
              color: getCellColor(value),
              borderColor: isSelected ? '#1890ff' : 'transparent',
              borderWidth: isSelected ? 2 : 0
            },
            source: i,
            target: j
          });
        }
      }
    }

    // 加入热力图系列
    series.push({
      name: '相对强弱',
      type: 'heatmap',
      data: heatmapData,
      label: {
        show: true,
        formatter: (params: any) => {
          const value = params.data.value[2];
          return value === 0 ? '' : value.toFixed(1);
        },
        color: '#000'
      },
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      }
    });
    
    return {
      tooltip: {
        position: 'top',
        formatter: (params: any) => {
          const i = params.data.source;
          const j = params.data.target;
          const value = params.data.value[2];
          
          if (i === j) return `${categories[i].name}`;
          
          const sourceAsset = categories[i].name;
          const targetAsset = categories[j].name;
          
          if (value === 0) {
            return `${sourceAsset} vs ${targetAsset}<br/>无明显信号`;
          }
          
          let signalStrength = '';
          if (Math.abs(value) >= 80) signalStrength = '极强';
          else if (Math.abs(value) >= 60) signalStrength = '很强';
          else if (Math.abs(value) >= 40) signalStrength = '中等';
          else if (Math.abs(value) >= 20) signalStrength = '较弱';
          else signalStrength = '很弱';
          
          if (value > 0) {
            return `${sourceAsset} vs ${targetAsset}<br/>信号: <span style="color:#d81e06">做多${assets[i]}, 做空${assets[j]}</span><br/>强度: ${signalStrength} (${value.toFixed(1)})`;
          } else {
            return `${sourceAsset} vs ${targetAsset}<br/>信号: <span style="color:#1ca642">做空${assets[i]}, 做多${assets[j]}</span><br/>强度: ${signalStrength} (${value.toFixed(1)})`;
          }
        }
      },
      grid: {
        height: '70%',
        top: '10%'
      },
      xAxis: {
        type: 'category',
        data: assets,
        splitArea: {
          show: true
        },
        axisLabel: {
          formatter: (value: string, index: number) => {
            return value;
          },
          interval: 0,
          rotate: 30,
          fontSize: 11
        }
      },
      yAxis: {
        type: 'category',
        data: assets,
        splitArea: {
          show: true
        },
        axisLabel: {
          formatter: (value: string, index: number) => {
            return value;
          }
        }
      },
      visualMap: {
        min: -100,
        max: 100,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        text: ['做多列资产', '做多行资产'],
        inRange: {
          color: ['#91CC75', '#DCDCDC', '#EE6666']
        }
      },
      series
    };
  };

  // 初始化和更新图表
  useEffect(() => {
    if (!chartRef.current || !data) return;

    // 处理窗口大小变化
    const handleResize = debounce(() => {
      if (chartInstance.current) {
        chartInstance.current.resize();
      }
    }, 300);

    // 只在第一次渲染时初始化图表
    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
      
      // 添加点击事件处理
      chartInstance.current.on('click', (params) => {
        if (params.data && params.data.source !== params.data.target) {
          const sourceAsset = data.assets[params.data.source];
          const targetAsset = data.assets[params.data.target];
          onPairSelect(sourceAsset, targetAsset);
          
          // 自动跳转到"资产对详情"页面
          const detailTabPane = document.querySelector('.ant-tabs-tab[data-node-key="2"]') as HTMLElement;
          if (detailTabPane) {
            detailTabPane.click();
          }
        }
      });
    }
    
    // 更新图表
    const option = generateChartData();
    if (option && chartInstance.current) {
      chartInstance.current.setOption(option, true);
    }
    
    // 添加窗口大小变化事件监听
    window.addEventListener('resize', handleResize);
    
    // 清理函数
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, [data, selectedPair]);
  
  // 组件卸载时销毁图表
  useEffect(() => {
    return () => {
      if (chartInstance.current) {
        chartInstance.current.dispose();
        chartInstance.current = null;
      }
    };
  }, []);

  // 如果没有数据，显示空状态
  if (!data || !data.relationsMatrix) {
    return (
      <Card title="资产相对强弱矩阵" style={{ height: '100%', minHeight: 400 }}>
        <Empty description="暂无分析数据" />
      </Card>
    );
  }

  return (
    <Card title="资产相对强弱矩阵" style={{ height: '100%', minHeight: 400 }}>
      <div style={{ height: 400, width: '100%' }} ref={chartRef}></div>
      <div style={{ textAlign: 'center', marginTop: 10, fontSize: '0.9em', color: '#666' }}>
        矩阵说明: 行资产相对于列资产的信号强度。正值(红色)表示做多行资产做空列资产，负值(绿色)表示做空行资产做多列资产。
      </div>
    </Card>
  );
};

export default AssetCorrelationMatrix; 