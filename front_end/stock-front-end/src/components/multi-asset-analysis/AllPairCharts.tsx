import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Spin, Empty, Typography, Tooltip } from 'antd';
import * as echarts from 'echarts';
import axios from 'axios';
import { useParams, useSearchParams } from 'react-router-dom';

const { Title } = Typography;

// 图表尺寸设置
const CHART_HEIGHT = 240;

interface ChartData {
  code_a: string;
  code_b: string;
  dates: string[];
  ratio: number[];
  fitting_line: number[];
  green_points: {date: string; value: number}[];
  red_points: {date: string; value: number}[];
  current_ratio: number;
}

interface AllPairChartsProps {
  assets?: string[];
  duration?: string;
  polynomialDegree?: number;
  thresholdMultiplier?: number;
  klineType?: string;
}

const AllPairCharts: React.FC<AllPairChartsProps> = ({ 
  assets = [],
  duration = "2y",
  polynomialDegree = 3,
  thresholdMultiplier = 2.0,
  klineType = "daily"
}) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [chartData, setChartData] = useState<{[key: string]: ChartData}>({});
  const [assetNames, setAssetNames] = useState<{[key: string]: string}>({});
  const [searchParams] = useSearchParams();
  
  // 在组件挂载时或参数变化时获取数据
  useEffect(() => {
    const assetsParam = searchParams.get('assets');
    const durationParam = searchParams.get('duration');
    const degreeParam = searchParams.get('degree');
    const thresholdParam = searchParams.get('threshold');
    const klineTypeParam = searchParams.get('kline_type');
    
    // 使用URL参数或传入的props
    const assetsToUse = assetsParam ? assetsParam.split(',') : assets;
    const durationToUse = durationParam || duration;
    const degreeToUse = degreeParam ? parseInt(degreeParam) : polynomialDegree;
    const thresholdToUse = thresholdParam ? parseFloat(thresholdParam) : thresholdMultiplier;
    const klineTypeToUse = klineTypeParam || klineType;
    
    if (assetsToUse.length < 2) {
      setLoading(false);
      return;
    }
    
    fetchChartData(assetsToUse, durationToUse, degreeToUse, thresholdToUse, klineTypeToUse);
  }, [assets, duration, polynomialDegree, thresholdMultiplier, klineType, searchParams]);
  
  // 获取所有资产对的比值图表数据
  const fetchChartData = async (
    assetsList: string[], 
    durationValue: string,
    degreeValue: number,
    thresholdValue: number,
    klineTypeValue: string
  ) => {
    setLoading(true);
    try {
      const response = await axios.post('http://localhost:8000/get_all_pair_charts/', {
        assets: assetsList,
        duration: durationValue,
        polynomial_degree: degreeValue,
        threshold_multiplier: thresholdValue,
        kline_type: klineTypeValue
      });
      
      if (response.data) {
        setChartData(response.data.pairCharts || {});
        
        // 构建资产代码到名称的映射
        const nameMap: {[key: string]: string} = {};
        const names = response.data.assetNames || [];
        const codes = response.data.assets || [];
        
        codes.forEach((code: string, index: number) => {
          nameMap[code] = names[index] || code;
        });
        
        setAssetNames(nameMap);
      }
    } catch (error) {
      console.error("获取资产对比值图表数据失败:", error);
    } finally {
      setLoading(false);
    }
  };
  
  // 创建图表
  useEffect(() => {
    if (loading || Object.keys(chartData).length === 0) return;
    
    // 创建所有图表
    Object.entries(chartData).forEach(([pairKey, data]) => {
      const chartContainer = document.getElementById(`chart-${pairKey}`);
      if (!chartContainer) return;
      
      const chartInstance = echarts.init(chartContainer);
      
      // 绿色点和红色点数据
      const greenData = data.green_points.map(point => [point.date, point.value]);
      const redData = data.red_points.map(point => [point.date, point.value]);
      
      // 计算时间跨度和合适的标签间隔
      const dateCount = data.dates.length;
      const interval = Math.max(Math.floor(dateCount / 5), 1);
      
      // 构建图表选项
      const option = {
        grid: {
          left: '10%',
          right: '5%',
          top: '15%',
          bottom: '15%'
        },
        tooltip: {
          trigger: 'axis',
          formatter: (params: any) => {
            let result = params[0].axisValue + '<br/>';
            
            // 添加价格比值
            const ratioItem = params.find((item: any) => item.seriesName === '价格比值');
            if (ratioItem) {
              result += `价格比值: ${ratioItem.value[1].toFixed(4)}<br/>`;
            }
            
            // 添加拟合曲线
            const fittingItem = params.find((item: any) => item.seriesName === '拟合曲线');
            if (fittingItem) {
              result += `拟合曲线: ${fittingItem.value[1].toFixed(4)}<br/>`;
            }
            
            return result;
          }
        },
        xAxis: {
          type: 'category',
          data: data.dates,
          boundaryGap: false,
          axisLabel: {
            interval: interval,
            formatter: (value: string) => {
              const date = new Date(value);
              return `${date.getMonth() + 1}/${date.getDate()}/${date.getFullYear().toString().substr(-2)}`;
            }
          }
        },
        yAxis: {
          type: 'value',
          scale: true
        },
        series: [
          {
            name: '价格比值',
            type: 'line',
            data: data.dates.map((date, index) => [date, data.ratio[index]]),
            showSymbol: false,
            lineStyle: {
              width: 1,
              color: '#5470C6'
            }
          },
          {
            name: '拟合曲线',
            type: 'line',
            data: data.dates.map((date, index) => [date, data.fitting_line[index]]),
            showSymbol: false,
            lineStyle: {
              width: 2,
              type: 'dashed',
              color: '#91CC75'
            }
          },
          {
            name: '绿色区域',
            type: 'scatter',
            data: greenData,
            symbolSize: 8,
            itemStyle: {
              color: '#91CC75'  // 绿色，与矩阵和其他组件颜色一致
            }
          },
          {
            name: '红色区域',
            type: 'scatter',
            data: redData,
            symbolSize: 8,
            itemStyle: {
              color: '#EE6666'  // 红色，与矩阵和其他组件颜色一致
            }
          }
        ]
      };
      
      chartInstance.setOption(option);
      
      // 窗口大小变化时自动调整图表大小
      const handleResize = () => {
        chartInstance.resize();
      };
      
      window.addEventListener('resize', handleResize);
      
      // 清理函数
      return () => {
        window.removeEventListener('resize', handleResize);
        chartInstance.dispose();
      };
    });
  }, [chartData, loading]);
  
  // 渲染所有图表
  const renderCharts = () => {
    if (loading) {
      return (
        <div style={{ textAlign: 'center', padding: 50 }}>
          <Spin size="large" tip="正在加载所有资产对比值图表..." />
        </div>
      );
    }
    
    if (Object.keys(chartData).length === 0) {
      return (
        <Empty description="没有可用的资产对比值图表数据" />
      );
    }
    
    return (
      <Row gutter={[16, 16]}>
        {Object.entries(chartData).map(([pairKey, data]) => {
          const [codeA, codeB] = pairKey.split('_');
          const nameA = assetNames[codeA] || codeA;
          const nameB = assetNames[codeB] || codeB;
          const currentRatio = data.current_ratio.toFixed(4);
          
          return (
            <Col xs={24} sm={24} md={12} lg={8} key={pairKey}>
              <Card 
                size="small" 
                title={
                  <Tooltip title={`${nameA}(${codeA}) / ${nameB}(${codeB})`}>
                    <div style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {nameA} / {nameB}
                    </div>
                  </Tooltip>
                }
                extra={
                  <div style={{ fontWeight: 'bold' }}>
                    {currentRatio}
                  </div>
                }
              >
                <div id={`chart-${pairKey}`} style={{ height: CHART_HEIGHT, width: '100%' }}></div>
                <div style={{ marginTop: 8, fontSize: '0.85em', color: '#666' }}>
                  <Row>
                    <Col span={12}><span style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: '#91CC75', marginRight: 4 }}></span> 绿色区域: 做空{codeA}做多{codeB}</Col>
                    <Col span={12}><span style={{ display: 'inline-block', width: 12, height: 12, backgroundColor: '#EE6666', marginRight: 4 }}></span> 红色区域: 做多{codeA}做空{codeB}</Col>
                  </Row>
                </div>
              </Card>
            </Col>
          );
        })}
      </Row>
    );
  };
  
  return (
    <div className="all-pair-charts">
      <Title level={3}>资产对比值图表</Title>
      {renderCharts()}
    </div>
  );
};

export default AllPairCharts; 