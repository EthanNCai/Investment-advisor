import React, { useState, useEffect, useCallback } from 'react';
import { Card, message, Spin, Tabs, Space, Checkbox, Row, Col } from 'antd';
import ReactECharts from 'echarts-for-react';
import { useLocalStorage } from '../../LocalStorageContext';

const { TabPane } = Tabs;

interface RatioIndicatorsProps {
  stockA: string;
  stockB: string;
  duration: string;
}

interface IndicatorsData {
  code_a: string;
  code_b: string;
  dates: string[];
  ratio: number[];
  indicators: {
    ma5: number[];
    ma10: number[];
    ma20: number[];
    ma60: number[];
    macd: {
      dif: number[];
      dea: number[];
      macd: number[];
    };
    rsi: {
      rsi6: number[];
      rsi12: number[];
      rsi24: number[];
    };
    [key: string]: any; // 添加索引签名
  };
}

const RatioIndicators: React.FC<RatioIndicatorsProps> = ({ stockA, stockB, duration }) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [indicatorsData, setIndicatorsData] = useLocalStorage<IndicatorsData | null>('ratio-indicators-data', null);
  const [activeTab, setActiveTab] = useLocalStorage<string>('ratio-indicators-tab', '1');
  
  // 重新初始化maLines，确保它始终是数组
  const [maLines, setMaLines] = useState<string[]>(['ma5', 'ma10', 'ma20']);
  
  // 在组件加载时，从localStorage中获取并验证maLines
  useEffect(() => {
    try {
      const storedValue = localStorage.getItem('ratio-indicators-ma-lines');
      if (storedValue) {
        const parsedValue = JSON.parse(storedValue);
        if (Array.isArray(parsedValue) && parsedValue.length > 0) {
          setMaLines(parsedValue);
        } else {
          // 如果存储的值不是数组或为空，使用默认值并重置
          localStorage.setItem('ratio-indicators-ma-lines', JSON.stringify(['ma5', 'ma10', 'ma20']));
        }
      }
    } catch (error) {
      console.error('解析本地存储的maLines失败:', error);
      localStorage.setItem('ratio-indicators-ma-lines', JSON.stringify(['ma5', 'ma10', 'ma20']));
    }
  }, []);

  useEffect(() => {
    if (stockA && stockB) {
      fetchIndicatorsData();
    }
  }, [stockA, stockB, duration]);

  const fetchIndicatorsData = async () => {
    if (!stockA || !stockB) {
      message.warning('请选择两只股票');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/get_ratio_indicators/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          code_a: stockA,
          code_b: stockB,
          duration: duration
        }),
      });

      const data = await response.json();
      setIndicatorsData(data);
    } catch (error) {
      console.error('获取比值指标数据失败:', error);
      message.error('获取比值指标数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleMaLineChange = (checkedValues: string[]) => {
    // 确保至少选择一项
    if (checkedValues.length === 0) {
      message.warning('请至少选择一条均线');
      return;
    }
    
    // 保存到state和localStorage
    setMaLines(checkedValues);
    localStorage.setItem('ratio-indicators-ma-lines', JSON.stringify(checkedValues));
  };

  // 安全的格式化函数
  const safeToFixed = (value: any, digits: number = 4): string => {
    if (value === undefined || value === null || isNaN(value)) {
      return '--';
    }
    
    const numberValue = Number(value);
    if (isNaN(numberValue)) {
      return '--';
    }
    
    return numberValue.toFixed(digits);
  };

  const getRatioChartOption = (showLegend: boolean = true) => {
    if (!indicatorsData) return {};

    const { dates, ratio } = indicatorsData;
    
    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        },
        formatter: (params: any) => {
          if (!params || !Array.isArray(params) || params.length === 0) {
            return '';
          }
          
          const dateIndex = params[0].dataIndex;
          if (dateIndex === undefined || dateIndex < 0 || dateIndex >= dates.length) {
            return '';
          }
          
          const date = dates[dateIndex];
          const value = params[0].value;
          
          if (value === undefined || value === null) {
            return `${date}<br/>价格比值: --`;
          }
          
          return `${date}<br/>价格比值: ${safeToFixed(value)}`;
        }
      },
      legend: {
        show: showLegend,
        data: ['价格比值'],
        top: 10
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          },
          interval: Math.floor(dates.length / 10)
        }
      },
      yAxis: {
        type: 'value',
        scale: true, // 启用缩放以适应数据范围
        axisLabel: {
          formatter: (value: number) => safeToFixed(value, 2)
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: '价格比值',
          type: 'line',
          data: ratio,
          symbol: 'circle',
          symbolSize: 4,
          showSymbol: false,
          // 只在鼠标经过时显示点
          emphasis: {
            focus: 'series',
            scale: true,
            symbolSize: 6
          }
        }
      ]
    };
  };

  const getRatioMaChartOption = () => {
    if (!indicatorsData) return {};

    const { dates, ratio, indicators } = indicatorsData;
    const series = [
      {
        name: '价格比值',
        type: 'line',
        data: ratio,
        symbol: 'none',
        lineStyle: {
          width: 1.5,
          color: '#5470C6'
        }
      }
    ];

    // 添加选中的移动平均线
    const maColors: {[key: string]: string} = {
      ma5: '#91CC75',
      ma10: '#FAC858',
      ma20: '#EE6666',
      ma60: '#73C0DE'
    };

    const maNames: {[key: string]: string} = {
      ma5: 'MA5',
      ma10: 'MA10',
      ma20: 'MA20',
      ma60: 'MA60'
    };

    // 确保maLines是数组
    const maLinesArray = Array.isArray(maLines) ? maLines : ['ma5', 'ma10', 'ma20'];
    
    // 使用安全的数组遍历
    maLinesArray.forEach(line => {
      if (indicators[line]) {
        series.push({
          name: maNames[line] || line,
          type: 'line',
          data: indicators[line],
          symbol: 'none',
          lineStyle: {
            width: 1.5,
            color: maColors[line] || '#91CC75'
          }
        });
      }
    });

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        },
        formatter: function(params: any) {
          if (!params || !Array.isArray(params) || params.length === 0) {
            return '';
          }
          
          const dateIndex = params[0].dataIndex;
          if (dateIndex === undefined || dateIndex < 0 || dateIndex >= dates.length) {
            return '';
          }
          
          const date = dates[dateIndex];
          let html = `<div><strong>${date}</strong></div>`;
          
          params.forEach((param: any) => {
            if (param && param.value !== undefined && param.value !== null) {
              html += `<div>${param.seriesName}: ${safeToFixed(param.value)}</div>`;
            } else {
              html += `<div>${param?.seriesName || '--'}: --</div>`;
            }
          });
          
          return html;
        }
      },
      legend: {
        data: ['价格比值', ...maLinesArray.map(line => maNames[line] || line)],
        top: 10
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLabel: {
          formatter: (value: string) => {
            const date = new Date(value);
            return `${date.getMonth() + 1}/${date.getDate()}`;
          },
          interval: Math.floor(dates.length / 10)
        }
      },
      yAxis: {
        type: 'value',
        scale: true, // 启用缩放以适应数据范围
        axisLabel: {
          formatter: (value: number) => safeToFixed(value, 2)
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      series
    };
  };

  const getMacdChartOption = () => {
    if (!indicatorsData) return {};

    const { dates, indicators } = indicatorsData;
    const { dif, dea, macd } = indicators.macd;

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        },
        formatter: function(params: any) {
          if (!params || !Array.isArray(params) || params.length === 0) {
            return '';
          }
          
          const dateIndex = params[0].dataIndex;
          if (dateIndex === undefined || dateIndex < 0 || dateIndex >= dates.length) {
            return '';
          }
          
          const date = dates[dateIndex];
          let html = `<div><strong>${date}</strong></div>`;
          
          params.forEach((param: any) => {
            if (param && param.value !== undefined && param.value !== null) {
              html += `<div>${param.seriesName}: ${safeToFixed(param.value)}</div>`;
            } else {
              html += `<div>${param?.seriesName || '--'}: --</div>`;
            }
          });
          
          return html;
        }
      },
      legend: {
        data: ['DIF', 'DEA', 'MACD'],
        top: 10
      },
      grid: [
        {
          left: '3%',
          right: '4%',
          height: '60%'
        },
        {
          left: '3%',
          right: '4%',
          top: '75%',
          height: '20%'
        }
      ],
      xAxis: [
        {
          type: 'category',
          data: dates,
          axisLine: { lineStyle: { color: '#666' } },
          gridIndex: 0
        },
        {
          type: 'category',
          data: dates,
          gridIndex: 1,
          axisLine: { lineStyle: { color: '#666' } },
          axisLabel: { show: false }
        }
      ],
      yAxis: [
        {
          type: 'value',
          gridIndex: 0,
          scale: true, // 启用缩放以适应数据范围
          axisLine: { lineStyle: { color: '#666' } },
          splitLine: {
            lineStyle: {
              color: '#ddd',
              type: 'dashed'
            }
          },
          axisLabel: {
            formatter: (value: number) => safeToFixed(value, 4)
          }
        },
        {
          type: 'value',
          gridIndex: 1,
          scale: true, // 启用缩放以适应数据范围
          axisLine: { lineStyle: { color: '#666' } },
          splitLine: {
            lineStyle: {
              color: '#ddd',
              type: 'dashed'
            }
          },
          axisLabel: {
            formatter: (value: number) => safeToFixed(value, 4)
          }
        }
      ],
      dataZoom: [
        {
          type: 'inside',
          xAxisIndex: [0, 1],
          start: 0,
          end: 100
        },
        {
          xAxisIndex: [0, 1],
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: 'DIF',
          type: 'line',
          data: dif,
          symbol: 'none',
          lineStyle: {
            width: 1.5,
            color: '#91CC75'
          },
          xAxisIndex: 0,
          yAxisIndex: 0
        },
        {
          name: 'DEA',
          type: 'line',
          data: dea,
          symbol: 'none',
          lineStyle: {
            width: 1.5,
            color: '#FAC858'
          },
          xAxisIndex: 0,
          yAxisIndex: 0
        },
        {
          name: 'MACD',
          type: 'bar',
          data: macd,
          itemStyle: {
            color: (params: any) => {
              if (!params || params.data === undefined || params.data === null) {
                return '#CCCCCC';
              }
              return params.data >= 0 ? '#91CC75' : '#EE6666';
            }
          },
          xAxisIndex: 1,
          yAxisIndex: 1
        }
      ]
    };
  };

  const getRsiChartOption = () => {
    if (!indicatorsData) return {};

    const { dates, indicators } = indicatorsData;
    const { rsi6, rsi12, rsi24 } = indicators.rsi;

    return {
      tooltip: {
        trigger: 'axis',
        axisPointer: {
          type: 'cross'
        },
        formatter: function(params: any) {
          if (!params || !Array.isArray(params) || params.length === 0) {
            return '';
          }
          
          const dateIndex = params[0].dataIndex;
          if (dateIndex === undefined || dateIndex < 0 || dateIndex >= dates.length) {
            return '';
          }
          
          const date = dates[dateIndex];
          let html = `<div><strong>${date}</strong></div>`;
          
          params.forEach((param: any) => {
            if (param && param.value !== undefined && param.value !== null) {
              html += `<div>${param.seriesName}: ${safeToFixed(param.value, 2)}</div>`;
            } else {
              html += `<div>${param?.seriesName || '--'}: --</div>`;
            }
          });
          
          return html;
        }
      },
      legend: {
        data: ['RSI6', 'RSI12', 'RSI24'],
        top: 10
      },
      grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
      },
      xAxis: {
        type: 'category',
        data: dates,
        axisLine: { lineStyle: { color: '#666' } }
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 100,
        interval: 20,
        axisLine: { lineStyle: { color: '#666' } },
        splitLine: {
          lineStyle: {
            color: '#ddd',
            type: 'dashed'
          }
        },
        axisLabel: {
          formatter: '{value}%'
        }
      },
      dataZoom: [
        {
          type: 'inside',
          start: 0,
          end: 100
        },
        {
          start: 0,
          end: 100
        }
      ],
      series: [
        {
          name: 'RSI6',
          type: 'line',
          data: rsi6,
          symbol: 'none',
          lineStyle: {
            width: 1.5,
            color: '#91CC75'
          }
        },
        {
          name: 'RSI12',
          type: 'line',
          data: rsi12,
          symbol: 'none',
          lineStyle: {
            width: 1.5,
            color: '#FAC858'
          }
        },
        {
          name: 'RSI24',
          type: 'line',
          data: rsi24,
          symbol: 'none',
          lineStyle: {
            width: 1.5,
            color: '#EE6666'
          }
        }
      ],
      visualMap: {
        show: false,
        pieces: [
          {
            gt: 80,
            lte: 100,
            color: '#EE6666'
          },
          {
            gt: 20,
            lte: 80,
            color: '#91CC75'
          },
          {
            gt: 0,
            lte: 20,
            color: '#EE6666'
          }
        ]
      }
    };
  };

  // 使用useCallback包装图表渲染函数以减少不必要的重新计算
  const renderMacdTab = useCallback(() => {
    if (!indicatorsData) return null;
    
    return (
      <Row gutter={16}>
        <Col span={12}>
          <ReactECharts 
            option={getRatioChartOption(false)}
            style={{ height: 400 }}
            notMerge={true}
            lazyUpdate={true}
            opts={{ renderer: 'svg' }}
          />
        </Col>
        <Col span={12}>
          <ReactECharts 
            option={getMacdChartOption()}
            style={{ height: 400 }}
            notMerge={true}
            lazyUpdate={true}
            opts={{ renderer: 'svg' }}
          />
        </Col>
      </Row>
    );
  }, [indicatorsData]);

  const renderRsiTab = useCallback(() => {
    if (!indicatorsData) return null;
    
    return (
      <Row gutter={16}>
        <Col span={12}>
          <ReactECharts 
            option={getRatioChartOption(false)}
            style={{ height: 400 }}
            notMerge={true}
            lazyUpdate={true}
            opts={{ renderer: 'svg' }}
          />
        </Col>
        <Col span={12}>
          <ReactECharts 
            option={getRsiChartOption()}
            style={{ height: 400 }}
            notMerge={true}
            lazyUpdate={true}
            opts={{ renderer: 'svg' }}
          />
        </Col>
      </Row>
    );
  }, [indicatorsData]);

  return (
    <div className="ratio-indicators">
      {loading ? (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          <Spin size="large" tip="加载中..." />
        </div>
      ) : indicatorsData ? (
        <Card>
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab="移动平均线" key="1">
              <div style={{ marginBottom: 16 }}>
                <Space>
                  选择显示的均线:
                  <Checkbox.Group
                    value={maLines}
                    onChange={handleMaLineChange}
                    options={[
                      { label: 'MA5', value: 'ma5' },
                      { label: 'MA10', value: 'ma10' },
                      { label: 'MA20', value: 'ma20' },
                      { label: 'MA60', value: 'ma60' }
                    ]}
                  />
                </Space>
              </div>
              <ReactECharts 
                option={getRatioMaChartOption()}
                style={{ height: 400 }}
                notMerge={true}
                lazyUpdate={true}
                opts={{ renderer: 'svg' }}
              />
            </TabPane>
            <TabPane tab="MACD指标" key="2">
              {renderMacdTab()}
            </TabPane>
            <TabPane tab="RSI指标" key="3">
              {renderRsiTab()}
            </TabPane>
          </Tabs>
        </Card>
      ) : (
        <div style={{ textAlign: 'center', padding: '50px 0' }}>
          请选择两只股票查看比值指标
        </div>
      )}
    </div>
  );
};

export default RatioIndicators; 