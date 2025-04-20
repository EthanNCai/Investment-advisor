/**
 * 预测分析工具函数
 */

/**
 * 格式化预测结果供图表使用
 * @param historicalData 历史数据
 * @param predictionData 预测数据
 * @returns 格式化后的合并数据
 */
export const formatPredictionData = (historicalData: any, predictionData: any) => {
  if (!historicalData || !predictionData) return null;

  // 确保历史数据和预测数据格式正确
  const { dates: histDates, ratio: histValues } = historicalData;
  const { dates: predDates, values: predValues, upper_bound, lower_bound } = predictionData;
  
  return {
    allDates: [...histDates, ...predDates],
    historicalSeries: [...histValues, ...new Array(predValues.length).fill(null)],
    predictionSeries: [...new Array(histValues.length).fill(null), ...predValues],
    upperBoundSeries: [...new Array(histValues.length).fill(null), ...upper_bound],
    lowerBoundSeries: [...new Array(histValues.length).fill(null), ...lower_bound]
  };
};

/**
 * 计算预测数据的趋势方向
 * @param predictionValues 预测值数组
 * @returns 趋势方向: 'up'|'down'|'stable'
 */
export const calculateTrend = (predictionValues: number[]): 'up' | 'down' | 'stable' => {
  if (!predictionValues || predictionValues.length < 2) return 'stable';
  
  // 计算线性回归斜率
  const n = predictionValues.length;
  const indices = Array.from({length: n}, (_, i) => i);
  
  // 计算x和y的平均值
  const sumX = indices.reduce((sum, x) => sum + x, 0);
  const sumY = predictionValues.reduce((sum, y) => sum + y, 0);
  const meanX = sumX / n;
  const meanY = sumY / n;
  
  // 计算斜率
  let numerator = 0;
  let denominator = 0;
  
  for (let i = 0; i < n; i++) {
    numerator += (indices[i] - meanX) * (predictionValues[i] - meanY);
    denominator += Math.pow(indices[i] - meanX, 2);
  }
  
  const slope = denominator !== 0 ? numerator / denominator : 0;
  
  // 根据斜率判断趋势
  if (Math.abs(slope) < 0.001) return 'stable';
  return slope > 0 ? 'up' : 'down';
};

/**
 * 计算预测置信区间
 * @param predictionValues 预测值数组
 * @param stdDev 标准差
 * @param confidenceLevel 置信水平 (0.9, 0.95, 0.99)
 * @returns {upper: number[], lower: number[]} 上下置信区间
 */
export const calculateConfidenceIntervals = (
  predictionValues: number[], 
  stdDev: number, 
  confidenceLevel: number = 0.95
): {upper: number[], lower: number[]} => {
  // 不同置信水平对应的Z值
  const zScores: {[key: number]: number} = {
    0.9: 1.645,
    0.95: 1.96,
    0.99: 2.576
  };
  
  const zScore = zScores[confidenceLevel] || 1.96; // 默认使用95%置信水平
  const margin = zScore * stdDev;
  
  const upper = predictionValues.map(value => value + margin);
  const lower = predictionValues.map(value => value - margin);
  
  return { upper, lower };
};

/**
 * 评估预测风险级别
 * @param predictionData 预测数据
 * @param historicalVolatility 历史波动率
 * @returns 风险级别: 'low'|'medium'|'high'
 */
export const evaluateRiskLevel = (
  predictionData: any, 
  historicalVolatility: number
): 'low' | 'medium' | 'high' => {
  // 如果没有预测数据，返回低风险
  if (!predictionData) return 'low';
  
  const { values, upper_bound, lower_bound } = predictionData;
  
  // 计算预测区间范围相对于预测值的比例
  const intervalWidths = upper_bound.map((upper: number, i: number) => upper - lower_bound[i]);
  const relativeSpreads = intervalWidths.map((width: number, i: number) => width / values[i]);
  const avgRelativeSpread = relativeSpreads.reduce((sum: number, val: number) => sum + val, 0) / relativeSpreads.length;
  
  // 计算预测趋势的变化率
  const firstValue = values[0];
  const lastValue = values[values.length - 1];
  const changeRate = Math.abs((lastValue - firstValue) / firstValue);
  
  // 基于多个因素评估风险
  if (avgRelativeSpread > 0.2 || changeRate > 0.15 || historicalVolatility > 0.08) {
    return 'high';
  } else if (avgRelativeSpread > 0.1 || changeRate > 0.08 || historicalVolatility > 0.05) {
    return 'medium';
  } else {
    return 'low';
  }
};

export default {
  formatPredictionData,
  calculateTrend,
  calculateConfidenceIntervals,
  evaluateRiskLevel
}; 