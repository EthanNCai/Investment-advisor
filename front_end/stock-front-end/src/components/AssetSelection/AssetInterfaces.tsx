import { createContext } from 'react';

// 股票基本信息接口
export interface AssetInfo {
    code: string;
    name: string;
    type: string;
}

// K线数据点接口
export interface KlineDataPoint {
    date: string;
    open: number;
    close: number;
    high: number;
    low: number;
    volume: number;
}

// 股票趋势数据点接口
export interface TrendDataPoint {
    date: string;          // 格式: "2025-04-18 09:30:00"
    current_price: number; // 当前价格
    volume: number;        // 成交量
}

// MACD指标接口
export interface MacdIndicator {
    dif: number[];
    dea: number[];
    macd: number[];
}

// RSI指标接口
export interface RsiIndicator {
    rsi6: number[];
    rsi12: number[];
    rsi24: number[];
}

// KDJ指标接口
export interface KdjIndicator {
    k: number[];
    d: number[];
    j: number[];
}

// 技术指标接口
export interface TechnicalIndicators {
    ma5: number[];
    ma10: number[];
    ma20: number[];
    ma30: number[];
    ma60: number[];
    macd: MacdIndicator;
    rsi: RsiIndicator;
    kdj?: KdjIndicator;
}

// 股票K线数据接口
export interface StockKlineData {
    code: string;
    name: string;
    type: string;
    kline_data: KlineDataPoint[];
    indicators: TechnicalIndicators;
    trends?: TrendDataPoint[]; // 实时价格趋势数据，仅在klineType为'realtime'时有值
}

// K线类型
export type KlineType = 'daily' | 'weekly' | 'monthly' | 'yearly' | 'realtime';

// 资产选择上下文
export interface AssetSelectionContext {
    selectedAsset: AssetInfo | null;
    setSelectedAsset: (asset: AssetInfo | null) => void;
    assetList: AssetInfo[];
    setAssetList: (assets: AssetInfo[]) => void;
    selectedType: string;
    setSelectedType: (type: string) => void;
    searchKeyword: string;
    setSearchKeyword: (keyword: string) => void;
    klineData: StockKlineData | null;
    setKlineData: (data: StockKlineData | null) => void;
    klineType: KlineType;
    setKlineType: (type: KlineType) => void;
    duration: string;
    setDuration: (duration: string) => void;
    selectedIndicators: string[];
    setSelectedIndicators: (indicators: string[]) => void;
    isLoading: boolean;
    setIsLoading: (loading: boolean) => void;
}

// 创建上下文
export const AssetSelectionContext = createContext<AssetSelectionContext | null>(null); 