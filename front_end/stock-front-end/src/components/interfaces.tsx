import React, { createContext } from "react";

export interface StockContextInterface {

  kChartInfo: KChartInfo | undefined;
  duration:string;
  setDuration:React.Dispatch<React.SetStateAction<string>>;
  degree: number;
  setDegree:React.Dispatch<React.SetStateAction<number>>;
  threshold_arg: number;
  setThreshold_arg:React.Dispatch<React.SetStateAction<number>>;
  stockInfoA:StockInfo;
  setStockInfoA:React.Dispatch<React.SetStateAction<StockInfo>>;
  stockInfoB:StockInfo;
  setStockInfoB:React.Dispatch<React.SetStateAction<StockInfo>>;
  showRatio:boolean;
  setShowRatio:React.Dispatch<React.SetStateAction<boolean>>;
  showDelta:boolean;
  setShowDelta:React.Dispatch<React.SetStateAction<boolean>>;

}

export interface UserOptionInfo{
  duration:string;
  degree:number;
  // threshold_arg:number;
  code_a:string;
  code_b:string;
}

export interface StockInfo {
  code: string;
  name: string;
  type: string;
}

export interface KChartInfo {
  ratio: number[];
  dates: string[];
  // outlier_date_splitters:string[];
  // colors:string[];
  close_a:number[];
  close_b:number[];
  fitting_line:number[];
  delta:number[];
  threshold:number;
}

export const StockContext = createContext<StockContextInterface | undefined>(
  undefined
);
