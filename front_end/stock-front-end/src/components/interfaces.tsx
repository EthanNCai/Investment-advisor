import React, { createContext } from "react";

export interface StockContextInterface {
  codeA: string;
  setCodeA: React.Dispatch<React.SetStateAction<string>>;
  codeB: string;
  setCodeB: React.Dispatch<React.SetStateAction<string>>;
  kChartInfo: KChartInfo | undefined;
  duration:string;
  setDuration:React.Dispatch<React.SetStateAction<string>>;
  degree: number;
  setDegree:React.Dispatch<React.SetStateAction<number>>;
  threshold: number;
  setThreshold:React.Dispatch<React.SetStateAction<number>>;
}

export interface UserOptionInfo{
  duration:string;
  degree:number;
  threshold:number;
  code_a:string;
  code_b:string;
}

export interface KChartInfo {
  ratio: number[];
  dates: string[];
  outlier_date_splitters:string[];
  colors:string[];
  close_a:number[];
  close_b:number[];
  fitting_line:number[];
  delta:number[];
  thres:number;
}

export const StockContext = createContext<StockContextInterface | undefined>(
  undefined
);
