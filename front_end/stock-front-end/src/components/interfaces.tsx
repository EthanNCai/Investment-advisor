import React, { createContext } from "react";

export interface StockContextInterface {
  codeA: string;
  setCodeA: React.Dispatch<React.SetStateAction<string>>;
  codeB: string;
  setCodeB: React.Dispatch<React.SetStateAction<string>>;
  kChartInfo: KChartInfo|undefined;
  userOptions: UserOptions;
}

export interface KChartInfo {
    ratio:number[];
    dates:string[];
    // marked_dates:string[];
    // colors:string[];
    // close_a
    // close_b

}

export interface UserOptions {
    ratio:number[];
    dates:string[];
    // marked_dates:string[];
    // colors:string[];

}

export const StockContext = createContext<StockContextInterface | undefined>(undefined);