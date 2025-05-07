import {useQuery, UseQueryOptions, UseQueryResult} from '@tanstack/react-query';
import {useNavigate} from 'react-router-dom';

// 接口定义
interface TrendData {
    time: string;
    current_price: number;
    change_percent: number;
}

interface AssetPair {
    code_a: string;
    name_a: string;
    code_b: string;
    name_b: string;
    current_ratio: number;
    change_ratio: number;
    trends_a: TrendData[];
    trends_b: TrendData[];
    signals: any[];
    latest_date: string;
    recommendation?: string;
}

interface Asset {
    code: string;
    name: string;
    current_price: number;
    price_change: number;
    trends: TrendData[];
}

export interface DashboardData {
    recent_pairs: AssetPair[];
    favorite_assets: Asset[];
}

/**
 * 获取仪表盘数据的API函数
 */
export const fetchDashboardData = async (): Promise<DashboardData> => {
    const response = await fetch('http://localhost:8000/api/dashboard', {
        method: 'GET',
        credentials: 'include',
    });

    if (!response.ok) {
        // 处理HTTP错误
        if (response.status === 401) {
            throw new Error('401: 会话已过期');
        }
        throw new Error(`${response.status}: 获取仪表盘数据失败`);
    }

    const result = await response.json();
    if (result.status !== 'success') {
        throw new Error(result.message || '获取仪表盘数据失败');
    }

    return result.data;
};

// 自定义Hook参数接口
interface UseDashboardDataOptions {
    onSuccess?: (data: DashboardData) => void;
    onError?: (error: Error) => void;
    refetchInterval?: number;
    enabled?: boolean;
}

/**
 * 自定义Hook：获取仪表盘数据并处理缓存
 * @param isLoggedIn 用户是否已登录
 * @param options 额外的查询选项
 * @returns 查询结果
 */
export const useDashboardData = (
    isLoggedIn: boolean,
    options?: UseDashboardDataOptions
): UseQueryResult<DashboardData, Error> => {
    const navigate = useNavigate();

    // 提取选项
    const {onSuccess, onError, refetchInterval = 60 * 1000, enabled = true} = options || {};

    return useQuery<DashboardData, Error>({
        queryKey: ['dashboardData'],
        queryFn: fetchDashboardData,
        // 默认配置
        staleTime: 60 * 1000, // 1分钟内数据不会变stale
        refetchInterval,
        enabled: isLoggedIn && enabled,
        onSuccess,
        onError: (error: Error) => {
            // 默认的错误处理
            if (error.message.includes('401')) {
                navigate('/login');
            }
            console.error('获取仪表盘数据失败:', error);

            // 如果提供了自定义错误处理，也调用它
            if (onError) {
                onError(error);
            }
        }
    });
}; 