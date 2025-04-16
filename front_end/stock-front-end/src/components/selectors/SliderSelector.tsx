import React, {useEffect, useState} from "react";
import Box from '@mui/material/Box';
import Stack from "@mui/material/Stack";
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';

interface SliderSelectorProps {
    title?: string;
    current_value?: number;
    set_current_value?: React.Dispatch<React.SetStateAction<number>>;
    value?: number;
    onChange?: (value: number) => void;
    defaultValue?: number;
    min: number;
    max: number;
    step: number;
}

export const SliderSelector: React.FC<SliderSelectorProps> = ({
    title,
    current_value,
    set_current_value,
    value,
    onChange,
    defaultValue,
    min,
    max,
    step
}) => {
    // 兼容新旧两种使用方式
    const isControlled = value !== undefined && onChange !== undefined;
    const initialValue = isControlled ? value : (current_value || defaultValue || min);

    const [tempValue, setTempValue] = useState<number>(initialValue);

    // 当外部value变化时更新内部状态
    useEffect(() => {
        if (isControlled && value !== tempValue) {
            setTempValue(value);
        } else if (!isControlled && current_value !== tempValue) {
            setTempValue(current_value || min);
        }
    }, [value, current_value, isControlled, min, tempValue]);

    const handleChange = (_: Event, newValue: number | number[]) => {
        const newVal = newValue as number;
        setTempValue(newVal);
        if (isControlled && onChange) {
            onChange(newVal);
        }
    };

    const handleCommit = () => {
        if (!isControlled && set_current_value) {
            set_current_value(tempValue);
        }
    };

    return (
        <Stack direction="column" spacing={0}>
            {title && <Typography>{title}</Typography>}
            <Slider
                valueLabelDisplay="auto"
                value={tempValue}
                onChange={handleChange}
                onChangeCommitted={handleCommit}
                aria-labelledby="continuous-slider"
                step={step}
                marks
                min={min}
                max={max}
            />
        </Stack>
    );
};