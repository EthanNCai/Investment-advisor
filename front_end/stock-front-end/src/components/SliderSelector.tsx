import React from "react";
import Slider from "@mui/material/Slider";
import { Stack } from "@mui/material";
import Typography from "@mui/material/Typography";

interface SliderSelectorProps {
    title: string;
    current_value: number; // Controlled value
    set_current_value: React.Dispatch<React.SetStateAction<number>>;
    min:number;
    max:number;
    step:number;
}

export const SliderSelector: React.FC<SliderSelectorProps> = ({
                                                                  title,
                                                                  current_value,
                                                                  set_current_value,
                                                              min,
    max,step
                                                              }) => {
    const handleChange = (_: Event, newValue: number | number[]) => {
        set_current_value(newValue as number);
    };

    return (
        <Stack direction="row" spacing={2} alignItems="center">
            <Typography>{title}</Typography>
            <Slider
                valueLabelDisplay="auto"
                value={current_value}
                onChange={handleChange}
                aria-labelledby="continuous-slider"
                step={step}
                marks
                min={min}
                max={max}

            />
        </Stack>
    );
};