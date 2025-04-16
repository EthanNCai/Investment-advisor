import React from "react";
import { Stack } from "@mui/material";
import Typography from "@mui/material/Typography";
import Checkbox from "@mui/material/Checkbox";

interface CheckSelectorProps {
    title: string;
    value: boolean; // Controlled value
    set_value: React.Dispatch<React.SetStateAction<boolean>>;
}

export const CheckSelector: React.FC<CheckSelectorProps> = ({
                                                                title,
                                                                value,
                                                                set_value,
                                                            }) => {
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = event.target.checked ? true : false; // Example: 1 for checked, 0 for unchecked
        set_value(newValue);
    };

    return (
        <Stack direction="row" spacing={0} alignItems="center">
            <Typography>{title}</Typography>
            <Checkbox
                checked={value === true}
                onChange={handleChange}
                color="primary"
            />
        </Stack>
    );
};