
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import Radio from "@mui/material/Radio";
import FormControl from "@mui/material/FormControl";
import React from "react";
import {Stack} from "@mui/material";
import Typography from "@mui/material/Typography";

interface RadioSelectorProps {
    title: string;
    available_options: string[];
    option_labels: string[];
    current_option: string; // New prop for controlled value
    set_current_option: React.Dispatch<React.SetStateAction<string>>;
}

export const RadioSelector: React.FC<RadioSelectorProps> = ({
                                                                title,
                                                                available_options,
                                                                option_labels,
                                                                current_option,
                                                                set_current_option,
                                                            }) => {
    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        set_current_option(event.target.value);
    };

    return (
        <Stack direction="row" spacing={2}>
            <Typography> {title}</Typography>
            <FormControl>

                <RadioGroup
                    row
                    aria-labelledby="demo-radio-buttons-group-label"
                    value={current_option} // Controlled value
                    name="radio-buttons-group"
                    onChange={handleChange} // Attach handleChange here
                >
                    {available_options.map((option_value, index) => (
                        <FormControlLabel
                            key={option_value}
                            value={option_value}
                            control={<Radio size="small" />}
                            label={option_labels[index]} // Use corresponding label
                        />
                    ))}
                </RadioGroup>
            </FormControl>
        </Stack>
    );
};