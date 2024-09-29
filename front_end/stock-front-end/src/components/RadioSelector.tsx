import FormLabel from "@mui/material/FormLabel";
import RadioGroup from "@mui/material/RadioGroup";
import FormControlLabel from "@mui/material/FormControlLabel";
import Radio from "@mui/material/Radio";
import FormControl from "@mui/material/FormControl";

interface RadioSelectorProps {
  title: string;
  option_values: string[];
  option_labels: string[];
}

export const RadioSelector: React.FC<RadioSelectorProps> = ({title,option_values,option_labels}) =>{

    return(
    <FormControl>
      <FormLabel id="demo-radio-buttons-group-label">{title}</FormLabel>
      <RadioGroup
        row
        aria-labelledby="demo-radio-buttons-group-label"
        defaultValue={option_values[0]}
        name="radio-buttons-group"
      >
           {option_values.map((option_value, index) => (
            <FormControlLabel
            key={option_value}
            value={option_value}
            control={<Radio />}
            label={option_labels[index]} // 使用对应索引处的标签
          />
        ))}
      </RadioGroup>
    </FormControl>
    )
}