import { Box, Checkbox, Typography } from '@mui/material';

const CheckboxField = ({ label, isChecked, handleCheck }) => {
  const handleChange = (event) => {
    handleCheck(event.target.checked);
  };
  return (
    <Box display={'flex'} alignItems={'center'} flexDirection={'row'} gap={'10px'}>
      <Checkbox
        checked={isChecked}
        onChange={handleChange}
        disableRipple
        sx={{
          padding: 0,
          margin: 0,
          color: '#76ABAE',
          '&.Mui-checked': {
            color: '#76ABAE'
          }
        }}
      />
      <Typography fontFamily={'Poppins'} fontSize={'16px'} fontWeight={500} fontColor={'#333'}>
        {label}
      </Typography>
    </Box>
  );
};

export default CheckboxField;
