import { Button } from '@mui/material';
import React from 'react';

const PrimaryButton = ({ label, disabled = false, handleClick, ...props }) => {
  return (
    <Button
      onClick={
        disabled
          ? (e) => {
              e.preventDefault();
            }
          : handleClick
      }
      disabled={disabled}
      disableRipple
      sx={{
        textTransform: 'none',
        width: '256px',
        height: '56px',
        padding: '8px 16px',
        color: '#FFF',
        borderRadius: '6px',
        backgroundColor: disabled ? '#D9D9D9' : '#76ABAE',
        '&:hover': {
          backgroundColor: '#91BCBE',
          boxShadow: 'none'
        },
        '&.Mui-disabled': {
          backgroundColor: '#EFF5F5',
          color: '#333',
          boxShadow: 'none'
        },
        ...props
      }}>
      {label}
    </Button>
  );
};

export default PrimaryButton;
