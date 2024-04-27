import { Button } from '@mui/material';
import React from 'react';

const SecondaryButton = ({ label, disabled = false, handleClick, ...props }) => {
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
        color: '#31363F',
        borderRadius: '6px',
        textTransform: 'none',
        padding: '8px 16px',
        backgroundColor: 'transparent',
        borderColor: '#31363F',
        borderWidth: '1px',
        borderStyle: 'solid',
        '&:hover': {
          boxShadow: 'none'
        },
        '&:active': {
          backgroundColor: '#31363F',
          borderColor: '#31363F',
          color: '#FFF'
        },
        '&.Mui-disabled': {
          color: '#31363F',
          borderColor: '#31363F',
          boxShadow: 'none'
        },
        ...props
      }}>
      {label}
    </Button>
  );
};

export default SecondaryButton;
