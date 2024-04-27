import { Box } from '@mui/system';
import { IconButton, Typography } from '@mui/material';
import { ReactComponent as Fail } from '../../assets/error_circle_icon.svg';
import { ReactComponent as CloseIcon } from '../../assets/close_icon.svg';
import { Store } from 'react-notifications-component';
import { v4 as uuidv4 } from 'uuid';
import React from 'react';

const FileExceededNotification = ({ allowed, provided, closeNotification }) => {
  return (
    <Box
      display={'flex'}
      width={'552px'}
      boxSizing={'border-box'}
      padding={'8px 16px'}
      backgroundColor={'#2E2E2E'}
      borderRadius={'6px'}>
      <Box
        display={'flex'}
        flexDirection={'row'}
        alignItems={'center'}
        justifyContent={'space-between'}
        width={'100%'}>
        <Box display={'flex'} flexDirection={'row'} alignItems={'center'} gap={'16px'}>
          <Fail style={{ width: '24px', height: '24px' }} />
          <Box
            display={'flex'}
            flexDirection={'column'}
            alignItems={'flex-start'}
            justifyContent={'space-between'}>
            <Typography fontSize={'14px'} fontWeight={600} color={'#FFF'}>
              {'File limit exceeded'}
            </Typography>
            <Typography fontSize={'14px'} fontWeight={400} color={'#FFF'}>
              {`File limit is ${allowed} but you provided ${provided} files. Please try again.`}
            </Typography>
          </Box>
        </Box>
        <IconButton onClick={closeNotification}>
          <CloseIcon style={{ width: '16px', height: '16px' }} />
        </IconButton>
      </Box>
    </Box>
  );
};

export const showFileExceededNotification = (allowed, provided) => {
  const id = uuidv4();
  Store.addNotification({
    id: id,
    insert: 'top',
    container: 'top-center',
    animationOut: ['animate__animated', 'animate__zoomOut'],
    dismiss: {
      duration: 3000
    },
    width: 552,
    content: (
      <FileExceededNotification
        allowed={allowed}
        provided={provided}
        closeNotification={() => hideFileExceededNotification(id)}
      />
    )
  });
};

const hideFileExceededNotification = (id) => {
  Store.removeNotification(id);
};
