import { Box, Typography } from '@mui/material';
import React from 'react';
import { ReactComponent as WarningIcon } from '../../assets/error_circle_icon.svg';

const InferenceInfo = ({ inference }) => {
  const ValueText = ({ children, severity = false }) => {
    let color = '#EEE';
    if (severity) {
      if (children === 'BENIGN') {
        color = '#F8CE46';
      } else if (children === 'MALIGNANT') {
        color = '#CC2E2E';
      } else if (children === 'NORMAL') {
        color = '#76ABAE';
      }
    }
    const textStyles = {
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      color: color,
      fontWeight: 600,
      fontSize: '20px'
    };

    return <Typography sx={textStyles}>{children}</Typography>;
  };

  const LabelText = ({ children }) => {
    const textStyles = {
      whiteSpace: 'nowrap',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      color: '#707070',
      fontWeight: 600,
      fontSize: '20px'
    };

    return <Typography sx={textStyles}>{children}</Typography>;
  };
  console.log(inference);

  return (
    <Box
      display={'flex'}
      flexDirection={'column'}
      alignItems={'center'}
      width={'600px'}
      boxSizing={'border-box'}
      height={'auto'}
      margin={'50px'}
      padding={'15px'}
      borderRadius={'8px'}
      gap={'15px'}
      backgroundColor={'#31363F'}>
      <Box
        display={'flex'}
        flexDirection={'row'}
        alignItems={'center'}
        width={'100%'}
        justifyContent={'space-between'}
        sx={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
        <Box display={'flex'} flexDirection={'column'} alignItems={'flex-start'}>
          <LabelText>Modality</LabelText>
          <ValueText>{inference.modality === 'MG' ? 'Mammography' : 'Ultrasound'}</ValueText>
        </Box>
        <Box
          display={'flex'}
          flexDirection={'column'}
          alignItems={'center'}
          justifyContent={'center'}>
          <LabelText>Severity</LabelText>
          <ValueText severity={true}>
            {inference.severity ?? inference.severityMask ?? '-'}
          </ValueText>
        </Box>
        <Box
          display={'flex'}
          flexDirection={'column'}
          alignItems={'center'}
          justifyContent={'center'}>
          <LabelText>Q-ty of lesions</LabelText>
          <ValueText>{inference?.mask?.numberOfLesions ?? '-'}</ValueText>
        </Box>
      </Box>
      <Box width={'100%'} display={'flex'} flexDirection={'row'} alignItems={'center'} gap={'8px'}>
        <WarningIcon width={'24px'} height={'24px'} />
        <Typography color={'#F8CE46'} fontSize={'11px'} textAlign={'justify'}>
          {/* eslint-disable-next-line react/no-unescaped-entities */}
          This service provides a second opinion only. It's intended to complement professional
          medical advice, not replace it.
        </Typography>
      </Box>
    </Box>
  );
};

export default InferenceInfo;
