import { Box, Typography } from '@mui/material';
import { useCallback, useRef } from 'react';
import { useDropzone } from 'react-dropzone';
import { showFileExceededNotification } from './FileExceedNotification';

const Dropzone = ({ uploadFile, ...props }) => {
  const inputRef = useRef(null);
  const resetInput = () => {
    if (inputRef.current) {
      inputRef.current.value = '';
    }
  };
  const onDrop = useCallback(
    (acceptedFile) => {
      uploadFile(acceptedFile[0]);
      resetInput();
    },
    [uploadFile]
  );
  const onDropRejected = useCallback((rejectedFiles) => {
    if (rejectedFiles.length > 0) {
      showFileExceededNotification(1, rejectedFiles.length);
      resetInput();
    }
  }, []);
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    maxFiles: 1,
    onDrop: onDrop,
    onDropRejected: onDropRejected
  });

  return (
    <Box
      className={isDragActive ? 'dragActiveBorder' : 'dragDisabledBorder'}
      display={'flex'}
      flexDirection={'column'}
      alignItems={'center'}
      justifyContent={'center'}
      backgroundColor={'rgba(118, 171, 174, 0.1)'}
      height={'164px'}
      width={'100%'}
      boxSizing={'border-box'}
      cursor={'pointer'}
      padding={'24px 48px'}
      sx={{
        cursor: 'pointer'
      }}
      {...getRootProps()}
      {...props}>
      <input {...getInputProps()} ref={inputRef} style={{ display: 'none', cursor: 'pointer' }} />
      <Box
        display="flex"
        flexDirection={'column'}
        alignItems="center"
        justifyContent="center"
        gap={'5px'}>
        <Typography variant={'body1'} fontWeight={700} fontSize={'16px'} color={'#707070'}>
          {`Browse or drag & drop it here`}
        </Typography>
        <Typography variant={'body1'} fontWeight={400} fontSize={'14px'} color={'#707070'}>
          {`DICOM format only`}
        </Typography>
      </Box>
    </Box>
  );
};

export default Dropzone;
