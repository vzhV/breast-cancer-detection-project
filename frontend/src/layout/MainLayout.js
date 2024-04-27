import { Box } from '@mui/material';
import { useAutoAnimate } from '@formkit/auto-animate/react';

const MainLayout = ({ children }) => {
  const [block] = useAutoAnimate();
  return (
    <Box
      ref={block}
      display={'flex'}
      flexDirection={'column'}
      alignItems={'center'}
      justifyContent={'center'}
      height={'100%'}
      width={'100%'}
      minHeight={'100vh'}
      minWidth={'100vw'}>
      {children}
    </Box>
  );
};

export default MainLayout;
