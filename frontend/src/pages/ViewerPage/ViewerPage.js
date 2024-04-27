/* eslint-disable */
import React, { useEffect, useState } from 'react';
import { useAutoAnimate } from '@formkit/auto-animate/react';
import { Box, IconButton, Typography } from '@mui/material';
import PrimaryButton from '../../components/Button/PrimaryButton';
import Dropzone from '../../components/Dropzone/Dropzone';
import CheckboxField from '../../components/Checkbox/CheckboxField';
import axios from 'axios';
import InferenceInfo from '../../components/InferenceInfo/InferenceInfo';
import DicomViewer from '../../components/DicomViewer/DicomViewer';
import { ReactComponent as HomeIcon } from '../../assets/home.svg';
import { ReactComponent as MagnifierONIcon } from '../../assets/magnifier_on.svg';
import { ReactComponent as MagnifierOFFIcon } from '../../assets/magnifier_off.svg';
import { ReactComponent as SegmentationONIcon } from '../../assets/show_segmentation.svg';
import { ReactComponent as SegmentationOFFIcon } from '../../assets/hide_segmentation.svg';
import { ReactComponent as PlusIcon } from '../../assets/plus.svg';
import { ReactComponent as MinusIcon } from '../../assets/minus.svg';

const ViewerPage = () => {
  const [block] = useAutoAnimate();
  const [stage, setStage] = useState(1);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [selectedService, setSelectedService] = useState('MG');
  const [selectedTask, setSelectedTask] = useState('CLASSIFICATION');
  const [inference, setInference] = useState({});
  const [showViewer, setShowViewer] = useState(false);
  const [showSegmentation, setShowSegmentation] = useState(true);
  const [showMagnifier, setShowMagnifier] = useState(false);
  const [magnifierValue, setMagnifierValue] = useState(2.0);

  const handleShowSegmentation = () => {
    setShowSegmentation((prev) => !prev);
  };

  const handleShowMagnifier = () => {
    setShowMagnifier((prev) => !prev);
  };

  const handleIncreaseMagnifierValue = () => {
    setMagnifierValue((prev) => (prev < 3.5 ? prev + 0.5 : prev));
  };

  const handleDecreaseMagnifierValue = () => {
    setMagnifierValue((prev) => (prev > 1.5 ? prev - 0.5 : prev));
  };

  const handleSetSelectedTask = (value) => {
    setSelectedTask((prev) => (prev === value ? prev : value));
  };

  const handleSelectedServiceChanged = (value) => {
    setSelectedService((prev) => (prev === value ? prev : value));
  };

  const handleFileUpload = (file) => {
    console.log(file.name);
    setUploadedFile(file);
  };

  const handleStateChange = () => {
    setStage((prev) => (prev < 3 ? prev + 1 : prev));
  };

  const resetToFirstPage = () => {
    setStage(1);
    setUploadedFile(null);
    setSelectedService('MG');
    setSelectedTask('CLASSIFICATION');
    setInference({});
    setShowViewer(false);
    setShowMagnifier(false);
    setShowSegmentation(true);
    setMagnifierValue(2.0);
  };

  useEffect(() => {
    if (inference && Object.keys(inference).length > 0) {
      setTimeout(() => {
        setShowViewer(true);
      },2000)
    }
  }, [inference]);

  const handleUpload = () => {
    const formData = new FormData();
    const tempTask =
      selectedTask === 'ALL'
        ? selectedService === 'US'
          ? 'CLASSIFICATION_OVERLAID'
          : 'ALL'
        : selectedTask;
    formData.append('file', uploadedFile);
    formData.append('action', tempTask);
    handleStateChange();

    axios
      .post(
        `http://127.0.0.1:8000/${selectedService === 'MG' ? 'mammogram' : 'ultrasound'}`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' }
        }
      )
      .then((res) => {
        console.log(res.data);
        setInference(res.data);
      })
      .catch((err) => console.error(err));
  };

  return (
    <>
      {stage === 1 && (
        <Box
          ref={block}
          display={'flex'}
          flexDirection={'column'}
          alignItems={'center'}
          justifyContent={'space-between'}
          width={'600px'}
          boxSizing={'border-box'}
          height={'800px'}
          margin={'50px'}
          padding={'30px'}
          borderRadius={'16px'}
          gap={'40px'}
          backgroundColor={'#EEEEEE'}>
          <Box
            display={'flex'}
            flexDirection={'column'}
            gap={'40px'}
            alignItems={'center'}
            justifyContent={'center'}>
            <Typography
              fontSize={'32px'}
              fontWeight={550}
              fontColor={'#31363F'}
              fontFamily={'Poppins'}>
              Breast Cancer Detector
            </Typography>
            <Typography
              fontSize={'16px'}
              fontWeight={500}
              fontColor={'#333'}
              fontFamily={'Poppins'}
              textAlign={'justify'}>
              Discover peace of mind with our Breast Cancer Detector, your trusted second opinion
              service for patients and doctors alike. By minimizing false positives and identifying
              symptoms early, we enhance your chances for effective treatment and cure. Experience
              the power of precision and early intervention with us today.
            </Typography>
          </Box>
          <PrimaryButton label={`Proceed to upload`} handleClick={handleStateChange} />
          <Typography
            fontSize={'11px'}
            fontWeight={400}
            fontColor={'#707070'}
            fontFamily={'Poppins'}
            textAlign={'center'}>
            We prioritize the security of your information. All our services are designed to ensure
            strict confidentiality, with no personal data stored.
          </Typography>
        </Box>
      )}
      {stage === 2 && (
        <Box
          display={'flex'}
          flexDirection={'column'}
          alignItems={'center'}
          justifyContent={'space-between'}
          width={'600px'}
          boxSizing={'border-box'}
          height={'auto'}
          margin={'50px'}
          padding={'30px'}
          borderRadius={'16px'}
          gap={'25px'}
          backgroundColor={'#EEEEEE'}>
          {!uploadedFile && <Dropzone uploadFile={handleFileUpload} />}
          {uploadedFile && (
            <>
              <Typography
                fontSize={'24px'}
                fontWeight={700}
                fontColor={'#333'}
                textAlign={'justify'}>
                Select uploaded modality
              </Typography>
              <Box
                display={'flex'}
                width={'100%'}
                flexDirection={'row'}
                alignItems={'center'}
                gap={'100px'}
                justifyContent={'center'}>
                <CheckboxField
                  label={'Mammography'}
                  handleCheck={() => handleSelectedServiceChanged('MG')}
                  isChecked={selectedService === 'MG'}
                />
                <CheckboxField
                  label={'Ultrasound'}
                  handleCheck={() => handleSelectedServiceChanged('US')}
                  isChecked={selectedService === 'US'}
                />
              </Box>
              <Box width={'100%'} height={'1px'} backgroundColor={'#76ABAE'} />
              <Typography
                fontSize={'24px'}
                fontWeight={700}
                fontColor={'#333'}
                textAlign={'justify'}>
                Select desired task
              </Typography>
              <Box
                display={'flex'}
                width={'100%'}
                flexDirection={'row'}
                alignItems={'center'}
                gap={'100px'}
                marginBottom={'15px'}
                justifyContent={'center'}>
                <CheckboxField
                  label={'Classification'}
                  handleCheck={() => handleSetSelectedTask('CLASSIFICATION')}
                  isChecked={selectedTask === 'CLASSIFICATION'}
                />
                <CheckboxField
                  label={'Segmentation'}
                  handleCheck={() => handleSetSelectedTask('SEGMENTATION')}
                  isChecked={selectedTask === 'SEGMENTATION'}
                />
                <CheckboxField
                  label={'Both'}
                  handleCheck={() => handleSetSelectedTask('ALL')}
                  isChecked={selectedTask === 'ALL'}
                />
              </Box>
              <PrimaryButton label={`Upload`} handleClick={handleUpload} disabled={!uploadedFile} />
            </>
          )}
        </Box>
      )}
      {stage === 3 && (
        <>
          {showViewer && <InferenceInfo inference={inference} />}
          <DicomViewer
            file={uploadedFile}
            lesions={inference?.mask?.mask ?? []}
            showContent={showViewer}
            showMagnifier={showMagnifier}
            showSegmentation={showSegmentation}
            magnifierScale={magnifierValue}
          />
          {showViewer && (
            <>
              <Box display={'flex'} flexDirection={'row'} gap={'0px'} alignItems={'center'}>
                <Box
                  display={'flex'}
                  flexDirection={'row'}
                  alignItems={'center'}
                  justifyContent={'space-between'}
                  width={'250px'}
                  boxSizing={'border-box'}
                  height={'40px'}
                  margin={'50px'}
                  padding={'30px'}
                  borderRadius={'16px'}
                  gap={'25px'}
                  backgroundColor={'#31363F'}>
                  <IconButton onClick={resetToFirstPage}>
                    <HomeIcon style={{ width: '24px', height: '24px', fill: '#EEE' }} />
                  </IconButton>
                  <IconButton onClick={handleShowSegmentation}>
                    {showSegmentation ? (
                      <SegmentationOFFIcon
                        style={{ width: '24px', height: '24px', fill: '#EEE' }}
                      />
                    ) : (
                      <SegmentationONIcon style={{ width: '24px', height: '24px', fill: '#EEE' }} />
                    )}
                  </IconButton>
                  <IconButton onClick={handleShowMagnifier}>
                    {showMagnifier ? (
                      <MagnifierOFFIcon style={{ width: '24px', height: '24px', fill: '#EEE' }} />
                    ) : (
                      <MagnifierONIcon style={{ width: '24px', height: '24px', fill: '#EEE' }} />
                    )}
                  </IconButton>
                </Box>
                {showMagnifier && (
                  <Box
                    display={'flex'}
                    flexDirection={'row'}
                    alignItems={'center'}
                    justifyContent={'space-between'}
                    width={'250px'}
                    boxSizing={'border-box'}
                    height={'40px'}
                    margin={'50px'}
                    padding={'30px'}
                    borderRadius={'16px'}
                    gap={'25px'}
                    backgroundColor={'#31363F'}>
                    <IconButton onClick={handleIncreaseMagnifierValue}>
                      <PlusIcon style={{ width: '24px', height: '24px', fill: '#EEE' }} />
                    </IconButton>
                    <Typography fontSize={'16px'} fontWeight={500} color={'#EEE'}>
                      {magnifierValue}
                    </Typography>
                    <IconButton onClick={handleDecreaseMagnifierValue}>
                      <MinusIcon style={{ width: '24px', height: '24px', fill: '#EEE' }} />
                    </IconButton>
                  </Box>
                )}
              </Box>
            </>
          )}
        </>
      )}
    </>
  );
};

export default ViewerPage;
