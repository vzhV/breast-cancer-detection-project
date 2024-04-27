import { useEffect, useState } from 'react';
import { Box, CircularProgress } from '@mui/material';
import { Stage, Layer, Image as KonvaImage, Line } from 'react-konva';

const DicomViewer = ({
  file,
  showSegmentation,
  showMagnifier,
  lesions,
  showContent,
  magnifierScale
}) => {
  const [magnifierVisible, setMagnifierVisible] = useState(false);
  const [magnifierPosition, setMagnifierPosition] = useState({ x: 0, y: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [imageObj, setImageObj] = useState(null);
  const [width, setWidth] = useState(0);
  const [height, setHeight] = useState(0);
  const maxDimension = 600; // Maximum size for the larger dimension

  useEffect(() => {
    setIsLoading(!showContent);
  }, [showContent]);

  useEffect(() => {
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const img = new window.Image();
        img.onload = () => {
          let scale = 1;
          let newWidth = img.width;
          let newHeight = img.height;
          // Adjust dimensions if the height exceeds the max allowed height
          if (img.height > maxDimension) {
            scale = maxDimension / img.height;
            newWidth = img.width * scale;
            newHeight = maxDimension;
          }
          setImageObj(img);
          setWidth(newWidth);
          setHeight(newHeight);

          if (showContent) {
            setIsLoading(false);
          }
        };
        img.src = e.target.result;
      };
      reader.readAsDataURL(file);
    }
  }, [file]);

  const handleMouseMove = (e) => {
    const stage = e.currentTarget;
    const point = stage.getPointerPosition();
    setMagnifierPosition(point);
  };

  const handleMouseOver = () => {
    if (showMagnifier) {
      setMagnifierVisible(true);
    }
  };

  const handleMouseOut = () => {
    setMagnifierVisible(false);
  };

  const drawLesions = (scale = 1, strokeWidth = 2) => {
    return lesions.map((lesion, index) => (
      <Line
        key={index}
        points={lesion.map((point) => [point.X * scale, point.Y * scale]).flat()}
        closed={true}
        stroke="red"
        strokeWidth={strokeWidth}
      />
    ));
  };

  return (
    <Box height={height} width={width} position="relative" border={'3px solid #31363F'}>
      {isLoading ? (
        <Box display="flex" justifyContent="center" alignItems="center" height="100%">
          <CircularProgress />
        </Box>
      ) : (
        <div>
          <Stage
            width={width}
            height={height}
            onMouseMove={handleMouseMove}
            onMouseOver={handleMouseOver}
            onMouseOut={handleMouseOut}>
            <Layer>
              <KonvaImage image={imageObj} width={width} height={height} />
              {showSegmentation && drawLesions(width / imageObj.width)}
            </Layer>
          </Stage>

          {magnifierVisible && (
            <div
              style={{
                position: 'absolute',
                backgroundColor: '#222831',
                border: '1px solid white',
                top: magnifierPosition.y - 100,
                left: magnifierPosition.x - 100,
                width: 200,
                height: 200,
                overflow: 'hidden',
                borderRadius: '50%',
                display: 'block',
                pointerEvents: 'none',
                zIndex: 1000
              }}>
              <Stage
                width={200}
                height={200}
                scaleX={magnifierScale}
                scaleY={magnifierScale}
                x={-(magnifierPosition.x * magnifierScale) + 100}
                y={-(magnifierPosition.y * magnifierScale) + 100}>
                <Layer>
                  <KonvaImage image={imageObj} width={width} height={height} />
                  {showSegmentation && drawLesions(width / imageObj.width, 1)}
                </Layer>
              </Stage>
            </div>
          )}
        </div>
      )}
    </Box>
  );
};

export default DicomViewer;
