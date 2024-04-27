import React from 'react';
import ViewerPage from '../pages/ViewerPage/ViewerPage';
import MainLayout from '../layout/MainLayout';

const MainRoutes = {
  path: '/',
  children: [
    {
      path: '/',
      element: (
        <MainLayout>
          <ViewerPage />
        </MainLayout>
      )
    }
  ]
};

export default MainRoutes;
