import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { RootLayout } from './components/RootLayout';
import { HomePage } from './components/HomePage';
import { ClassificationPage } from './components/ClassificationPage';
import { RegressionPage } from './components/RegressionPage';
import { ImageRecognitionPage } from './components/ImageRecognitionPage';
import { NeuralNetworkVisualizerPage } from './components/NeuralNetworkVisualizerPage';
import { AboutPage } from './components/AboutPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RootLayout />}>
          <Route index element={<HomePage />} />
          <Route path="classification" element={<ClassificationPage />} />
          <Route path="regression" element={<RegressionPage />} />
          <Route path="image-recognition" element={<ImageRecognitionPage />} />
          <Route path="neural-network" element={<NeuralNetworkVisualizerPage />} />
          <Route path="about" element={<AboutPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}