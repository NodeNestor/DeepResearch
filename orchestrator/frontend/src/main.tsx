import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import App from './App';

// Default to dark mode
if (!localStorage.getItem('dr-theme')) {
  document.documentElement.classList.add('dark');
} else {
  const theme = JSON.parse(localStorage.getItem('dr-theme')!);
  if (theme === 'dark') document.documentElement.classList.add('dark');
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
