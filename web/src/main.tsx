import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';

// Self-hosted, so nothing is fetched from a font CDN at runtime. Literata is a
// typeface made for reading books, and carries everything that comes from the
// book itself; Inter carries the interface around it.
//
// The opsz build adds the optical size axis to the weight axis, which is what
// keeps Literata readable at 15px in a chapter list and still elegant at 56px.
import '@fontsource-variable/literata/opsz.css';
import '@fontsource-variable/inter';

import './styles/tokens.css';
import './styles/base.css';
import App from './App';

const root = document.getElementById('root');
if (!root) throw new Error('No #root element to mount into.');

createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
