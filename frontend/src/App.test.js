import { render, screen } from '@testing-library/react';
import App from './App';

test('renders ML Trainer header', () => {
  render(<App />);
  const headerElement = screen.getByText(/ML Trainer/i);
  expect(headerElement).toBeInTheDocument();
});
