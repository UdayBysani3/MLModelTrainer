import React, { useState, useEffect } from 'react'; 
import PropTypes from 'prop-types';
import {
  AppBar,
  Toolbar,
  Container,
  Typography,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  CssBaseline,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableRow,
  TableContainer,
  Paper as MuiPaper,
  Dialog,
  IconButton,
  GlobalStyles,
  TextField
} from '@mui/material';
import { createTheme, ThemeProvider } from '@mui/material/styles';

function CustomCloseIcon(props) {
  return (
    <svg
      {...props}
      viewBox="0 0 24 24"
      width="24"
      height="24"
      fill="currentColor"
      aria-label="Close dialog"
      style={{ transition: 'transform 0.3s' }}
      onMouseEnter={(e) => (e.currentTarget.style.transform = 'rotate(90deg)')}
      onMouseLeave={(e) => (e.currentTarget.style.transform = 'rotate(0deg)')}
    >
      <path d="M19,6.41L17.59,5L12,10.59L6.41,5L5,6.41L10.59,12L5,17.59L6.41,19L12,13.41L17.59,19L19,17.59L13.41,12Z" />
    </svg>
  );
}

CustomCloseIcon.propTypes = {
  style: PropTypes.object
};

function RenderMetrics({ metrics }) {
  const classificationOrder = [
    'accuracy', 'precision', 'recall', 'f1_score',
    'confusion_matrix', 'classification_report', 'feature_importances'
  ];
  const regressionOrder = [
    'mse', 'r2_score', 'mean_absolute_error',
    'coefficients', 'tree_plot', 'feature_importances'
  ];

  const isClassification = metrics.hasOwnProperty('accuracy');
  let orderedKeys = [];
  if (isClassification) {
    for (let key of classificationOrder) {
      if (metrics.hasOwnProperty(key)) {
        orderedKeys.push(key);
      }
    }
    for (let key in metrics) {
      if (!orderedKeys.includes(key)) {
        orderedKeys.push(key);
      }
    }
  } else {
    for (let key of regressionOrder) {
      if (metrics.hasOwnProperty(key)) {
        orderedKeys.push(key);
      }
    }
    for (let key in metrics) {
      if (!orderedKeys.includes(key)) {
        orderedKeys.push(key);
      }
    }
  }

  return (
    <Box
      sx={{
        mt: 3,
        p: 2,
        backgroundColor: 'rgba(255,255,255,0.9)',
        borderRadius: '8px',
        boxShadow: '0px 2px 8px rgba(0,0,0,0.15)',
        animation: 'fadeIn 1s ease-out'
      }}
    >
      <Typography
        variant="h6"
        sx={{
          color: '#1976d2',
          mb: 2,
          fontWeight: 'bold',
          textAlign: 'center',
          borderBottom: '2px solid #1976d2',
          pb: 0.5
        }}
      >
        Metrics
      </Typography>
      {orderedKeys.map((key) => {
        const value = metrics[key];
        if (key === 'confusion_matrix' && Array.isArray(value)) {
          return (
            <Box key={key} sx={{ mb: 3, animation: 'scaleUp 0.5s ease-out' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#333', mb: 1 }}>
                Confusion Matrix:
              </Typography>
              <TableContainer component={MuiPaper} sx={{ maxWidth: 400, mx: 'auto' }}>
                <Table size="small">
                  <TableBody>
                    {value.map((row, rowIndex) => (
                      <TableRow key={rowIndex}>
                        {row.map((cell, cellIndex) => (
                          <TableCell
                            key={cellIndex}
                            sx={{
                              border: '1px solid #ddd',
                              textAlign: 'center',
                              padding: '6px',
                              transition: 'background-color 0.3s, transform 0.3s',
                              '&:hover': { backgroundColor: '#f0f0f0', transform: 'scale(1.05)' }
                            }}
                          >
                            {cell}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Box>
          );
        }
        else if (key === 'classification_report') {
          return (
            <Box key={key} sx={{ mb: 3, animation: 'fadeIn 0.8s ease-out' }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 'bold', color: '#333', mb: 1 }}>
                Classification Report:
              </Typography>
              <Box
                sx={{
                  backgroundColor: '#eef6fc',
                  p: 2,
                  borderRadius: '4px',
                  overflowX: 'auto'
                }}
              >
                <Typography
                  variant="body2"
                  component="pre"
                  sx={{
                    m: 0,
                    whiteSpace: 'pre',
                    fontFamily: 'monospace',
                    fontSize: '0.9rem'
                  }}
                >
                  {value}
                </Typography>
              </Box>
            </Box>
          );
        }
        else {
          return (
            <Typography
              key={key}
              variant="body2"
              sx={{
                mb: 2,
                backgroundColor: '#fff',
                p: 1,
                borderRadius: '4px',
                boxShadow: 'inset 0 0 4px rgba(0,0,0,0.1)',
                fontWeight: 500,
                transition: 'transform 0.3s',
                '&:hover': { transform: 'translateX(5px)' }
              }}
            >
              <span style={{ fontWeight: 'bold', color: '#333' }}>
                {`${key.charAt(0).toUpperCase() + key.slice(1)}:`}
              </span>{' '}
              {Array.isArray(value) ? JSON.stringify(value) : value}
            </Typography>
          );
        }
      })}
    </Box>
  );
}

RenderMetrics.propTypes = {
  metrics: PropTypes.object.isRequired
};

function App() {
  const [file, setFile] = useState(null);
  const [algorithm, setAlgorithm] = useState('');
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [problemType, setProblemType] = useState(null);
  const [availableAlgorithms, setAvailableAlgorithms] = useState([]);
  const [analyzed, setAnalyzed] = useState(false);
  const [showOverviewPreview, setShowOverviewPreview] = useState(false);
  const [rowCount, setRowCount] = useState(0);
  const [independentVars, setIndependentVars] = useState([]);
  const [predictionInputs, setPredictionInputs] = useState({});
  const [predictionResult, setPredictionResult] = useState(null);
  const [datasetType, setDatasetType] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setAnalyzed(false);
    setProblemType(null);
    setAlgorithm('');
    setMetrics(null);
    setAvailableAlgorithms([]);
    setIndependentVars([]);
    setPredictionInputs({});
    setPredictionResult(null);
    setDatasetType(null);
    setErrorMessage(null);

    if (selectedFile && selectedFile.type === 'text/csv') {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target.result;
        const rows = text.split('\n').filter(line => line.trim() !== '');
        const count = rows.length > 0 ? rows.length - 1 : 0;
        setRowCount(count);
      };
      reader.readAsText(selectedFile);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      alert('Please select a CSV file first!');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    try {
      setAnalyzing(true);
      const analyzeRes = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
      });
      if (!analyzeRes.ok) {
        const error = await analyzeRes.json();
        throw new Error(error.error || 'Analysis failed');
      }
      const analyzeResult = await analyzeRes.json();
      if (analyzeResult.error) throw new Error(analyzeResult.error);

      setProblemType(analyzeResult.problem_type);
      setAnalyzed(true);
      setDatasetType(analyzeResult.dataset_type);
      if (analyzeResult.dataset_type === 'mnist') {
        setIndependentVars(analyzeResult.columns.filter(col => col !== 'label'));
      } else {
        setIndependentVars(analyzeResult.columns.slice(0, -1));
      }
      if (analyzeResult.problem_type === 'classification') {
        setAvailableAlgorithms([
          'logistic_regression',
          'knn',
          'naive_bayes',
          'decision_tree',
          'svm'
        ]);
      } else {
        setAvailableAlgorithms([
          'linear_regression',
          'decision_tree',
          'svm'
        ]);
      }
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleSubmit = async () => {
    if (!file || !algorithm) {
      alert('Please select a file and an algorithm!');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);
    formData.append('algorithm', algorithm);
    formData.append('show_metrics', 'true');
    try {
      setLoading(true);
      const trainRes = await fetch('http://localhost:5000/train', {
        method: 'POST',
        body: formData,
      });
      if (!trainRes.ok) {
        const error = await trainRes.json();
        throw new Error(error.error || 'Training failed');
      }
      const trainResult = await trainRes.json();
      if (trainResult.error) throw new Error(trainResult.error);

      let metricsCopy = trainResult.metrics ? { ...trainResult.metrics } : null;
      let plotImage = null;

      if (metricsCopy && metricsCopy.plot) {
        plotImage = metricsCopy.plot;
        delete metricsCopy.plot;
      }
      setMetrics({ ...metricsCopy, plot: plotImage });
    } catch (err) {
      alert('Error: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (varName, value) => {
    setPredictionInputs(prev => ({ ...prev, [varName]: value }));
    setErrorMessage(null);
  };

  const handlePredict = async () => {
    if (Object.keys(predictionInputs).length !== independentVars.length) {
      setErrorMessage('Please fill in all fields.');
      return;
    }

    // Validate and convert inputs
    const features = independentVars.map(varName => {
      const value = predictionInputs[varName];
      if (!value && value !== 0) {
        throw new Error(`Value for ${varName} is missing.`);
      }
      // Attempt to convert numerical inputs to numbers
      const numValue = parseFloat(value);
      if (!isNaN(numValue) && !varName.toLowerCase().includes('outlook') && !varName.toLowerCase().includes('humidity') && !varName.toLowerCase().includes('wind')) {
        return numValue;
      }
      return value; // Keep as string for categorical variables
    });

    try {
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features })
      });
      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.error || 'Prediction failed');
      }
      const result = await res.json();
      setPredictionResult(result.prediction);
      setErrorMessage(null);
    } catch (err) {
      setErrorMessage('Error: ' + err.message);
    }
  };

  const theme = createTheme({
    palette: {
      primary: { main: '#1976d2' },
      secondary: { main: '#f50057' },
      success: { main: '#4caf50' }
    },
    typography: { fontFamily: 'Roboto, sans-serif' },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 'bold',
            borderRadius: '12px',
            transition: 'background-color 0.3s, color 0.3s, box-shadow 0.3s, transform 0.3s',
            '&:hover': {
              backgroundColor: 'white',
              color: 'var(--btn-bg)',
              transform: 'scale(1.05)',
              boxShadow: '0px 4px 12px rgba(0,0,0,0.2)'
            }
          }
        }
      }
    }
  });

  useEffect(() => {
    const overviewEl = document.getElementById('overview-header');
    if (overviewEl) {
      setTimeout(() => {
        overviewEl.style.opacity = '1';
        overviewEl.style.transform = 'translateZ(0) scale(1)';
      }, 200);
    }
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <GlobalStyles
        styles={{
          '@keyframes fadeIn': {
            '0%': { opacity: 0 },
            '100%': { opacity: 1 }
          },
          '@keyframes slideDown': {
            '0%': { transform: 'translateY(-20px)', opacity: 0 },
            '100%': { transform: 'translateY(0)', opacity: 1 }
          },
          '@keyframes scaleUp': {
            '0%': { transform: 'scale(0.8)' },
            '100%': { transform: 'scale(1)' }
          },
          '@keyframes pulse': {
            '0%': { transform: 'scale(1)' },
            '50%': { transform: 'scale(1.02)' },
            '100%': { transform: 'scale(1)' }
          }
        }}
      />

      <AppBar position="static" sx={{ background: '#393939', animation: 'slideDown 0.5s ease-out' }}>
        <Toolbar>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <img
              src="./div.jpg"
              alt="App Logo"
              style={{
                height: '45px',
                marginRight: '12px',
                borderRadius: '4px',
                transition: 'transform 0.5s'
              }}
              onMouseEnter={(e) => (e.currentTarget.style.transform = 'rotate(10deg) scale(1.05)')}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'rotate(0deg) scale(1)')}
            />
            <Typography variant="h6" sx={{ flexGrow: 1, fontFamily: 'Ubuntu', fontWeight: 'bold' }}>
              Machine Learning Model Trainer
            </Typography>
          </Box>
        </Toolbar>
      </AppBar>

      <Box
        id="overview-header"
        onClick={() => setShowOverviewPreview(true)}
        sx={{
          textAlign: 'center',
          my: 2,
          opacity: 0,
          transform: 'translateZ(0) scale(0.6)',
          transition: 'opacity 1s ease, transform 1s ease',
          cursor: 'pointer'
        }}
        data-aos="zoom-in"
        data-aos-duration="1000"
      >
        <Typography
          variant="h3"
          sx={{
            fontFamily: 'Ubuntu',
            fontWeight: 600,
            backdropFilter: 'blur(10px)',
            borderRadius: '2rem',
            boxShadow: '0px 4px 10px rgba(0,0,0,0.5)',
            width: 'fit-content',
            mx: 'auto',
            background: 'linear-gradient(0deg, #ea00f6 24%, #15f00e 35%, #ea00e1 45%)',
            WebkitTextFillColor: 'transparent',
            WebkitBackgroundClip: 'text',
            fontSize: { xs: '2rem', sm: '2.5rem' },
            p: 2,
            animation: 'fadeIn 1s ease-out'
          }}
        >
          Overview
        </Typography>
      </Box>

      <Dialog
        open={showOverviewPreview}
        onClose={() => setShowOverviewPreview(false)}
        maxWidth="md"
        fullWidth
        aria-labelledby="overview-dialog-title"
        PaperProps={{
          sx: {
            backgroundColor: '#FAF5E9',
            color: 'white',
            animation: 'fadeIn 0.5s ease-out'
          }
        }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', p: 1 }}>
          <IconButton onClick={() => setShowOverviewPreview(false)} aria-label="Close overview">
            <CustomCloseIcon />
          </IconButton>
        </Box>
        <Box
          sx={{
            textAlign: 'center',
            pb: 2,
            p: 2,
            m: 2,
            background: 'linear-gradient(135deg, #0A1828, #1a1a1a)',
            borderRadius: '16px',
            boxShadow: '0px 0px 15px rgba(0,0,0,0.5)',
            animation: 'pulse 2s infinite'
          }}
        >
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: '16px',
              padding: '16px'
            }}
          >
            <img
              src="./ml.png"
              alt="Machine Learning Overview"
              style={{
                maxWidth: '100%',
                width: '500px',
                borderRadius: 8,
                transition: 'transform 0.5s'
              }}
              onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.1)')}
              onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1)')}
            />
            <Box sx={{ textAlign: 'left', maxWidth: '80%', mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                What is Machine Learning?
              </Typography>
              <Typography variant="body1" paragraph>
                Machine Learning (ML) is a subfield of artificial intelligence (AI) that focuses on creating
                algorithms and statistical models enabling computers to learn from data. Rather than following
                explicitly programmed instructions, machine learning systems improve their performance on a task
                by identifying patterns and insights from training data.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Receiving the Data:
              </Typography>
              <Typography variant="body1" paragraph>
                The system starts by accepting a data file from the user, which could contain numbers, text, or even images.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Understanding the Data:
              </Typography>
              <Typography variant="body1" paragraph>
                It then looks at the file to determine what kind of information it contains. For example, if the file has image data, it knows to treat it differently than a typical table of numbers.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Identifying the Task:
              </Typography>
              <Typography variant="body1" paragraph>
                Based on the data, the system figures out whether it needs to learn to categorize items or predict a value.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Providing Suggestions and Preparing the Data:
              </Typography>
              <Typography variant="body1" paragraph>
                The system suggests which models might work best and then cleans/organizes the data for effective use.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Training the Model:
              </Typography>
              <Typography variant="body1" paragraph>
                With the data ready, the system trains a model using a portion of the data while reserving some for validation.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Evaluating Performance:
              </Typography>
              <Typography variant="body1" paragraph>
                The model’s performance is checked using various metrics, ensuring it can make accurate predictions.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Saving the Learned Model:
              </Typography>
              <Typography variant="body1" paragraph>
                Once evaluated, the trained model is saved for later use.
              </Typography>
              <Typography variant="subtitle1" gutterBottom sx={{ fontWeight: 'bold' }}>
                Making Predictions:
              </Typography>
              <Typography variant="body1" paragraph>
                The saved model processes new data and produces predictions or classifications based on its training.
              </Typography>
              <Typography variant="h6" gutterBottom>
                This overview outlines the journey from receiving raw data to providing useful insights.
              </Typography>
            </Box>
          </div>
        </Box>
      </Dialog>

      <Container maxWidth="lg" sx={{ py: 4, animation: 'fadeIn 1s ease-out' }}>
        <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
          <Box
            sx={{
              flex: 1,
              backgroundColor: '#fff',
              p: 4,
              borderRadius: 2,
              boxShadow: '0px 4px 10px rgba(0,0,0,0.1)',
              minHeight: '500px',
              transition: 'transform 0.5s',
              '&:hover': { transform: 'scale(1.02)' },
              animation: 'slideDown 0.6s ease-out'
            }}
          >
            <Typography
              sx={{
                textAlign: 'center',
                fontSize: '32px',
                fontFamily: 'Ubuntu',
                fontWeight: 600,
                mb: 3,
                color: '#7f39fb'
              }}
            >
              Upload & Train
            </Typography>
            <Box sx={{ textAlign: 'center', mb: 3 }}>
              <input
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                style={{ display: 'none' }}
                id="file-upload"
              />
              <label htmlFor="file-upload">
                <Button
                  variant="contained"
                  component="span"
                  aria-label="Upload CSV file"
                  sx={{
                    backgroundColor: '#1976d2',
                    color: 'white',
                    mr: 2,
                    px: 3,
                    py: 1,
                    borderRadius: '12px',
                    '--btn-bg': '#1976d2',
                    '&:hover': {
                      backgroundColor: 'white',
                      color: '#1976d2'
                    },
                    transition: 'all 0.3s'
                  }}
                  disabled={analyzing}
                >
                  Upload CSV
                </Button>
              </label>
              <Button
                variant="contained"
                onClick={handleAnalyze}
                sx={{
                  backgroundColor: 'orange',
                  color: 'white',
                  px: 3,
                  py: 1,
                  borderRadius: '12px',
                  '--btn-bg': 'orange',
                  '&:hover': {
                    backgroundColor: 'white',
                    color: 'orange'
                  },
                  transition: 'all 0.3s'
                }}
                disabled={!file || analyzing}
              >
                {analyzing ? <CircularProgress size={24} sx={{ color: 'orange' }} /> : 'Analyze'}
              </Button>
              {file && (
                <Typography sx={{ mt: 2, color: '#333', fontStyle: 'italic' }}>
                  Selected file: {file.name} (Rows: {rowCount})
                </Typography>
              )}
            </Box>
            {analyzed && problemType && (
              <FormControl fullWidth sx={{ mt: 3, animation: 'fadeIn 0.5s ease-out' }}>
                <InputLabel sx={{ color: '#333' }}>Algorithm</InputLabel>
                <Select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  label="Algorithm"
                  sx={{ color: '#333' }}
                >
                  {availableAlgorithms.map((alg) => (
                    <MenuItem key={alg} value={alg}>
                      {alg.replace(/_/g, ' ')}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            )}
            <Button
              variant="contained"
              fullWidth
              onClick={handleSubmit}
              sx={{
                mt: 4,
                backgroundColor: '#4caf50',
                color: 'white',
                borderRadius: '12px',
                '--btn-bg': '#4caf50',
                '&:hover': {
                  backgroundColor: 'white',
                  color: '#4caf50'
                },
                py: 1.5,
                fontSize: '1rem',
                transition: 'all 0.3s'
              }}
              disabled={loading || !analyzed || !algorithm}
            >
              {loading ? <CircularProgress size={24} sx={{ color: '#fff' }} /> : 'Train & Show'}
            </Button>
          </Box>

          <Box
            sx={{
              flex: 1,
              background: 'linear-gradient(135deg, #7f39fb, #9c27b0)',
              p: 4,
              borderRadius: 2,
              minHeight: '500px',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'flex-start',
              transition: 'transform 0.5s',
              '&:hover': { transform: 'scale(1.02)' },
              animation: 'slideDown 0.6s ease-out'
            }}
          >
            <Typography
              variant="h5"
              sx={{
                textAlign: 'center',
                fontWeight: 'bold',
                color: '#fff',
                mb: 3
              }}
            >
              Results
            </Typography>
            {metrics && (
              <Box
                sx={{
                  backgroundColor: 'rgba(255,255,255,0.85)',
                  borderRadius: 2,
                  p: 2,
                  boxShadow: '0 0 10px rgba(0,0,0,0.1)',
                  flex: '1 1 auto',
                  animation: 'fadeIn 0.8s ease-out'
                }}
              >
                <RenderMetrics
                  metrics={(() => {
                    const m = { ...metrics };
                    delete m.plot;
                    return m;
                  })()}
                />
              </Box>
            )}
          </Box>
        </Box>
      </Container>

      {metrics && (
        <Container maxWidth="lg" sx={{ py: 2 }}>
          <Box
            sx={{
              backgroundColor: 'rgba(255,255,255,0.9)',
              borderRadius: 2,
              p: 2,
              boxShadow: '0 0 10px rgba(0,0,0,0.1)',
              textAlign: 'center',
              animation: 'fadeIn 0.8s ease-out'
            }}
          >
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold', color: '#1976d2' }}>
              Plot
            </Typography>
            {(algorithm === 'linear_regression' || algorithm === 'svm') ? (
              metrics.plot ? (
                <img
                  src={`data:image/png;base64,${metrics.plot}`}
                  alt="Scatter Plot"
                  style={{ maxWidth: '100%', borderRadius: '4px' }}
                />
              ) : null
            ) : (
              metrics.plot && (
                <img
                  src={`data:image/png;base64,${metrics.plot}`}
                  alt="Plot"
                  style={{ maxWidth: '100%', borderRadius: '4px' }}
                />
              )
            )}
          </Box>
        </Container>
      )}

      {metrics && datasetType === 'tabular' && (
        <Container maxWidth="lg" sx={{ py: 2 }}>
          <Box
            sx={{
              backgroundColor: 'rgba(255,255,255,0.9)',
              borderRadius: 2,
              p: 2,
              boxShadow: '0 0 10px rgba(0,0,0,0.1)',
              animation: 'fadeIn 0.8s ease-out'
            }}
          >
            <Typography variant="h6" sx={{ mb: 2, fontWeight: 'bold', color: '#800080', textAlign: 'center' }}>
              Make Predictions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: { xs: 'column', md: 'row' }, gap: 4 }}>
              <Box sx={{ flex: 1 }}>
                {independentVars.map((varName) => (
                  <TextField
                    key={varName}
                    label={varName}
                    variant="outlined"
                    fullWidth
                    sx={{ mt: 2 }}
                    onChange={(e) => handleInputChange(varName, e.target.value)}
                  />
                ))}
                <Button
                  variant="contained"
                  onClick={handlePredict}
                  sx={{
                    mt: 2,
                    backgroundColor: '#1976d2',
                    color: 'white',
                    '--btn-bg': '#1976d2',
                    '&:hover': {
                      backgroundColor: 'white',
                      color: '#1976d2'
                    }
                  }}
                >
                  Predict
                </Button>
              </Box>
              <Box
                sx={{
                  flex: 1,
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  alignItems: 'center',
                  p: 2,
                  backgroundColor: '#E6E6FA',
                  borderRadius: 2,
                  minHeight: '200px'
                }}
              >
                {predictionResult && (
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h6" sx={{ color: '#0000CD', fontWeight: 'bold' }}>
                      Prediction:{' '}
                      <span style={{ color: '#000000' }}>{predictionResult}</span>
                    </Typography>
                  </Box>
                )}
                {errorMessage && (
                  <Box sx={{ mt: 2, textAlign: 'center' }}>
                    <Typography variant="body1" sx={{ color: 'red' }}>
                      {errorMessage}
                    </Typography>
                  </Box>
                )}
                {!predictionResult && !errorMessage && (
                  <Typography variant="body1" sx={{ color: '#666', fontStyle: 'italic', textAlign: 'center' }}>
                    Enter values and click Predict to see the result.
                  </Typography>
                )}
              </Box>
            </Box>
          </Box>
        </Container>
      )}
    </ThemeProvider>
  );
}

export default App;
