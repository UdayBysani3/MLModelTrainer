import React, { useState } from 'react';
import { BentoGrid } from './components/ui/bento-grid';
import { Button as MovingButton } from './components/ui/moving-border';
import { Input } from './components/ui/input';
import { Modal } from './components/ui/modal';
import { BackgroundBeams } from './components/ui/background-beams';
import { TypewriterEffect } from './components/ui/typewriter-effect';
import { motion } from 'framer-motion';
import { Upload, Brain, Activity, Zap, CheckCircle, BarChart3, List, Table2 } from 'lucide-react';
import { cn } from './lib/utils';

import { CardContainer, CardBody, CardItem } from './components/ui/3d-card';

// Metrics Renderer compatible with BentoGrid
function RenderMetricsBento({ metrics }) {
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
    orderedKeys = [...classificationOrder.filter(k => metrics.hasOwnProperty(k)), ...Object.keys(metrics).filter(k => !classificationOrder.includes(k))];
  } else {
    orderedKeys = [...regressionOrder.filter(k => metrics.hasOwnProperty(k)), ...Object.keys(metrics).filter(k => !regressionOrder.includes(k))];
  }

  orderedKeys = [...new Set(orderedKeys)].filter(key => key !== 'plot');

  return (
    <BentoGrid className="max-w-6xl mx-auto md:auto-rows-auto grid-cols-1 md:grid-cols-2 gap-6">
      {orderedKeys.map((key, i) => {
        const value = metrics[key];

        // Custom formatting for complex metrics
        if (key === 'confusion_matrix' && Array.isArray(value)) {
          return (
            <CardContainer key={key} containerClassName="md:col-span-2 py-0 h-full w-full" className="h-full w-full">
              <CardBody className="bg-gradient-to-br from-neutral-900 via-neutral-900 to-black relative group/card dark:hover:shadow-2xl dark:hover:shadow-violet-500/[0.1] dark:bg-black border-white/[0.2] w-full h-auto rounded-xl p-6 border h-full flex flex-col">
                <CardItem translateZ="50" className="flex items-center gap-2 mb-4">
                  <div className="p-2 bg-violet-500/10 rounded-lg">
                    <Table2 className="h-5 w-5 text-violet-400" />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-neutral-200">Confusion Matrix</h3>
                    <p className="text-xs text-neutral-500">Predicted vs Actual Labels</p>
                  </div>
                </CardItem>
                <CardItem translateZ="100" className="w-full flex-1 flex items-center justify-center">
                  <div className="p-4 bg-black/40 rounded-xl border border-white/5 backdrop-blur-sm shadow-inner shadow-black/50 overflow-x-auto w-full flex justify-center">
                    <table className="border-collapse text-sm md:text-base">
                      <tbody>
                        {value.map((row, rowIndex) => (
                          <tr key={rowIndex}>
                            {row.map((cell, cellIndex) => (
                              <td key={cellIndex} className="border border-violet-500/20 p-3 md:p-4 text-center min-w-[50px] text-neutral-200 font-mono bg-neutral-900/50 group-hover/card:bg-violet-500/10 transition-colors">
                                {cell}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </CardItem>
              </CardBody>
            </CardContainer>
          );
        } else if (key === 'classification_report') {
          return (
            <CardContainer key={key} containerClassName="md:col-span-2 py-0 h-full w-full" className="h-full w-full">
              <CardBody className="bg-gradient-to-br from-neutral-900 via-neutral-900 to-black relative group/card dark:hover:shadow-2xl dark:hover:shadow-cyan-500/[0.1] dark:bg-black border-white/[0.2] w-full h-auto rounded-xl p-6 border h-full flex flex-col">
                <CardItem translateZ="50" className="flex items-center gap-2 mb-4">
                  <div className="p-2 bg-cyan-500/10 rounded-lg">
                    <List className="h-5 w-5 text-cyan-400" />
                  </div>
                  <div>
                    <h3 className="font-bold text-lg text-neutral-200">Classification Report</h3>
                    <p className="text-xs text-neutral-500">Detailed precision, recall, f1-score</p>
                  </div>
                </CardItem>
                <CardItem translateZ="80" className="w-full h-full">
                  <div className="w-full bg-black/40 rounded-xl p-6 border border-white/5 backdrop-blur-sm shadow-inner shadow-black/50 max-h-[400px] overflow-y-auto custom-scrollbar">
                    <pre className="text-xs md:text-sm font-mono text-neutral-300 whitespace-pre-wrap font-medium leading-relaxed">
                      {typeof value === 'object' ? JSON.stringify(value, null, 2) : value}
                    </pre>
                  </div>
                </CardItem>
              </CardBody>
            </CardContainer>
          );
        } else {
          // Standard Metric with 3D Card Effect

          const title = key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
          let displayValue = String(value);
          if (typeof value === 'number') {
            displayValue = value.toFixed(3);
          } else if (Array.isArray(value) && value.every(v => typeof v === 'number')) {
            displayValue = value.map(v => v.toFixed(3)).join(', ');
          }

          return (
            <CardContainer key={key} containerClassName="py-0 h-full w-full" className="h-full w-full">
              <CardBody className="bg-gradient-to-br from-neutral-900 to-black relative group/card dark:hover:shadow-2xl dark:hover:shadow-emerald-500/[0.1] dark:bg-black border-white/[0.2] w-full h-auto rounded-xl p-6 border h-full flex flex-col justify-between">
                <CardItem translateZ="50" className="text-xl font-bold text-neutral-300">
                  {title}
                </CardItem>
                <CardItem
                  as="p"
                  translateZ="60"
                  className="text-neutral-500 text-sm max-w-sm mt-2 dark:text-neutral-300"
                >
                  Performance Metric
                </CardItem>
                <CardItem translateZ="100" className="w-full mt-4">
                  <div className="flex items-center justify-center p-4 bg-white/5 rounded-xl border border-white/10 group-hover/card:bg-violet-500/10 group-hover/card:border-violet-500/50 transition-colors">
                    <span className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-violet-400 via-purple-400 to-cyan-400">
                      {displayValue}
                    </span>
                  </div>
                </CardItem>
              </CardBody>
            </CardContainer>
          );
        }
      })}
    </BentoGrid>
  );
}

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
  const [showPlotModal, setShowPlotModal] = useState(false);
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
      const analyzeRes = await fetch('https://mlmodeltrainer-backend.onrender.com/analyze', {
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

      // All datasets treated uniformly - last column is target
      setIndependentVars(analyzeResult.columns.slice(0, -1));

      if (analyzeResult.problem_type === 'classification') {
        setAvailableAlgorithms(['logistic_regression', 'knn', 'naive_bayes', 'decision_tree', 'svm']);
      } else {
        setAvailableAlgorithms(['linear_regression', 'ridge', 'lasso', 'elasticnet', 'decision_tree', 'svm']);
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
      const trainRes = await fetch('https://mlmodeltrainer-backend.onrender.com/train', {
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

    const features = independentVars.map(varName => {
      const value = predictionInputs[varName];
      if (!value && value !== 0) throw new Error(`Value for ${varName} is missing.`);
      const numValue = parseFloat(value);
      if (!isNaN(numValue) && !varName.toLowerCase().includes('outlook') && !varName.toLowerCase().includes('humidity') && !varName.toLowerCase().includes('wind')) {
        return numValue;
      }
      return value;
    });

    try {
      const res = await fetch('https://mlmodeltrainer-backend.onrender.com/predict', {
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

  return (
    <div className="w-full bg-neutral-950 relative flex flex-col antialiased min-h-screen">
      <BackgroundBeams />

      {/* Header */}
      <div className="fixed top-0 left-0 right-0 z-50 w-full p-4 md:p-6 flex justify-between items-center border-b border-white/10 bg-black/50 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <img src="/MainLogo.png" alt="ML Trainer Pro" className="h-10 w-auto object-contain" />
          <div>
            <TypewriterEffect
              words={[
                // { text: "ML", className: "text-white" },
                { text: "Model", className: "text-white" },
                { text: "Trainer", className: "text-violet-500" }
              ]}
              className="text-xl md:text-2xl"
              cursorClassName="bg-violet-500 h-6 md:h-8"
            />
          </div>
        </div>
        <button onClick={() => setShowOverviewPreview(true)} className="px-4 py-2 rounded-lg bg-neutral-800 hover:bg-neutral-700 text-neutral-300 text-sm transition font-medium border border-neutral-700">
          Overview
        </button>
      </div>

      <motion.div className={cn(
        "relative z-10 flex-1 w-full mx-auto p-4 md:p-8 pt-28 md:pt-32 transition-all duration-700",
        metrics ? "max-w-7xl grid grid-cols-1 lg:grid-cols-12 gap-8" : "max-w-2xl flex items-center justify-center"
      )}>

        {/* Left Sidebar / Config Panel */}
        <div className={cn(metrics ? "lg:col-span-4" : "w-full", "space-y-6")}>

          {/* Initial Layout: Quick Guide + Upload in Grid */}
          {/* Quick Start Guide */}
          {!analyzed && (
            <div className="bg-gradient-to-br from-violet-600/15 via-purple-600/10 to-violet-600/15 border-2 border-violet-400/30 rounded-2xl p-5 backdrop-blur-sm shadow-lg shadow-violet-500/10">
              <div className="flex items-start gap-4">
                <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center shadow-lg">
                  <span className="text-white text-xl">💡</span>
                </div>
                <div className="flex-1 space-y-3">
                  <h3 className="text-base font-bold text-white">Quick Start</h3>

                  <div className="space-y-2">
                    <p className="text-sm text-neutral-300 leading-relaxed flex items-start gap-2">
                      <span className="text-violet-400 mt-0.5">1.</span>
                      <span>Upload CSV</span>
                    </p>
                    <p className="text-sm text-neutral-300 leading-relaxed flex items-start gap-2">
                      <span className="text-violet-400 mt-0.5">2.</span>
                      <span>Analyze data</span>
                    </p>
                    <p className="text-sm text-neutral-300 leading-relaxed flex items-start gap-2">
                      <span className="text-violet-400 mt-0.5">3.</span>
                      <span>Train model</span>
                    </p>
                  </div>

                  <div className="bg-amber-500/10 border-2 border-amber-400/40 rounded-xl p-3 mt-3">
                    <div className="flex items-start gap-2">
                      <span className="text-amber-400 text-base mt-0.5">⚠️</span>
                      <div>
                        <p className="text-xs font-bold text-amber-300 mb-1">Important:</p>
                        <p className="text-xs text-neutral-200 leading-relaxed">
                          Target (Output) variable must be the <span className="text-amber-400 font-bold">last column</span>.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Dataset Upload Card */}
          <div className="bg-black/40 border border-white/10 rounded-2xl p-6 backdrop-blur-sm">
            <h2 className="text-lg font-semibold text-neutral-200 mb-4 flex items-center gap-2"><Upload className="w-4 h-4 text-cyan-400" /> Dataset</h2>

            <label className="flex flex-col items-center justify-center w-full h-24 sm:h-32 border-2 border-dashed border-neutral-700 rounded-xl cursor-pointer hover:border-violet-500 hover:bg-white/5 transition-all group">
              <input type="file" className="hidden" accept=".csv" onChange={handleFileChange} />
              <Upload className="w-8 h-8 mb-2 text-neutral-500 group-hover:text-violet-400 transition-colors" />
              <p className="text-xs text-neutral-400 text-center px-2">
                {file ? <span className="text-violet-400 font-medium">{file.name}</span> : "Drop CSV or Click to Upload"}
              </p>
            </label>

            <div className="mt-4">
              <MovingButton
                borderRadius="12px"
                className={cn("bg-neutral-900 text-white font-semibold transition-colors border-neutral-800", analyzing ? "opacity-50" : "")}
                containerClassName="w-full h-12"
                onClick={handleAnalyze}
                disabled={!file || analyzing}
              >
                {analyzing ? "Scanning Data..." : "Analyze Dataset"}
              </MovingButton>
            </div>

            {/* Integrated Algorithm Selection */}
            {analyzed && problemType && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                className="mt-6 pt-6 border-t border-white/10"
              >
                <div className="flex items-center gap-2 mb-4 text-emerald-400 bg-emerald-950/30 p-2 rounded-lg border border-emerald-500/20">
                  <CheckCircle className="w-4 h-4" />
                  <span className="text-xs font-bold uppercase tracking-wider">{problemType} Detected</span>
                </div>

                <div className="space-y-3">
                  <label className="text-xs font-semibold text-neutral-400 uppercase">Select Algorithm</label>
                  <div className="relative">
                    <select
                      value={algorithm}
                      onChange={(e) => setAlgorithm(e.target.value)}
                      className="w-full bg-neutral-900 text-white border border-neutral-700 rounded-lg p-3 text-sm focus:ring-2 focus:ring-violet-500 focus:outline-none appearance-none"
                    >
                      <option value="">-- Select --</option>
                      {availableAlgorithms.map(alg => (
                        <option key={alg} value={alg}>{alg.replace(/_/g, ' ')}</option>
                      ))}
                    </select>
                    <div className="absolute right-3 top-3.5 pointer-events-none text-neutral-500">▼</div>
                  </div>
                </div>

                <div className="mt-6">
                  <MovingButton
                    borderRadius="12px"
                    className={cn("bg-green-600 text-white font-bold tracking-wide", loading ? "opacity-70" : "")}
                    containerClassName="w-full h-14"
                    onClick={handleSubmit}
                    disabled={loading || !algorithm}
                  >
                    {loading ? "Training..." : "Start Training"}
                  </MovingButton>
                </div>
              </motion.div>
            )}
          </div>

          {/* Prediction Card */}
          {metrics && datasetType === 'tabular' && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gradient-to-b from-neutral-900 to-black border border-white/10 rounded-2xl p-6 relative overflow-hidden"
            >
              <div className="absolute inset-0 bg-violet-500/5 z-0" />
              <div className="relative z-10">
                <h3 className="text-lg font-bold text-white mb-2 flex items-center gap-2"><Zap className="w-4 h-4 text-yellow-400" /> Live Prediction</h3>
                <p className="text-xs text-neutral-400 mb-4">
                  These are the columns in the dataset you provided. Before filling the boxes, please review the dataset and provide the data for live predictions.
                </p>
                <div className="space-y-3 max-h-[300px] overflow-y-auto custom-scrollbar pr-2">
                  {independentVars.map((varName) => (
                    <div key={varName}>
                      <label className="text-[10px] uppercase font-bold text-neutral-500 mb-1 block">{varName}</label>
                      <Input
                        className="h-9 bg-neutral-950 text-white border-neutral-800 focus:border-violet-500"
                        placeholder="..."
                        onChange={(e) => handleInputChange(varName, e.target.value)}
                      />
                    </div>
                  ))}
                </div>
                <button
                  onClick={handlePredict}
                  className="w-full mt-4 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-lg text-white font-semibold text-sm hover:brightness-110 transition shadow-lg shadow-cyan-500/20"
                >
                  Predict
                </button>

                {predictionResult !== null && (
                  <div className="mt-4 p-4 bg-white/5 rounded-xl border border-white/10 text-center">
                    <span className="text-xs text-neutral-400">Prediction</span>
                    <div className="text-3xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-emerald-600">
                      {typeof predictionResult === 'number' ? (problemType === 'regression' ? predictionResult.toFixed(5) : predictionResult.toFixed(3)) : predictionResult}
                    </div>
                  </div>
                )}
                {errorMessage && <div className="mt-2 text-xs text-red-400 text-center">{errorMessage}</div>}
              </div>
            </motion.div>
          )}
        </div>

        {metrics && (
          <div className="lg:col-span-8 space-y-6">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
            >
              <h2 className="text-2xl font-bold text-white mb-6 pl-2 border-l-4 border-violet-500">Training Results</h2>

              {/* Metrics Grid */}
              <RenderMetricsBento metrics={{ ...metrics, plot: undefined }} />

              {/* Visualization */}
              {metrics.plot && (
                <div
                  className="mt-8 rounded-xl bg-gradient-to-br from-neutral-900 via-neutral-900 to-black border border-white/10 shadow-2xl hover:shadow-pink-500/20 transition-all duration-300 overflow-hidden cursor-pointer group"
                  onClick={() => setShowPlotModal(true)}
                >
                  <div className="bg-black/40 p-4 border-b border-white/5 flex items-center gap-3 backdrop-blur-sm">
                    <div className="p-2 bg-pink-500/10 rounded-lg group-hover:scale-110 transition-transform">
                      <BarChart3 className="w-5 h-5 text-pink-400" />
                    </div>
                    <div>
                      <h3 className="font-bold text-lg text-neutral-200">Plot</h3>
                      <p className="text-xs text-neutral-500 group-hover:text-pink-400 transition-colors">Click to expand</p>
                    </div>
                  </div>
                  <div className="p-6 bg-neutral-950 flex justify-center">
                    <div className="rounded-lg overflow-hidden border border-white/5 shadow-lg max-h-[400px]">
                      <img src={`data:image/png;base64,${metrics.plot}`} alt="Model Visualization" className="w-full h-full object-contain" />
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          </div>
        )}

      </motion.div>

      <Modal isOpen={showOverviewPreview} onClose={() => setShowOverviewPreview(false)} className="dark">
        {/* Same Modal Content as before but styled darker if needed, currently reusing logic */}
        <div className="space-y-6 text-neutral-300 max-h-[70vh] overflow-y-auto custom-scrollbar">
          <div>
            <h2 className="text-3xl font-bold text-white mb-2 bg-clip-text text-transparent bg-gradient-to-r from-violet-400 to-cyan-400">ML Trainer Pro Guide</h2>
            <p className="text-sm text-neutral-400">Learn how to train machine learning models in minutes</p>
          </div>

          {/* Step 1 */}
          <div className="p-5 bg-gradient-to-br from-violet-600/20 via-purple-600/15 to-violet-600/20 rounded-2xl border border-violet-400/30 shadow-lg shadow-violet-500/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-white font-bold shadow-lg">1</div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Upload Your Dataset</h3>
                <p className="text-sm text-neutral-300 mb-2">Click or drag-and-drop a <span className="text-violet-400 font-semibold">CSV file</span> containing your dataset.</p>
                <ul className="text-xs text-neutral-400 space-y-1 ml-4 list-disc">
                  <li>Last column of your CSV should be the target variable (what you want to predict)</li>
                  <li>Works with any dataset: MNIST, Iris, Titanic, Housing, etc.</li>
                  <li>Ensure data is clean (no excessive missing values)</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Step 2 */}
          <div className="p-5 bg-gradient-to-br from-cyan-600/20 via-blue-600/15 to-cyan-600/20 rounded-2xl border border-cyan-400/30 shadow-lg shadow-cyan-500/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white font-bold shadow-lg">2</div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Analyze Dataset</h3>
                <p className="text-sm text-neutral-300 mb-2">Click <span className="text-cyan-400 font-semibold">"Analyze Dataset"</span> to automatically detect:</p>
                <ul className="text-xs text-neutral-400 space-y-1 ml-4 list-disc">
                  <li><strong>Problem Type:</strong> Classification or Regression</li>
                  <li><strong>Features:</strong> Input variables in your dataset</li>
                  <li><strong>Suitable Algorithms:</strong> Best models for your data</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Step 3 */}
          <div className="p-5 bg-gradient-to-br from-emerald-600/20 via-green-600/15 to-emerald-600/20 rounded-2xl border border-emerald-400/30 shadow-lg shadow-emerald-500/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-emerald-500 to-green-600 flex items-center justify-center text-white font-bold shadow-lg">3</div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Select Algorithm</h3>
                <p className="text-sm text-neutral-300 mb-2">Choose from recommended algorithms:</p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="bg-black/30 p-2 rounded border border-emerald-500/20">
                    <strong className="text-emerald-400">Classification:</strong>
                    <ul className="text-neutral-400 mt-1 space-y-0.5 ml-2">
                      <li>• Logistic Regression</li>
                      <li>• KNN</li>
                      <li>• Naive Bayes</li>
                      <li>• Decision Tree</li>
                      <li>• SVM</li>
                    </ul>
                  </div>
                  <div className="bg-black/30 p-2 rounded border border-emerald-500/20">
                    <strong className="text-emerald-400">Regression:</strong>
                    <ul className="text-neutral-400 mt-1 space-y-0.5 ml-2">
                      <li>• Linear Regression</li>
                      <li>• Ridge Regression</li>
                      <li>• Lasso Regression</li>
                      <li>• ElasticNet</li>
                      <li>• Decision Tree</li>
                      <li>• SVM</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Step 4 */}
          <div className="p-5 bg-gradient-to-br from-pink-600/20 via-rose-600/15 to-pink-600/20 rounded-2xl border border-pink-400/30 shadow-lg shadow-pink-500/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-pink-500 to-rose-600 flex items-center justify-center text-white font-bold shadow-lg">4</div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Train Your Model</h3>
                <p className="text-sm text-neutral-300 mb-2">Click <span className="text-pink-400 font-semibold">"Start Training"</span> to train the model. You'll see:</p>
                <ul className="text-xs text-neutral-400 space-y-1 ml-4 list-disc">
                  <li><strong>Performance Metrics:</strong> Accuracy, Precision, F1-Score, MSE, R²</li>
                  <li><strong>Confusion Matrix:</strong> For classification problems</li>
                  <li><strong>Visualizations:</strong> ROC curves, scatter plots, decision trees</li>
                  <li><strong>Classification Report:</strong> Detailed per-class performance</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Step 5 */}
          <div className="p-5 bg-gradient-to-br from-amber-600/20 via-orange-600/15 to-amber-600/20 rounded-2xl border border-amber-400/30 shadow-lg shadow-amber-500/10">
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-white font-bold shadow-lg">5</div>
              <div className="flex-1">
                <h3 className="text-lg font-bold text-white mb-2">Make Predictions (Tabular Data)</h3>
                <p className="text-sm text-neutral-300 mb-2">For tabular datasets, use the <span className="text-yellow-400 font-semibold">"Live Prediction"</span> panel:</p>
                <ul className="text-xs text-neutral-400 space-y-1 ml-4 list-disc">
                  <li>Enter values for all input features</li>
                  <li>Click "Predict" to get instant predictions</li>
                  <li>The model uses your trained algorithm</li>
                </ul>
              </div>
            </div>
          </div>

          {/* Tips Section */}
          <div className="p-4 bg-gradient-to-r from-neutral-800 to-neutral-900 rounded-xl border border-neutral-700">
            <h3 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
              <Zap className="w-5 h-5 text-yellow-400" />
              Pro Tips
            </h3>
            <ul className="text-xs text-neutral-300 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-violet-400 font-bold">•</span>
                <span><strong>Data Quality:</strong> Clean data leads to better models. Remove or handle missing values before uploading.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-cyan-400 font-bold">•</span>
                <span><strong>Dataset Size:</strong> Large datasets ({'>'}15,000 rows) are automatically sampled for faster training.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 font-bold">•</span>
                <span><strong>Try Multiple Algorithms:</strong> Different algorithms work better for different datasets. Experiment!</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-pink-400 font-bold">•</span>
                <span><strong>Visualizations:</strong> Use plots to understand model behavior and performance.</span>
              </li>
            </ul>
          </div>

          {/* Footer */}
          <div className="text-center pt-4 border-t border-neutral-800">
            <p className="text-xs text-neutral-500">Ready to start? Close this guide and upload your dataset!</p>
          </div>
        </div>
      </Modal>

      <Modal isOpen={showPlotModal} onClose={() => setShowPlotModal(false)} className="dark">
        <div className="flex flex-col items-center justify-center p-2">
          <h2 className="text-xl font-bold text-white mb-4">Visualization Result</h2>
          <div className="bg-white/5 rounded-xl border border-white/10 p-2 shadow-2xl">
            <img src={`data:image/png;base64,${metrics?.plot}`} alt="Full Plot" className="max-h-[80vh] w-auto max-w-full rounded-lg" />
          </div>
          <button onClick={() => setShowPlotModal(false)} className="mt-4 px-4 py-2 bg-neutral-800 rounded-lg text-white text-sm hover:bg-neutral-700">Close</button>
        </div>
      </Modal>

    </div>
  );
}

export default App;
