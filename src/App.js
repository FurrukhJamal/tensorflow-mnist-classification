import React, { useState, useEffect } from "react";
import App1 from "./App1";
import App2 from "./App2";
import "./App.css";
import * as tf from "@tensorflow/tfjs";

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

import { AppContext } from "./context";

function App() {
  const [showBasicVersion, setShowBasicVersion] = useState(true);
  const [model, setModel] = useState(null);
  const [inputs, setInputs] = useState([]);
  const [outputs, setOutputs] = useState([]);
  const [inputsTensor, setInputsTensor] = useState(null);
  const [outputsTensor, setOutputsTensor] = useState(null);
  const [isTrainingDone, setIsTrainingDone] = useState(false);

  useEffect(() => {
    document.title = "TensorFlow MNIST Dataset Classification";
    // console.log("TRAINING_DATA : ", TRAINING_DATA);

    const inputs = TRAINING_DATA?.inputs;
    const outputs = TRAINING_DATA?.outputs;

    tf.util.shuffleCombo(inputs, outputs);

    const inputsTensor = tf.tensor2d(inputs);
    const outputsTensor = tf.oneHot(tf.tensor1d(outputs, "int32"), 10);

    setInputs(TRAINING_DATA.inputs);
    setOutputs(TRAINING_DATA.outputs);
    setInputsTensor(tf.tensor2d(inputs));
    setOutputsTensor(tf.oneHot(tf.tensor1d(outputs, "int32"), 10));

    let m = tf.sequential();
    setModel(m);
    console.log("one input.shape", inputsTensor.shape);
  }, []);

  useEffect(() => {
    if (model) {
      (async () => {
        model.add(
          tf.layers.conv2d({
            inputShape: [28, 28, 1],
            filters: 16,
            kernelSize: [3, 3],
            strides: 1,
            padding: "same",
            activation: "relu",
          })
        );

        model.add(tf.layers.maxPool2d({ poolSize: [2, 2], strides: 2 }));

        model.add(
          tf.layers.conv2d({
            filters: 32,
            kernelSize: [3, 3],
            strides: 1,
            padding: "same",
            activation: "relu",
          })
        );

        model.add(tf.layers.maxPool2d({ poolSize: [2, 2], strides: 2 }));

        model.add(tf.layers.dropout(0.25));

        model.add(tf.layers.flatten());
        model.add(tf.layers.dense({ units: 128, activation: "relu" }));
        model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

        model.compile({
          optimizer: "adam",
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        });

        let reshapedInputs = inputsTensor.reshape([inputs.length, 28, 28, 1]);

        let result = await model.fit(reshapedInputs, outputsTensor, {
          epochs: 30,
          validationSplit: 0.2,
          shuffle: true,
          batchSize: 256,
          callbacks: {
            onEpochEnd: (epoch, logs) =>
              console.log("Data on epoch " + epoch, logs),
          },
        });

        inputsTensor.dispose();
        outputsTensor.dispose();

        setIsTrainingDone(true);
      })();
    }
  }, [model]);

  function handleAppVersionChange() {
    setShowBasicVersion((prev) => !prev);
  }

  return (
    <AppContext.Provider
      value={{
        isTrainingDone,
        model,
        inputs,
        setInputs,
        outputs,
        setOutputs,
        inputsTensor,
        setInputsTensor,
        outputsTensor,
        setOutputsTensor,
      }}
    >
      <div>
        {showBasicVersion ? <App1 /> : <App2 />}

        <div
          style={{ display: "flex", justifyContent: "center", width: "100%" }}
        >
          <button onClick={handleAppVersionChange} style={{ marginTop: 10 }}>
            {showBasicVersion
              ? "Draw Numbers Manually"
              : "Select Numbers Automatically"}
          </button>
        </div>
      </div>
    </AppContext.Provider>
  );
}

export default App;
