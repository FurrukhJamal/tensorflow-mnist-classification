import logo from "./logo.svg";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useState, useRef } from "react";
import "./App.css";

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

function App1() {
  const [model, setModel] = useState(null);
  const [isTrainingDone, setIsTrainingDone] = useState(false);
  const [inputs, setInputs] = useState([]);
  const [outputs, setOutputs] = useState([]);
  const [isPrediction, setIsPrediction] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [inputsTensor, setInputsTensor] = useState(null);
  const [outputsTensor, setOutputsTensor] = useState(null);

  const canvasRef1 = useRef(null);

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
  }, []);

  useEffect(() => {
    if (model) {
      (async () => {
        model.add(
          tf.layers.dense({ inputShape: [784], units: 32, activation: "relu" })
        );
        model.add(tf.layers.dense({ units: 16, activation: "relu" }));
        model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

        model.compile({
          optimizer: "adam",
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        });

        let result = await model.fit(inputsTensor, outputsTensor, {
          epochs: 50,
          validationSplit: 0.2,
          shuffle: true,
          batchSize: 512,
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

  useEffect(() => {
    const intervalId = setInterval(evaluate, 4000);
    return () => {
      clearInterval(intervalId);
    };
  }, [isTrainingDone]);

  async function evaluate() {
    const OFFSET = Math.floor(Math.random() * inputs.length);
    if (inputs.length > 0) {
      let answer = tf.tidy(function () {
        let newInput = tf.tensor1d(inputs[OFFSET]);

        let output = model.predict(newInput.expandDims());
        output.print();
        return output.squeeze().argMax();
      });

      let index = await answer.array();
      if (outputs[OFFSET] === index) {
        setIsPrediction(true);
      } else {
        setIsPrediction(false);
      }

      setPrediction(index);
      // Bug fix to display zero as a string so that Iit doesnt get stored as false
      if (index === 0) {
        setPrediction("0");
      }

      answer.dispose();
      drawImage(inputs[OFFSET]);
    }
  }

  function drawImage(digit) {
    let CTX = canvasRef1.current.getContext("2d");
    let imageData = CTX.getImageData(0, 0, 28, 28);
    for (let i = 0; i < digit.length; i++) {
      imageData.data[i * 4] = digit[i] * 255;
      imageData.data[i * 4 + 1] = digit[i] * 255;
      imageData.data[i * 4 + 2] = digit[i] * 255;
      imageData.data[i * 4 + 3] = 255;
    }

    CTX.putImageData(imageData, 0, 0);
  }

  return (
    <div>
      <h1>TensorFlow.js MNIST Classifier</h1>

      <p>See console for even more outputs</p>
      {model ? (
        <>
          <section className="box">
            <h2>Input Image</h2>
            <p>
              Input Image is a 28x28 pixel greyscale image for MNIST dataset- a
              real hand drawn digit
            </p>
            <canvas
              ref={canvasRef1}
              width="28"
              height="28"
              style={{ zoom: 6 }}
            ></canvas>
          </section>

          <section className="box">
            <h2>Prediction</h2>
            <p>
              Below you see what number the trained Tensorflow.js model has
              predicted from the input image
            </p>
            <p>Red is wrong Prediction. Green is a correct one</p>
            {prediction ? (
              <>
                <p
                  id="prediction"
                  className={isPrediction ? "correct" : "wrong"}
                >
                  {prediction}
                </p>
              </>
            ) : (
              <>
                <p id="prediction">Training model . Please wait ...</p>
              </>
            )}
          </section>
        </>
      ) : (
        <div
          style={{
            marginTop: 10,
            width: "45%",
            height: "45%",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          <p>Model is been loaded ...</p>
        </div>
      )}
    </div>
  );
}

export default App1;
