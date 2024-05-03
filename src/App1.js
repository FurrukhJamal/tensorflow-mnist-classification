import logo from "./logo.svg";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useState, useRef, useContext } from "react";
import "./App.css";

// import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";
import { AppContext } from "./context";

function App1() {
  const [isPrediction, setIsPrediction] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const canvasRef1 = useRef(null);

  const {
    isTrainingDone,
    model,
    inputs,
    outputs,
    inputsTensor,
    outputsTensor,
  } = useContext(AppContext);

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

        // let output = model.predict(newInput.expandDims());
        let output = model.predict(newInput.reshape([1, 28, 28, 1]));
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
