import logo from "./logo.svg";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useState, useRef } from "react";
import "./App.css";

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

function App2() {
  const [model, setModel] = useState(null);
  const [isTrainingDone, setIsTrainingDone] = useState(false);
  const [inputs, setInputs] = useState([]);
  const [outputs, setOutputs] = useState([]);
  const [isPrediction, setIsPrediction] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [inputsTensor, setInputsTensor] = useState(null);
  const [outputsTensor, setOutputsTensor] = useState(null);

  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef(null);
  const [forDrawingPrevX, setForDrawingPrevX] = useState(null);
  const [forDrawingPrevY, setForDrawingPrevY] = useState(null);

  const [showBasicVersionApp, setShowBasicVersionApp] = useState(true);

  const canvasDebug = useRef(null);

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

        // inputsTensor.dispose();
        outputsTensor.dispose();

        setIsTrainingDone(true);
      })();
    }
  }, [model]);

  useEffect(() => {
    // setInterval(evaluate, 4000);
  }, [isTrainingDone]);

  useEffect(() => {
    if (model && isTrainingDone) {
      //initializing the canvas background color to be black
      let ctx = canvasRef.current.getContext("2d");
      ctx.fillstyle = "black";
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [isTrainingDone, model]);

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
    let CTX = canvasRef.current.getContext("2d");
    let imageData = CTX.getImageData(0, 0, 28, 28);
    for (let i = 0; i < digit.length; i++) {
      imageData.data[i * 4] = digit[i] * 255;
      imageData.data[i * 4 + 1] = digit[i] * 255;
      imageData.data[i * 4 + 2] = digit[i] * 255;
      imageData.data[i * 4 + 3] = 255;
    }

    CTX.putImageData(imageData, 0, 0);
  }

  function drawCustomCircle(event) {
    setIsDrawing(true);
    console.log("mouse clicked");

    // console.log("width of canvas : ", canvasRef.current.clientWidth);
    const [canvasRelativeX, canvasRelativeY] = calculateXYCanvas(event);

    console.log(`x : ${canvasRelativeX}, y : ${canvasRelativeY}`);

    let CTX = canvasRef.current.getContext("2d");

    if (forDrawingPrevX == null && forDrawingPrevY == null) {
      CTX.beginPath();
      CTX.arc(canvasRelativeX, canvasRelativeY, 3, 0, Math.PI * 2);
      CTX.fillStyle = "white";
      CTX.fill();
      CTX.strokeStyle = "white";
      CTX.stroke();
      // setForDrawingPrevX(canvasRelativeX);
      // setForDrawingPrevY(canvasRelativeY);
    } else {
      CTX.beginPath();
      CTX.moveTo(forDrawingPrevX, forDrawingPrevY);
      CTX.lineTo(canvasRelativeX, canvasRelativeY);
      CTX.lineWidth = 10;
      CTX.strokeStyle = "white";
      CTX.stroke();
    }
  }

  function hanleClickedMouseMoveOnCanvas(event) {
    if (isDrawing) {
      let [x, y] = calculateXYCanvas(event);
      setForDrawingPrevX(x);
      setForDrawingPrevY(y);
      drawCustomCircle(event);
    }
  }

  function handleMouseUpOnCanvas() {
    setIsDrawing(false);
    setForDrawingPrevX(null);
    setForDrawingPrevY(null);
  }

  function handleClearCanvas() {
    let ctx = canvasRef.current.getContext("2d");
    ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    // refilling the black background
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
  }

  function calculateXYCanvas(event) {
    let rect = canvasRef.current.getBoundingClientRect();
    //console.log("getBoundingRect", rect);

    let elementRelativeX = event.clientX - rect.left;
    let elementRelativeY = event.clientY - rect.top;

    let canvasRelativeX =
      (elementRelativeX * canvasRef.current.clientWidth) / rect.width;
    let canvasRelativeY =
      (elementRelativeY * canvasRef.current.clientHeight) / rect.height;

    return [canvasRelativeX, canvasRelativeY];
  }

  async function makePrediction() {
    let ctx = canvasRef.current.getContext("2d");
    let imageData = ctx.getImageData(
      0,
      0,
      canvasRef.current.width,
      canvasRef.current.height
    );
    console.log("imageData is : ", imageData);

    //converting to a tensor
    let imageTensor = tf.browser.fromPixels(imageData);
    console.log("imageTensor.shape :", imageTensor.shape);

    //converting to grayscale
    let greyscaleImageTensor = tf.image.rgbToGrayscale(imageTensor);
    // let greyscaleImageTensor = tf.image.rgbToGrayscale(normalizedInput);
    console.log("greyscaleImageTensor.shape : ", greyscaleImageTensor.shape);

    //normalizing
    let normalizedInput = greyscaleImageTensor.div(tf.scalar(255));

    let resizedImageTensor = tf.image.resizeBilinear(
      normalizedInput,
      [28, 28],
      true
    );
    console.log("resizedImageTensor.shape : ", resizedImageTensor.shape);

    // let newInput = tf.reshape(resizedImageTensor, [784]);
    let newInput = resizedImageTensor.flatten();
    console.log("newInput.shape: ", newInput.shape);
    console.log("newInput : ", newInput);

    // Debugging
    let CTX = canvasDebug.current.getContext("2d");
    // let testTensor = tf.reshape(imageTensor, [112896]);

    let imageData2 = CTX.getImageData(0, 0, 28, 28);
    console.log("imageData2.data.length : ", imageData2.data.length);
    let data = await newInput.data();

    console.log("data.length : ", data.length);

    for (let i = 0; i < data.length; i++) {
      // console.log("data[i] : ", data[i]);
      imageData2.data[i * 4] = data[i] * 255;
      imageData2.data[i * 4 + 1] = data[i] * 255;
      imageData2.data[i * 4 + 2] = data[i] * 255;
      imageData2.data[i * 4 + 3] = 255;
    }

    CTX.putImageData(imageData2, 0, 0);
    //END Debugging

    let opt = model.predict(newInput.expandDims());
    opt.print();

    let answer = opt.squeeze().argMax();

    let index = await answer.array();
    setIsPrediction(true);
    setPrediction(index);
    if (index === 0) {
      setPrediction("0");
    }
  }

  return (
    <div>
      <h1>TensorFlow.js MNIST Classifier</h1>

      <p>See console for even more outputs</p>
      {model ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
          }}
        >
          <section className="box">
            <h2>Input Image</h2>
            <p>
              Input Image is a 28x28 pixel greyscale image for MNIST dataset- a
              real hand drawn digit
            </p>
            {isTrainingDone && (
              <>
                <canvas
                  ref={canvasRef}
                  width="168"
                  height="168"
                  onMouseMove={hanleClickedMouseMoveOnCanvas}
                  onMouseDown={drawCustomCircle}
                  onMouseUp={handleMouseUpOnCanvas}
                  style={{
                    borderWidth: 2,
                    borderStyle: "solid",
                    borderColor: "yellow",
                  }}
                ></canvas>
                <button onClick={handleClearCanvas}>Clear</button>
              </>
            )}
          </section>

          <section>
            <button onClick={makePrediction}>Predict</button>
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
        </div>
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
      {/* debugging canvas */}
      <div>
        <h1>Debugging Images</h1>
        <canvas
          ref={canvasDebug}
          width="28"
          height="28"
          style={{ borderStyle: "solid", borderWidth: 2, zoom: 6 }}
        ></canvas>
      </div>
    </div>
  );
}

export default App2;
