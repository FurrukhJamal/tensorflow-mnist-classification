import logo from "./logo.svg";
import * as tf from "@tensorflow/tfjs";
import { useEffect, useState, useRef, useContext } from "react";
import "./App.css";

// import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";
import { AppContext } from "./context";

function App2() {
  const [isPrediction, setIsPrediction] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const canvasRef = useRef(null);
  const [forDrawingPrevX, setForDrawingPrevX] = useState(null);
  const [forDrawingPrevY, setForDrawingPrevY] = useState(null);

  const [showBasicVersionApp, setShowBasicVersionApp] = useState(true);

  const canvasDebug = useRef(null);

  const {
    isTrainingDone,
    model,
    inputs,
    outputs,
    inputsTensor,
    outputsTensor,
  } = useContext(AppContext);

  useEffect(() => {
    if (model && isTrainingDone) {
      //initializing the canvas background color to be black
      let ctx = canvasRef.current.getContext("2d");
      ctx.fillstyle = "black";
      ctx.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  }, [isTrainingDone, model]);

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
      CTX.lineWidth = 14;
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

    //clearing the prediction and debug image canvas
    setPrediction(null);
    let ctxDebug = canvasDebug.current.getContext("2d");
    ctxDebug.clearRect(
      0,
      0,
      canvasDebug.current.width,
      canvasDebug.current.height
    );
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

    // Code For Image Debugging
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
              Draw a number between 0 and 9 and then click Predict to see what
              the AI thinks the number is. Please try to draw on the center of
              the area below to get best results
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
                <p id="prediction">Awaiting your input</p>
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
        <p>
          This is the image zoomed by a factor of 6, what the model will see
        </p>
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
