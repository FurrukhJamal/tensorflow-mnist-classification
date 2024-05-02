import React, { useState } from "react";
import App1 from "./App1";
import App2 from "./App2";
import "./App.css";

import { AppContext } from "./context";

function App() {
  const [showBasicVersion, setShowBasicVersion] = useState(true);

  function handleAppVersionChange() {
    setShowBasicVersion((prev) => !prev);
  }

  return (
    <AppContext.Provider value={{}}>
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
