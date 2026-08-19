const moves = {
    stone: { icon: "✊", label: "Stone" },
    paper: { icon: "✋", label: "Paper" },
    scissor: { icon: "✌️", label: "Scissor" },
    none: { icon: "?", label: "No hand" }
};

const state = {
    playerOneScore: 0,
    playerTwoScore: 0,
    round: 1,
    history: [],
    stream: null,
    timer: null,
    latest: null,
    lockedSignature: "",
    lastSignature: "",
    stableCount: 0,
    busy: false
};

const cameraFeed = document.querySelector("#cameraFeed");
const captureCanvas = document.querySelector("#captureCanvas");
const playerOneScore = document.querySelector("#playerOneScore");
const playerTwoScore = document.querySelector("#playerTwoScore");
const roundCount = document.querySelector("#roundCount");
const playerOneOrb = document.querySelector("#playerOneOrb");
const playerTwoOrb = document.querySelector("#playerTwoOrb");
const playerOneMove = document.querySelector("#playerOneMove");
const playerTwoMove = document.querySelector("#playerTwoMove");
const playerOneConfidence = document.querySelector("#playerOneConfidence");
const playerTwoConfidence = document.querySelector("#playerTwoConfidence");
const resultTitle = document.querySelector("#resultTitle");
const resultText = document.querySelector("#resultText");
const statusText = document.querySelector("#statusText");
const modelStatus = document.querySelector("#modelStatus");
const playerOneDebug = document.querySelector("#playerOneDebug");
const playerTwoDebug = document.querySelector("#playerTwoDebug");
const historyList = document.querySelector("#historyList");
const startBtn = document.querySelector("#startBtn");
const lockBtn = document.querySelector("#lockBtn");
const resetBtn = document.querySelector("#resetBtn");

function moveLabel(move) {
    return moves[move]?.label || "No hand";
}

function moveIcon(move) {
    return moves[move]?.icon || "?";
}

function renderScore() {
    playerOneScore.textContent = state.playerOneScore;
    playerTwoScore.textContent = state.playerTwoScore;
    roundCount.textContent = state.round;
}

function confidenceText(value) {
    const percent = Math.round((value || 0) * 100);
    return percent >= 50 ? `${percent}% confidence` : `${percent}% low confidence`;
}

function scoreText(scores) {
    if (!scores || Object.keys(scores).length === 0) return "no scores";
    return Object.entries(scores)
        .map(([label, score]) => `${label}:${Math.round(score * 100)}%`)
        .join(" ");
}

async function loadModelStatus() {
    try {
        const response = await fetch("/status");
        const data = await response.json();
        const labels = data.labels && data.labels.length ? data.labels.join(", ") : "none";
        const loader = data.loader ? ` loader: ${data.loader}` : "";
        modelStatus.textContent = data.loaded
            ? `MODEL LOADED - labels: ${labels} -${loader}`
            : `FALLBACK MODE -${loader} - ${data.error || "model not loaded"}`;
        modelStatus.className = data.loaded ? "ok" : "warn";
    } catch (error) {
        modelStatus.textContent = "STATUS ERROR - run python app.py";
        modelStatus.className = "warn";
    }
}

function renderPrediction(data) {
    state.latest = data;

    const p1 = data.playerOne;
    const p2 = data.playerTwo;
    const winner = data.winner;
    const signature = `${p1.move}:${p2.move}:${winner.result}`;
    const scoreReady = winner.result !== "waiting" && signature !== state.lockedSignature;

    playerOneOrb.textContent = moveIcon(p1.move);
    playerTwoOrb.textContent = moveIcon(p2.move);
    playerOneMove.textContent = moveLabel(p1.move);
    playerTwoMove.textContent = moveLabel(p2.move);
    playerOneConfidence.textContent = confidenceText(p1.confidence);
    playerTwoConfidence.textContent = confidenceText(p2.confidence);
    playerOneDebug.textContent = `P1 ${p1.source}: ${scoreText(p1.scores)}`;
    playerTwoDebug.textContent = `P2 ${p2.source}: ${scoreText(p2.scores)}`;

    resultTitle.textContent = winner.title;
    resultText.textContent = winner.message;
    lockBtn.disabled = !scoreReady;

    playerOneOrb.className = "choice-orb";
    playerTwoOrb.className = "choice-orb";

    if (winner.result === "p1") {
        playerOneOrb.classList.add("win");
        playerTwoOrb.classList.add("lose");
    } else if (winner.result === "p2") {
        playerOneOrb.classList.add("lose");
        playerTwoOrb.classList.add("win");
    } else if (winner.result === "draw") {
        playerOneOrb.classList.add("draw");
        playerTwoOrb.classList.add("draw");
    }

    if (winner.result === "waiting") {
        state.lockedSignature = "";
        state.lastSignature = "";
        state.stableCount = 0;
        return;
    }

    if (!scoreReady) {
        statusText.textContent = "Remove hands for next round";
        return;
    }

    if (signature === state.lastSignature) {
        state.stableCount += 1;
    } else {
        state.lastSignature = signature;
        state.stableCount = 1;
    }

    // Lock point automatically after 3 stable consecutive frames
    if (state.stableCount >= 3) {
        lockRound(true);
    }
}

function renderHistory() {
    if (state.history.length === 0) {
        historyList.innerHTML = '<li class="empty-history">Locked rounds will appear here.</li>';
        return;
    }

    historyList.innerHTML = state.history
        .slice(0, 8)
        .map((item) => `
            <li>
                <span>${item.title}</span>
                <small>${item.detail}</small>
            </li>
        `)
        .join("");
}

function captureFrame() {
    const width = cameraFeed.videoWidth;
    const height = cameraFeed.videoHeight;
    if (!width || !height) return null;

    const targetWidth = 480;
    const targetHeight = Math.round((height / width) * targetWidth);
    captureCanvas.width = targetWidth;
    captureCanvas.height = targetHeight;
    const context = captureCanvas.getContext("2d");
    context.drawImage(cameraFeed, 0, 0, targetWidth, targetHeight);
    
    // JPEG 0.55 Quality compression for fast network payload
    return captureCanvas.toDataURL("image/jpeg", 0.55);
}

async function predictLoop() {
    if (state.busy || !state.stream) return;

    const image = captureFrame();
    if (!image) return;

    state.busy = true;
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image })
        });

        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed");
        }

        statusText.textContent = "Camera tracking";
        renderPrediction(data);
    } catch (error) {
        statusText.textContent = "Prediction error";
        resultTitle.textContent = "Check Python";
        resultText.textContent = error.message;
    } finally {
        state.busy = false;
    }
}

async function startCamera() {
    if (state.stream) {
        stopCamera();
        return;
    }

    try {
        state.stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        cameraFeed.srcObject = state.stream;
        startBtn.textContent = "Stop Camera";
        lockBtn.disabled = true;
        statusText.textContent = "Camera starting";
        
        // Accelerated loop: 130ms (~8 FPS real-time detection)
        state.timer = window.setInterval(predictLoop, 130);
    } catch (error) {
        statusText.textContent = "Camera blocked";
        resultTitle.textContent = "Camera Needed";
        resultText.textContent = "Allow camera permission and run from localhost.";
    }
}

function stopCamera() {
    if (state.timer) {
        window.clearInterval(state.timer);
        state.timer = null;
    }

    if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
        state.stream = null;
    }

    cameraFeed.srcObject = null;
    startBtn.textContent = "Start Camera";
    lockBtn.disabled = true;
    statusText.textContent = "Camera off";
}

function lockRound(autoScored = false) {
    if (!state.latest || state.latest.winner.result === "waiting") return;

    const winner = state.latest.winner;
    state.lockedSignature = `${state.latest.playerOne.move}:${state.latest.playerTwo.move}:${winner.result}`;
    
    if (winner.result === "p1") state.playerOneScore += 1;
    else if (winner.result === "p2") state.playerTwoScore += 1;

    state.history.unshift({
        title: winner.title,
        detail: `P1 ${moveLabel(state.latest.playerOne.move)} vs P2 ${moveLabel(state.latest.playerTwo.move)}`
    });

    state.round += 1;
    renderScore();
    renderHistory();
    lockBtn.disabled = true;
    state.stableCount = 0;
    statusText.textContent = autoScored ? "Point added automatically" : "Point added";
    resultText.textContent = `${winner.message} Remove hands once for the next round.`;
}

function resetGame() {
    state.playerOneScore = 0;
    state.playerTwoScore = 0;
    state.round = 1;
    state.history = [];
    state.latest = null;
    state.lockedSignature = "";
    state.lastSignature = "";
    state.stableCount = 0;

    playerOneOrb.textContent = "?";
    playerTwoOrb.textContent = "?";
    playerOneOrb.className = "choice-orb";
    playerTwoOrb.className = "choice-orb";
    playerOneMove.textContent = "Waiting";
    playerTwoMove.textContent = "Waiting";
    playerOneConfidence.textContent = "0% confidence";
    playerTwoConfidence.textContent = "0% confidence";
    resultTitle.textContent = "Ready?";
    resultText.textContent = "Start camera. Player 1 use left side, Player 2 use right side.";
    lockBtn.disabled = true;

    renderScore();
    renderHistory();
}

startBtn.addEventListener("click", startCamera);
lockBtn.addEventListener("click", () => lockRound(false));
resetBtn.addEventListener("click", resetGame);

renderScore();
renderHistory();
loadModelStatus();