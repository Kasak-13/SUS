// Chart.js Default configuration for Dark Mode
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.05)';
Chart.defaults.font.family = "'Inter', sans-serif";

const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: {
        duration: 0
    },
    plugins: {
        legend: { display: false }
    },
    scales: {
        x: { display: false },
        y: { beginAtZero: true }
    }
};

// Initialize Charts
const ctxSensor = document.getElementById('sensorChart').getContext('2d');
const sensorChart = new Chart(ctxSensor, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Sensor Mean',
            data: [],
            borderColor: '#3b82f6',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.2
        },
        {
            label: 'True Anomaly Active',
            data: [],
            backgroundColor: 'rgba(239, 68, 68, 0.25)',
            borderColor: 'transparent',
            fill: true,
            pointRadius: 0
        }]
    },
    options: {
        ...commonOptions,
        plugins: {
            legend: { display: false }
        }
    }
});

const ctxAnomaly = document.getElementById('anomalyChart').getContext('2d');
const anomalyChart = new Chart(ctxAnomaly, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Anomaly Fraction',
            data: [],
            borderColor: '#ef4444',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.2
        }]
    },
    options: {
        ...commonOptions,
        scales: {
            y: {
                beginAtZero: true,
                max: 1.0,
                grid: { color: (ctx) => ctx.tick && ctx.tick.value === 0.1 ? 'rgba(239, 68, 68, 0.6)' : 'rgba(255,255,255,0.05)' }
            }
        }
    }
});

const ctxDrift = document.getElementById('driftChart').getContext('2d');
const driftChart = new Chart(ctxDrift, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Drift Score (KS)',
            data: [],
            borderColor: '#10b981',
            borderWidth: 2,
            pointRadius: 0,
            fill: true,
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            tension: 0.2
        }]
    },
    options: {
        ...commonOptions,
        scales: {
            y: {
                beginAtZero: true,
                max: 1.0,
                grid: { color: (ctx) => ctx.tick && ctx.tick.value === 0.75 ? 'rgba(16, 185, 129, 0.6)' : 'rgba(255,255,255,0.05)' }
            }
        }
    }
});

// State Management
let currentData = [];
let currentIndex = 0;
let isPlaying = false;
let playInterval = null;
let currentSpeed = 200; // ms
let currentScenario = 'A';

// Live Injection Modifiers
let activeInjections = {
    spikeCountdown: 0,
    driftAccumulator: 0.0
};

// DOM Elements
const currentStepEl = document.getElementById('currentStep');
const valAnomaly = document.getElementById('valAnomaly');
const valDrift = document.getElementById('valDrift');
const valSafeguard = document.getElementById('valSafeguard');
const decisionDisplay = document.getElementById('decisionDisplay');
const decisionSubtext = document.getElementById('decisionSubtext');
const scenarioDesc = document.getElementById('scenarioDesc');
const playBtn = document.getElementById('playBtn');
const scenarioBtns = document.querySelectorAll('.scenario-btn');
const groundTruthBadge = document.getElementById('groundTruthBadge');

// Timeline & Phase Elements
const timelineSlider = document.getElementById('timelineSlider');
const scrubStepDisplay = document.getElementById('scrubStepDisplay');
const btnJumpStart = document.getElementById('btnJumpStart');
const btnJumpAction = document.getElementById('btnJumpAction');
const phaseBanner = document.getElementById('phaseBanner');
const phaseText = document.getElementById('phaseText');

// 4-System Arena Elements
const badgeStatic = document.getElementById('badgeStatic');
const scoreStatic = document.getElementById('scoreStatic');
const badgeNaive = document.getElementById('badgeNaive');
const scoreNaive = document.getElementById('scoreNaive');
const vulnNaive = document.getElementById('vulnNaive');
const badgeThresh = document.getElementById('badgeThresh');
const scoreThresh = document.getElementById('scoreThresh');
const badgeSUS = document.getElementById('badgeSUS');
const shieldState = document.getElementById('shieldState');

// Chaos Buttons
const btnInjectAnomaly = document.getElementById('btnInjectAnomaly');
const btnInjectDrift = document.getElementById('btnInjectDrift');
const btnResetInjections = document.getElementById('btnResetInjections');
const btnStep = document.getElementById('btnStep');
const speedBtns = document.querySelectorAll('.speed-btn');

// Benchmark Modal Elements
const benchmarkBtn = document.getElementById('benchmarkBtn');
const benchmarkModal = document.getElementById('benchmarkModal');
const closeModalBtn = document.getElementById('closeModalBtn');
const benchmarkTableBody = document.getElementById('benchmarkTableBody');

const descriptions = {
    'A': 'Scenario A (Stable): Clean baseline sensor stream from the SKAB water testbed. Demonstrates zero false alarms and stable KEEP baseline.',
    'B': 'Scenario B (Gradual Drift): 0.001 synthetic drift simulating component aging. Shows SUS adapting while Static throws false alarms.',
    'D': 'Scenario D (Physical Anomaly): Real physical valve & pump faults (steps 173-574). Proves SUS triggers HOLD and resists Anomaly Contamination.'
};

async function fetchScenarioData(scenario) {
    try {
        const response = await fetch(`/api/scenario/${scenario}`);
        currentData = await response.json();
        
        // Update timeline max
        timelineSlider.max = currentData.length - 1;
        timelineSlider.value = 0;
        scrubStepDisplay.innerText = `Step 0 / ${currentData.length}`;
        
        // Update Jump button label
        if (scenario === 'D') {
            btnJumpAction.innerText = "⚡ Jump to Fault (Step 175)";
            btnJumpAction.style.display = "inline-block";
        } else if (scenario === 'B') {
            btnJumpAction.innerText = "🌊 Jump to Drift (Step 105)";
            btnJumpAction.style.display = "inline-block";
        } else {
            btnJumpAction.innerText = "⚡ Inject Shock Spike";
            btnJumpAction.style.display = "inline-block";
        }
        
        resetSimulation();
    } catch (e) {
        console.error("Failed to load scenario data", e);
    }
}

function resetSimulation() {
    currentIndex = 0;
    activeInjections.spikeCountdown = 0;
    activeInjections.driftAccumulator = 0.0;
    
    // Clear charts
    sensorChart.data.labels = [];
    sensorChart.data.datasets[0].data = [];
    sensorChart.data.datasets[1].data = [];
    
    anomalyChart.data.labels = [];
    anomalyChart.data.datasets[0].data = [];
    
    driftChart.data.labels = [];
    driftChart.data.datasets[0].data = [];
    
    updateCharts();
    updateUI(null);
}

function updateCharts() {
    sensorChart.update();
    anomalyChart.update();
    driftChart.update();
}

function jumpToStep(targetIndex) {
    if (!currentData || currentData.length === 0) return;
    targetIndex = Math.max(0, Math.min(targetIndex, currentData.length - 1));
    currentIndex = targetIndex;
    
    // Populate charts with recent historical points up to targetIndex
    const startPoint = Math.max(0, targetIndex - 120);
    sensorChart.data.labels = [];
    sensorChart.data.datasets[0].data = [];
    sensorChart.data.datasets[1].data = [];
    anomalyChart.data.labels = [];
    anomalyChart.data.datasets[0].data = [];
    driftChart.data.labels = [];
    driftChart.data.datasets[0].data = [];
    
    for (let i = startPoint; i <= targetIndex; i++) {
        const item = currentData[i];
        sensorChart.data.labels.push(item.step);
        sensorChart.data.datasets[0].data.push(item.sensor_mean);
        sensorChart.data.datasets[1].data.push(item.is_true_anomaly ? 15 : -15);
        anomalyChart.data.labels.push(item.step);
        anomalyChart.data.datasets[0].data.push(item.anomaly_fraction);
        driftChart.data.labels.push(item.step);
        driftChart.data.datasets[0].data.push(item.drift_score);
    }
    
    updateCharts();
    updateUI(currentData[targetIndex]);
}

function updateUI(stepData) {
    if (!stepData) {
        currentStepEl.innerText = "0";
        timelineSlider.value = 0;
        scrubStepDisplay.innerText = `Step 0 / ${currentData.length || 173}`;
        valAnomaly.innerText = "0.00";
        valDrift.innerText = "0.00";
        valSafeguard.innerText = "Armed";
        decisionDisplay.innerText = "WAITING";
        decisionDisplay.className = "decision-display";
        decisionSubtext.innerText = "Press Play or Jump to start simulation";
        groundTruthBadge.innerText = "Ground Truth: NORMAL (0)";
        groundTruthBadge.className = "ground-truth-badge";
        
        badgeStatic.innerText = "NORMAL";
        badgeStatic.className = "sys-badge normal";
        badgeNaive.innerText = "NORMAL";
        badgeNaive.className = "sys-badge normal";
        badgeThresh.innerText = "NORMAL";
        badgeThresh.className = "sys-badge normal";
        badgeSUS.innerText = "KEEP";
        badgeSUS.className = "sys-badge sus-state keep";
        shieldState.innerText = "Armed (Gated)";
        
        phaseBanner.className = "phase-banner normal";
        phaseText.innerText = "✅ Normal Baseline Operation (Clean Hydraulic Data)";
        return;
    }

    currentStepEl.innerText = stepData.step;
    timelineSlider.value = stepData.step;
    scrubStepDisplay.innerText = `Step ${stepData.step} / ${currentData.length}`;
    valAnomaly.innerText = stepData.anomaly_fraction.toFixed(2);
    valDrift.innerText = stepData.drift_score.toFixed(2);

    // Ground Truth Badge & Phase Banner
    if (stepData.is_true_anomaly === 1 || activeInjections.spikeCountdown > 0) {
        groundTruthBadge.innerText = "Ground Truth: PHYSICAL FAULT ACTIVE (1)";
        groundTruthBadge.className = "ground-truth-badge anomaly";
        
        phaseBanner.className = "phase-banner fault";
        phaseText.innerText = "🚨 REAL PHYSICAL FAULT ACTIVE (SKAB Valve Anomaly) — Watch Naive Adaptive Contaminate & SUS Trigger HOLD!";
    } else if (currentScenario === 'B' && stepData.step >= 100) {
        groundTruthBadge.innerText = "Ground Truth: NORMAL (Drifting Baseline)";
        groundTruthBadge.className = "ground-truth-badge";
        
        phaseBanner.className = "phase-banner drift";
        phaseText.innerText = "🌊 GRADUAL DRIFT IN PROGRESS — Notice SUS entering ADAPT cleanly while Static triggers false alarms!";
    } else {
        groundTruthBadge.innerText = "Ground Truth: NORMAL (0)";
        groundTruthBadge.className = "ground-truth-badge";
        
        phaseBanner.className = "phase-banner normal";
        if (stepData.step < 100) {
            phaseText.innerText = "ℹ️ Initial Warmup Window (Filling 100-step sequential buffers: Normal Baseline)";
        } else {
            phaseText.innerText = "✅ Stable Normal Operation (Hydraulic Flow Clean)";
        }
    }

    // 1. Static Card
    scoreStatic.innerText = stepData.static_score.toFixed(4);
    if (stepData.static_pred === 1) {
        badgeStatic.innerText = "ALERT ⚠️";
        badgeStatic.className = "sys-badge alert";
    } else {
        badgeStatic.innerText = "NORMAL";
        badgeStatic.className = "sys-badge normal";
    }

    // 2. Naive Adaptive Card
    scoreNaive.innerText = stepData.naive_score.toFixed(4);
    if (stepData.is_true_anomaly === 1 && stepData.naive_pred === 0) {
        // CATASTROPHIC CONTAMINATION! Missed true anomaly!
        badgeNaive.innerText = "CONTAMINATED! ❌";
        badgeNaive.className = "sys-badge contaminated";
        vulnNaive.innerText = "🚨 Anomaly Contamination: Blindly learned fault as normal!";
        vulnNaive.style.color = "#f87171";
    } else if (stepData.naive_pred === 1) {
        badgeNaive.innerText = "ALERT ⚠️";
        badgeNaive.className = "sys-badge alert";
        vulnNaive.innerText = "⚠️ Blind retraining every 50 steps";
        vulnNaive.style.color = "#fbbf24";
    } else {
        badgeNaive.innerText = "NORMAL";
        badgeNaive.className = "sys-badge normal";
        vulnNaive.innerText = "⚠️ Vulnerable to Anomaly Contamination";
        vulnNaive.style.color = "#fbbf24";
    }

    // 3. Threshold Card
    scoreThresh.innerText = stepData.thresh_score.toFixed(4);
    if (stepData.thresh_pred === 1) {
        badgeThresh.innerText = "ALERT ⚠️";
        badgeThresh.className = "sys-badge alert";
    } else {
        badgeThresh.innerText = "NORMAL";
        badgeThresh.className = "sys-badge normal";
    }

    // 4. SUS Controller Card
    decisionDisplay.innerText = stepData.decision;
    decisionDisplay.className = `decision-display ${stepData.decision.toLowerCase()}`;
    
    badgeSUS.innerText = stepData.decision;
    badgeSUS.className = `sys-badge sus-state ${stepData.decision.toLowerCase()}`;

    if (stepData.decision === 'HOLD') {
        shieldState.innerText = "EMERGENCY BRAKE (Lockdown)";
        valSafeguard.innerText = "HOLD Lock (10-step timer active)";
        valSafeguard.style.color = "#ef4444";
    } else if (stepData.decision === 'ADAPT') {
        shieldState.innerText = "Safe Adaptation Permit";
        valSafeguard.innerText = "3-Consecutive Gate Passed";
        valSafeguard.style.color = "#3b82f6";
    } else if (stepData.decision === 'KEEP') {
        shieldState.innerText = "Stable Protection Active";
        valSafeguard.innerText = "Armed & Monitoring";
        valSafeguard.style.color = "#10b981";
    } else {
        shieldState.innerText = "Buffering Warmup";
        valSafeguard.innerText = "Buffering (100 steps)";
        valSafeguard.style.color = "#f59e0b";
    }

    // Reset quadrants
    document.querySelectorAll('.quad-cell').forEach(el => {
        el.className = 'quad-cell';
    });

    if (stepData.decision === 'BUFFERING') {
        decisionSubtext.innerText = "Filling initial 100-step warmup buffers (comparing sequential windows)...";
    } else {
        const isHighAnomaly = stepData.anomaly_fraction > 0.10;
        const isHighDrift = stepData.drift_score > 0.75;
        
        let activeId = '';
        if (!isHighAnomaly && !isHighDrift) {
            activeId = 'q-low-low';
            decisionSubtext.innerText = "Normal hydraulic data. No baseline adaptation needed.";
        } else if (isHighAnomaly && !isHighDrift) {
            activeId = 'q-low-high';
            decisionSubtext.innerText = "Isolated anomaly spike. Flagging alert, refusing adaptation.";
        } else if (!isHighAnomaly && isHighDrift) {
            activeId = 'q-high-low';
            decisionSubtext.innerText = "Legitimate distributional drift. Gated ADAPT permitted.";
        } else {
            activeId = 'q-high-high';
            decisionSubtext.innerText = "Massive anomaly contamination suspected! Emergency HOLD engaged.";
        }
        
        if (activeId) {
            const el = document.getElementById(activeId);
            el.className = `quad-cell active ${stepData.decision.toLowerCase()}`;
        }
    }
}

function tick() {
    if (currentIndex >= currentData.length) {
        pauseSimulation();
        playBtn.innerText = "↺ Restart Simulation";
        return;
    }

    // Clone step data
    const stepData = { ...currentData[currentIndex] };
    
    // Apply live chaos modifications if active
    if (activeInjections.spikeCountdown > 0) {
        stepData.sensor_mean *= 4.5;
        stepData.is_true_anomaly = 1;
        stepData.anomaly_fraction = Math.min(1.0, stepData.anomaly_fraction + 0.35);
        stepData.decision = 'HOLD';
        stepData.static_pred = 1;
        stepData.sus_pred = 1;
        activeInjections.spikeCountdown--;
    }

    if (activeInjections.driftAccumulator > 0) {
        stepData.sensor_mean += activeInjections.driftAccumulator;
        stepData.drift_score = Math.min(1.0, stepData.drift_score + 0.25);
        activeInjections.driftAccumulator += 0.05;
    }
    
    // Push data to charts
    sensorChart.data.labels.push(stepData.step);
    sensorChart.data.datasets[0].data.push(stepData.sensor_mean);
    sensorChart.data.datasets[1].data.push(stepData.is_true_anomaly ? 15 : -15);
    
    anomalyChart.data.labels.push(stepData.step);
    anomalyChart.data.datasets[0].data.push(stepData.anomaly_fraction);
    
    driftChart.data.labels.push(stepData.step);
    driftChart.data.datasets[0].data.push(stepData.drift_score);

    // Keep visible rolling window
    const MAX_POINTS = 220;
    if (sensorChart.data.labels.length > MAX_POINTS) {
        sensorChart.data.labels.shift();
        sensorChart.data.datasets[0].data.shift();
        sensorChart.data.datasets[1].data.shift();
        anomalyChart.data.labels.shift();
        anomalyChart.data.datasets[0].data.shift();
        driftChart.data.labels.shift();
        driftChart.data.datasets[0].data.shift();
    }

    updateCharts();
    updateUI(stepData);

    currentIndex++;
}

function playSimulation() {
    if (currentIndex >= currentData.length) {
        resetSimulation();
    }
    isPlaying = true;
    playBtn.innerText = "⏸ Pause Simulation";
    clearInterval(playInterval);
    playInterval = setInterval(tick, currentSpeed);
}

function pauseSimulation() {
    isPlaying = false;
    playBtn.innerText = "▶ Play Simulation";
    clearInterval(playInterval);
}

// Event Listeners
playBtn.addEventListener('click', () => {
    if (isPlaying) {
        pauseSimulation();
    } else {
        playSimulation();
    }
});

btnStep.addEventListener('click', () => {
    if (isPlaying) pauseSimulation();
    tick();
});

// Timeline Navigation Slider
timelineSlider.addEventListener('input', (e) => {
    if (isPlaying) pauseSimulation();
    const val = parseInt(e.target.value, 10);
    jumpToStep(val);
});

// Quick Jump Buttons
btnJumpStart.addEventListener('click', () => {
    if (isPlaying) pauseSimulation();
    jumpToStep(0);
});

btnJumpAction.addEventListener('click', () => {
    if (isPlaying) pauseSimulation();
    if (currentScenario === 'D') {
        jumpToStep(175); // Jump directly into physical anomaly
    } else if (currentScenario === 'B') {
        jumpToStep(105); // Jump directly into active drift
    } else {
        // Scenario A: inject shock
        activeInjections.spikeCountdown = 10;
        tick();
    }
});

scenarioBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        scenarioBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        
        currentScenario = e.target.dataset.scenario;
        scenarioDesc.innerText = descriptions[currentScenario];
        pauseSimulation();
        fetchScenarioData(currentScenario);
    });
});

speedBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        speedBtns.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        currentSpeed = parseInt(e.target.dataset.speed, 10);
        if (isPlaying) {
            clearInterval(playInterval);
            playInterval = setInterval(tick, currentSpeed);
        }
    });
});

// Chaos Lab Buttons
btnInjectAnomaly.addEventListener('click', () => {
    activeInjections.spikeCountdown = 10; // 10-step strong shock
    btnInjectAnomaly.innerText = "⚡ Shock Injected!";
    setTimeout(() => {
        btnInjectAnomaly.innerText = "⚡ Inject Shock Anomaly";
    }, 1200);
    // Immediately tick so the shock appears instantly on screen!
    tick();
});

btnInjectDrift.addEventListener('click', () => {
    activeInjections.driftAccumulator = 0.5;
    btnInjectDrift.innerText = "🌊 Drift Accelerating!";
    setTimeout(() => {
        btnInjectDrift.innerText = "🌊 Accelerate Sensor Drift";
    }, 1200);
    tick();
});

btnResetInjections.addEventListener('click', () => {
    activeInjections.spikeCountdown = 0;
    activeInjections.driftAccumulator = 0.0;
    btnResetInjections.innerText = "✓ Cleared";
    setTimeout(() => {
        btnResetInjections.innerText = "↺ Reset Injections";
    }, 1000);
    updateUI(currentData[currentIndex]);
});

// Benchmark Modal Logic
benchmarkBtn.addEventListener('click', async () => {
    benchmarkModal.classList.add('open');
    if (benchmarkTableBody.children.length === 0) {
        try {
            const res = await fetch('/api/benchmark');
            const data = await res.json();
            benchmarkTableBody.innerHTML = '';
            
            data.forEach(row => {
                const tr = document.createElement('tr');
                if (row.Detector.includes('SUS')) {
                    tr.className = 'highlight-sus';
                }
                
                let contamText = '<span class="tag-clean">None</span>';
                if (row.Scenario.includes('D')) {
                    if (row.Detector.includes('Naive')) {
                        contamText = '<span class="tag-contam">CATASTROPHIC (95% Missed)</span>';
                    } else if (row.Detector.includes('SUS')) {
                        contamText = '<span class="tag-protected">PROTECTED (HOLD Shield)</span>';
                    } else if (row.Detector.includes('Static')) {
                        contamText = '<span class="tag-clean">Immune (Frozen)</span>';
                    }
                } else if (row.Scenario.includes('B') && row.Detector.includes('Static')) {
                    contamText = '<span class="tag-alarm">High False Alarms (FPR)</span>';
                }
                
                tr.innerHTML = `
                    <td><strong>${row.Scenario}</strong></td>
                    <td>${row.Detector}</td>
                    <td>${(row.Precision * 100).toFixed(1)}%</td>
                    <td>${(row.Recall * 100).toFixed(1)}%</td>
                    <td>${row.F1_Score.toFixed(4)}</td>
                    <td>${(row['False_Positive_Rate (FPR)'] * 100).toFixed(1)}%</td>
                    <td>${(row['False_Negative_Rate (FNR)'] * 100).toFixed(1)}%</td>
                    <td>${contamText}</td>
                `;
                benchmarkTableBody.appendChild(tr);
            });
        } catch (e) {
            console.error("Failed to load benchmark table", e);
        }
    }
});

closeModalBtn.addEventListener('click', () => {
    benchmarkModal.classList.remove('open');
});

benchmarkModal.addEventListener('click', (e) => {
    if (e.target === benchmarkModal) {
        benchmarkModal.classList.remove('open');
    }
});

// Initial load
scenarioDesc.innerText = descriptions['A'];
fetchScenarioData('A');
