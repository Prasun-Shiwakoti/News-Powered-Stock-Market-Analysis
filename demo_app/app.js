// Application State
const state = {
	allData: null,
	testData: null,
	demoData: [],
	currentDay: null,
	predictions: [],
	actualPrices: [],
	predictedPrices: [],
	dates: [],
	directionPredictions: [],
	pendingActualPrices: [],
	modelLoaded: false,
	hasSentiment: false,
	selectedTicker: null,
	minDemoWindow: 100,

	// Trading State
	totalMoney: 1000,
	moneyInStocks: 0,
	stocksOwned: 0,
	aiSuggestion: "Analyzing...",
};

// Initialize the application
async function init() {
	console.log("Initializing application...");
	showLoading(true);

	try {
		await loadTestData();
		await loadModel();
		populateTickerSelect();
		initializeDemoState();
		renderChart();
		showLoading(false);
		setupEventListeners();
		await checkAndMakePrediction();

		updateTradingDisplay();
	} catch (error) {
		console.error("Initialization error:", error);
		showError("Failed to initialize application: " + error.message);
		showLoading(false);
	}
}

// Load test data
async function loadTestData() {
	try {
		const response = await fetch(`/api/data/test`);
		const data = await response.json();
		if (!response.ok) throw new Error(data.error || "Failed to load data");

		state.allData = data.data;
		state.hasSentiment = data.has_sentiment;
		console.log(`Loaded ${state.allData.length} days of test data`);
	} catch (error) {
		console.error("Error loading test data:", error);
		throw error;
	}
}

// Load Model
async function loadModel() {
	updateModelStatus("loading", "Loading model...");
	try {
		const modelType = document.getElementById("modelSelect").value;
		if (modelType === "lstm") {
			const cfg = await (await fetch("/api/model/config")).json();
			if (cfg.success) state.minDemoWindow = Math.max(cfg.minDemoWindow, 100);
		}

		const res = await fetch(`/api/model/load`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ model_type: modelType }),
		});
		const data = await res.json();
		if (!res.ok) throw new Error(data.error || "Failed to load model");

		state.modelLoaded = data.success;

		if (state.modelLoaded)
			updateModelStatus("success", "✅ Model loaded successfully!");
		else throw new Error(data.error || "Unknown model error");
	} catch (error) {
		console.error("Error loading model:", error);
		updateModelStatus("error", "❌ Failed to load model");
		state.modelLoaded = false;
		throw error;
	}
}

// Populate Ticker Select Dropdown
function populateTickerSelect() {
	const tickers = [...new Set(state.allData.map((d) => d.ticker))];
	const select = document.getElementById("tickerSelect");
	select.innerHTML = "";
	tickers.forEach((t) => {
		const option = document.createElement("option");
		option.value = t;
		option.textContent = t;
		select.appendChild(option);
	});
	state.selectedTicker = tickers[0];
	select.value = state.selectedTicker;
}

// Filter data for selected ticker
function filterDataForTicker() {
	state.testData = state.allData.filter(
		(d) => d.ticker === state.selectedTicker
	);
}

// Initialize Demo State
function initializeDemoState() {
	filterDataForTicker();
	state.demoData = state.testData.slice(0, state.minDemoWindow);
	state.currentDay = state.minDemoWindow;
	updateInfoDisplay();
	resetTradingState();
	updateTradingDisplay();
}

// Handle Ticker and Model Change
async function handleTickerChange(e) {
	state.selectedTicker = e.target.value;
	showLoading(true);
	await handleReset(true);
	showLoading(false);
}

async function handleModelChange(e) {
	showLoading(true);
	await loadModel();
	await handleReset(true);
	showLoading(false);
}

// Trading Logic
function resetTradingState() {
	state.totalMoney = 10000;
	state.moneyInStocks = 0;
	state.stocksOwned = 0;
	state.aiSuggestion = "Analyzing...";
	updateTradingDisplay();
}

function updateTradingDisplay() {
	document.getElementById("totalMoney").textContent =
		state.totalMoney.toFixed(2);
	document.getElementById("moneyInStocks").textContent =
		state.moneyInStocks.toFixed(2);
	document.getElementById("stocksOwned").textContent = state.stocksOwned;

	const suggestionBox = document.getElementById("aiSuggestionBox");
	const text = document.getElementById("aiSuggestionText");
	text.textContent = state.aiSuggestion;

	suggestionBox.className = "ai-suggestion-box"; // reset
	if (state.aiSuggestion.includes("BUY")) suggestionBox.classList.add("buy");
	else if (state.aiSuggestion.includes("SELL"))
		suggestionBox.classList.add("sell");
	else if (state.aiSuggestion.includes("HOLD"))
		suggestionBox.classList.add("hold");
	else suggestionBox.classList.add("wait");
}

function handleBuy() {
	if (state.currentDay >= state.testData.length) {
		alert("Market Not opened!");
		return;
	}

	const todayPrice = state.testData[state.currentDay - 1].close;
	if (state.totalMoney < todayPrice) {
		alert("You don't have enough money to buy any stocks!");
		return;
	}

	const maxStocks = Math.floor(state.totalMoney / todayPrice);
	state.stocksOwned = maxStocks;
	state.moneyInStocks = maxStocks * todayPrice;
	state.totalMoney -= state.moneyInStocks;

	console.log(`Bought ${maxStocks} stocks at $${todayPrice.toFixed(2)} each`);
	updateTradingDisplay();
	handleNextDay();
}

function handleSell() {
	if (state.stocksOwned === 0) {
		alert("You don't own any stocks to sell!");
		return;
	}
	if (state.currentDay >= state.testData.length) {
		alert("Market Not opened!");
		return;
	}

	const todayPrice = state.testData[state.currentDay - 1].close;
	const saleAmount = state.stocksOwned * todayPrice;
	state.totalMoney += saleAmount;
	state.stocksOwned = 0;
	state.moneyInStocks = 0;

	console.log(
		`Sold all stocks at $${todayPrice.toFixed(
			2
		)} each for $${saleAmount.toFixed(2)}`
	);
	updateTradingDisplay();
	handleNextDay();
}

// Prediction & AI Suggestion
async function checkAndMakePrediction() {
	if (!state.modelLoaded) return;
	const shouldPredict =
		state.demoData.length < state.testData.length &&
		state.predictions.length <=
			Math.max(0, state.demoData.length - state.minDemoWindow);

	if (shouldPredict) await makePrediction();
	if (state.predictions.length > 0 && state.currentDay < state.testData.length)
		enableButtons(true);
}

async function makePrediction() {
	try {
		const response = await fetch(`api/predict`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				data: state.demoData,
				has_sentiment: state.hasSentiment,
				model_type: document.getElementById("modelSelect").value,
			}),
		});

		if (!response.ok) throw new Error("Prediction failed");
		const result = await response.json();
		const prediction = result.prediction;

		const nextDayIdx = state.demoData.length;
		const nextDayData = state.testData[nextDayIdx];
		const nextDate = nextDayData.date;

		const currentPrice = state.demoData[state.demoData.length - 1].close;
		const predictedPrice = currentPrice * (1 + prediction);

		state.predictions.push(prediction);
		state.predictedPrices.push(predictedPrice);
		state.dates.push(nextDate);
		state.pendingActualPrices.push(nextDayData.close);

		updateAISuggestion(predictedPrice, currentPrice);

		renderChart();

		console.log(
			`Predicted ${nextDate}: ${predictedPrice.toFixed(2)} (change ${(
				prediction * 100
			).toFixed(2)}%)`
		);
	} catch (error) {
		console.error("Error making prediction:", error);
		showError("Failed to make prediction: " + error.message);
	}
}

function updateAISuggestion(predictedPrice, currentPrice) {
	const change = predictedPrice - currentPrice;
	let suggestion = "WAIT";

	if (state.stocksOwned === 0) {
		if (change > 0.2)
			suggestion = "BUY 📈 (Expected +$" + change.toFixed(2) + ")";
		else if (change > 0)
			suggestion = "WAIT (LOW INCREASE) 📉 (Expected +$" + change.toFixed(2) + ")";
		else
			suggestion = "WAIT 📉 (Expected -$" + Math.abs(change).toFixed(2) + ")";
	} else {
		if (change > 0.2)
			suggestion = "BUY 📈 (Expected +$" + change.toFixed(2) + ")";
		else if (change > 0)
			suggestion = "HOLD (LOW INCREASE) 📉 (Expected +$" + change.toFixed(2) + ")";
		else
			suggestion = "SELL 📉 (Expected -$" + Math.abs(change).toFixed(2) + ")";
	}

	state.aiSuggestion = suggestion;
	console.log("AI Suggestion:", suggestion);
	updateTradingDisplay();
}

// Advance to Next Day
async function handleNextDay() {
	enableButtons(false);
	await advanceToNextDay();
}

async function advanceToNextDay() {
	if (state.currentDay >= state.testData.length) return;

	const nextDayIdx = state.demoData.length;
	const nextDayData = state.testData[nextDayIdx];
	state.demoData.push(nextDayData);

	if (state.pendingActualPrices.length > 0) {
		const actualPrice = state.pendingActualPrices.shift();
		state.actualPrices.push(actualPrice);
	}

	state.currentDay++;
	updateInfoDisplay();
	renderChart();
	await checkAndMakePrediction();
}

// Reset Demo
async function handleReset(isSwitchingTicker = false) {
	if (
		!isSwitchingTicker &&
		!confirm("Are you sure you want to reset the demo?")
	)
		return;

	initializeDemoState();
	state.predictions = [];
	state.actualPrices = [];
	state.predictedPrices = [];
	state.dates = [];
	state.directionPredictions = [];
	state.pendingActualPrices = [];
	updateInfoDisplay();
	enableButtons(false);
	renderChart();
	await checkAndMakePrediction();
}

// UI
function renderChart() {
	const traces = [];

	// --- Historical Data (Blue) ---
	const histDates = state.testData
		.slice(0, state.minDemoWindow)
		.map((d) => d.date);
	const histPrices = state.testData
		.slice(0, state.minDemoWindow)
		.map((d) => d.close);

	traces.push({
		x: histDates,
		y: histPrices,
		mode: "lines",
		name: `Historical (${state.selectedTicker})`,
		line: { color: "#1f77b4", width: 2 },
	});

	// --- Actual Future Prices (Green) ---
	if (state.dates.length > 0 && state.actualPrices.length > 0) {
		traces.push({
			x: state.dates.slice(0, state.actualPrices.length),
			y: state.actualPrices,
			mode: "lines+markers",
			name: "Actual Prices",
			line: { color: "#2ca02c", width: 3 },
			marker: { size: 6, symbol: "square", color: "#2ca02c" },
		});
	}

	// --- Predicted Future Prices (Red) ---
	if (state.dates.length > 0 && state.predictedPrices.length > 0) {
		traces.push({
			x: state.dates,
			y: state.predictedPrices,
			mode: "lines+markers",
			name: "Predicted Prices",
			line: { color: "#d62728", width: 2 },
			marker: { size: 6, symbol: "triangle-up", color: "#d62728" },
		});
	}

	// --- Vertical Dotted Lines for Each Prediction Date ---
	const shapes = [];
	if (state.dates.length > 0) {
		state.dates.forEach((date) => {
			shapes.push({
				type: "line",
				xref: "x",
				yref: "paper",
				x0: date,
				x1: date,
				y0: 0,
				y1: 1,
				line: {
					color: "#e2e2e2ff",
					width: 0.1,
				},
			});
		});
	}

	// --- Plot Layout ---
	Plotly.newPlot(
		"chart",
		traces,
		{
			template: "plotly_dark",
			paper_bgcolor: "rgba(0,0,0,0)",
			plot_bgcolor: "rgba(0,0,0,0)",
			margin: { t: 20, b: 60, l: 60, r: 20 },
			font: { color: "#ffffff" },
			shapes: shapes,
		},
		{ responsive: true, displaylogo: false }
	);
}

function updateModelStatus(status, msg) {
	const ind = document.querySelector(".status-indicator");
	const txt = document.querySelector(".status-text");
	ind.className = "status-indicator " + status;
	txt.textContent = msg;
}

function updateInfoDisplay() {
	document.getElementById(
		"currentDay"
	).textContent = `Day: ${state.currentDay}`;
	document.getElementById(
		"totalPredictions"
	).textContent = `Predictions: ${state.predictions.length}`;
}

function enableButtons(enable) {
	document.getElementById("nextBtn").disabled = !enable;
	document.getElementById("buyBtn").disabled = !enable;
	document.getElementById("sellBtn").disabled = !enable;
}

function showLoading(show) {
	const overlay = document.getElementById("loadingOverlay");
	if (show) overlay.classList.remove("hidden");
	else overlay.classList.add("hidden");
}

function showError(msg) {
	alert("Error: " + msg);
}

// Event Listeners
function setupEventListeners() {
	document.getElementById("nextBtn").addEventListener("click", handleNextDay);
	document
		.getElementById("resetBtn")
		.addEventListener("click", () => handleReset(false));
	document
		.getElementById("tickerSelect")
		.addEventListener("change", handleTickerChange);
	document
		.getElementById("modelSelect")
		.addEventListener("change", handleModelChange);
	document.getElementById("buyBtn").addEventListener("click", handleBuy);
	document.getElementById("sellBtn").addEventListener("click", handleSell);
}

// Initialize
document.addEventListener("DOMContentLoaded", init);
