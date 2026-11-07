"use strict";

// ==========================================
// MARKETPULSE AI - PROFESSIONAL DASHBOARD
// ==========================================

let priceChart = null;
let selectedSymbol = "^NSEI";
let requestController = null;

const money = (value) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }

    return "₹" + Number(value).toLocaleString("en-IN", {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    });
};

const formatPercent = (value) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value))) {
        return "—";
    }

    return `${Number(value).toFixed(4)}%`;
};

// ==========================================
// DOM HELPERS
// ==========================================

function getElement(id) {
    return document.getElementById(id);
}

function setText(id, value) {
    const element = getElement(id);

    if (element) {
        element.textContent = value ?? "—";
    }
}

function showError(message) {
    const element = getElement("errorMessage");

    if (!element) return;

    element.textContent = message;
    element.hidden = false;
}

function clearError() {
    const element = getElement("errorMessage");

    if (!element) return;

    element.textContent = "";
    element.hidden = true;
}

function setLoading(loading) {
    document.body.classList.toggle("loading", loading);
}

function setConnectionStatus(message, connected = false) {
    const element = getElement("connectionStatus");

    if (!element) return;

    element.textContent = message;

    const pill = element.closest(".status-pill");

    if (pill) {
        pill.style.background = connected
            ? "rgba(34,214,155,.09)"
            : "rgba(255,98,123,.09)";

        pill.style.borderColor = connected
            ? "rgba(34,214,155,.25)"
            : "rgba(255,98,123,.25)";

        pill.style.color = connected ? "#22d69b" : "#ff8292";
    }
}

// ==========================================
// CHART CONFIGURATION
// ==========================================

function renderChart(data) {
    const canvas = getElement("predictionChart");
    const fallback = getElement("chartFallback");

    if (!canvas || !fallback) return;

    if (typeof Chart === "undefined") {
        canvas.hidden = true;
        fallback.hidden = false;
        fallback.textContent =
            "Chart library did not load. Check your internet connection or CDN access.";
        return;
    }

    if (!Array.isArray(data.labels) || !Array.isArray(data.prices)) {
        canvas.hidden = true;
        fallback.hidden = false;
        fallback.textContent = "Historical chart data is unavailable.";
        return;
    }

    if (data.labels.length !== data.prices.length || data.labels.length === 0) {
        canvas.hidden = true;
        fallback.hidden = false;
        fallback.textContent = "No valid historical price data available.";
        return;
    }

    canvas.hidden = false;
    fallback.hidden = true;

    // Historical values
    const labels = [...data.labels];
    const actual = [...data.prices];

    // Forecast values
    const forecast = new Array(actual.length).fill(null);

    if (
        data.model_available &&
        data.forecast_price !== null &&
        data.forecast_price !== undefined
    ) {
        forecast[forecast.length - 1] = data.current_price;

        labels.push(data.forecast_label || "Next session*");

        actual.push(null);
        forecast.push(data.forecast_price);
    }

    const ctx = canvas.getContext("2d");

    if (!ctx) {
        fallback.hidden = false;
        fallback.textContent = "Unable to initialize chart.";
        return;
    }

    // Create gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 350);

    gradient.addColorStop(0, "rgba(50,197,237,0.24)");
    gradient.addColorStop(1, "rgba(50,197,237,0.00)");

    // Destroy old chart before creating new one
    if (priceChart) {
        priceChart.destroy();
        priceChart = null;
    }

    priceChart = new Chart(ctx, {
        type: "line",

        data: {
            labels: labels,

            datasets: [
                {
                    label: "Historical close",
                    data: actual,

                    borderColor: "#32c5ed",
                    backgroundColor: gradient,

                    borderWidth: 2.5,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    pointHoverBackgroundColor: "#32c5ed",
                    pointHoverBorderColor: "#ffffff",
                    pointHoverBorderWidth: 2,

                    tension: 0.3,
                    fill: true,
                    spanGaps: false
                },

                {
                    label: "Model estimate",
                    data: forecast,

                    borderColor: "#f3b34c",
                    backgroundColor: "#f3b34c",

                    borderWidth: 2.5,
                    borderDash: [7, 5],

                    pointRadius: 4,
                    pointHoverRadius: 7,
                    pointBackgroundColor: "#f3b34c",
                    pointBorderColor: "#ffffff",
                    pointBorderWidth: 2,

                    tension: 0,
                    fill: false,
                    spanGaps: true
                }
            ]
        },

        options: {
            responsive: true,
            maintainAspectRatio: false,

            animation: {
                duration: 850,
                easing: "easeOutQuart"
            },

            interaction: {
                mode: "index",
                intersect: false
            },

            plugins: {
                legend: {
                    display: false
                },

                tooltip: {
                    enabled: true,

                    backgroundColor: "#172941",
                    titleColor: "#ffffff",
                    bodyColor: "#c5d5ec",

                    borderColor: "#344d6c",
                    borderWidth: 1,

                    padding: 12,
                    cornerRadius: 9,

                    displayColors: true,

                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label}: ${money(context.parsed.y)}`;
                        }
                    }
                }
            },

            scales: {
                x: {
                    grid: {
                        display: false
                    },

                    border: {
                        display: false
                    },

                    ticks: {
                        maxTicksLimit: 9,
                        color: "#8192ad",
                        font: {
                            family: "DM Sans",
                            size: 10
                        }
                    }
                },

                y: {
                    grid: {
                        color: "rgba(129,146,173,.13)"
                    },

                    border: {
                        display: false
                    },

                    ticks: {
                        color: "#8192ad",

                        font: {
                            family: "DM Sans",
                            size: 10
                        },

                        callback: function(value) {
                            return "₹" + Number(value).toLocaleString("en-IN");
                        }
                    }
                }
            }
        }
    });
}

// ==========================================
// UPDATE DASHBOARD METRICS
// ==========================================

function updateCards(data) {

    setText("selectedCompany", data.company);
    setText("chartTitle", `${data.company} price movement`);

    setText("currentPrice", money(data.current_price));

    setText(
        "dataDate",
        `Latest available close · ${data.data_date || "Date unavailable"}`
    );

    const predictedPrice = getElement("predictedPrice");
    const predictedReturn = getElement("predictedReturn");
    const direction = getElement("direction");

    if (!predictedPrice || !predictedReturn || !direction) {
        return;
    }

    // Reset direction styling
    direction.className = "metric-value direction-value";

    if (data.model_available && data.prediction) {

        const prediction = data.prediction;

        predictedPrice.textContent = money(prediction.predicted_price);

        predictedReturn.textContent = formatPercent(
            prediction.predicted_return_percent
        );

        const directionText = String(
            prediction.direction || "UNKNOWN"
        ).toUpperCase();

        direction.textContent = directionText;

        if (["UP", "DOWN", "FLAT"].includes(directionText)) {
            direction.classList.add(directionText.toLowerCase());
        }

        setText("modelStatus", "Stock-specific model loaded");

        setText(
            "notice",
            data.model_note || "Experimental model output; not a guaranteed forecast."
        );

    } else {

        predictedPrice.textContent = "Not trained";
        predictedReturn.textContent = "—";
        direction.textContent = "—";

        setText("modelStatus", "Train this stock's model first");

        setText(
            "notice",
            `${data.company}: historical chart is available, but no stock-specific trained model is installed. The NIFTY model is not reused for this company.`
        );
    }
}

// ==========================================
// WATCHLIST ACTIVE STATE
// ==========================================

function updateWatchlist(symbol) {

    document.querySelectorAll(".stock-button").forEach(button => {

        const isActive = button.dataset.symbol === symbol;

        button.classList.toggle("active", isActive);

        button.setAttribute(
            "aria-pressed",
            String(isActive)
        );
    });
}

// ==========================================
// LOAD STOCK DATA FROM FLASK API
// ==========================================

async function loadStock(symbol) {

    selectedSymbol = symbol;

    clearError();

    setLoading(true);

    setConnectionStatus("Loading data");

    updateWatchlist(symbol);

    // Cancel an older request if the user quickly selects another stock.
    if (requestController) {
        requestController.abort();
    }

    requestController = new AbortController();

    const currentController = requestController;

    try {

        const response = await fetch(
            `/api/dashboard?symbol=${encodeURIComponent(symbol)}`,
            {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                },
                signal: currentController.signal
            }
        );

        if (!response.ok) {
            throw new Error(
                `Server error: ${response.status} ${response.statusText}`
            );
        }

        const result = await response.json();

        if (!result.success || !result.data) {
            throw new Error(
                result.error || "Could not load stock data."
            );
        }

        // Prevent older responses from replacing the latest selection.
        if (currentController !== requestController) return;

        updateCards(result.data);

        renderChart(result.data);

        setConnectionStatus("Data connected", true);

    } catch (error) {

        if (error.name === "AbortError") {
            return;
        }

        showError(error.message);

        setConnectionStatus("Connection issue");

        setText("notice", "Could not load market data. Please try again.");

    } finally {

        if (currentController === requestController) {
            setLoading(false);
        }
    }
}

// ==========================================
// INITIALIZE DASHBOARD
// ==========================================

document.addEventListener("DOMContentLoaded", () => {

    // Watchlist buttons
    document.querySelectorAll(".stock-button").forEach(button => {

        button.addEventListener("click", () => {

            const symbol = button.dataset.symbol;

            if (symbol && symbol !== selectedSymbol) {
                loadStock(symbol);
            }
        });
    });

    // Initial stock loading
    loadStock(selectedSymbol);

});
