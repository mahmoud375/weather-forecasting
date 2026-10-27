/**
 * dashboard.js — Interactive Client Logic for Weather Forecasting Dashboard
 * Handles scenario presets autofill, Chart.js trend visualization, and clipboard helpers.
 */

document.addEventListener("DOMContentLoaded", function () {
    // 1. Scenario Presets
    const presets = {
        summer: {
            temp_max_c: 38.5,
            temp_min_c: 26.0,
            humidity_pct: 42.0,
            wind_speed_max_kmh: 18.0,
            precipitation_mm: 0.0,
            cloud_cover_pct: 10.0,
        },
        winter: {
            temp_max_c: 18.2,
            temp_min_c: 9.5,
            humidity_pct: 68.0,
            wind_speed_max_kmh: 14.0,
            precipitation_mm: 0.2,
            cloud_cover_pct: 45.0,
        },
        spring: {
            temp_max_c: 27.0,
            temp_min_c: 16.5,
            humidity_pct: 50.0,
            wind_speed_max_kmh: 15.0,
            precipitation_mm: 0.0,
            cloud_cover_pct: 20.0,
        },
        khamsin: {
            temp_max_c: 36.0,
            temp_min_c: 22.0,
            humidity_pct: 18.0,
            wind_speed_max_kmh: 42.0,
            precipitation_mm: 0.0,
            cloud_cover_pct: 35.0,
        },
        rain: {
            temp_max_c: 17.0,
            temp_min_c: 11.0,
            humidity_pct: 82.0,
            wind_speed_max_kmh: 22.0,
            precipitation_mm: 12.5,
            cloud_cover_pct: 90.0,
        },
    };

    const presetButtons = document.querySelectorAll(".preset-btn");
    presetButtons.forEach((btn) => {
        btn.addEventListener("click", function () {
            const key = this.getAttribute("data-preset");
            const data = presets[key];
            if (!data) return;

            document.getElementById("id_temp_max_c").value = data.temp_max_c;
            document.getElementById("id_temp_min_c").value = data.temp_min_c;
            document.getElementById("id_humidity_pct").value = data.humidity_pct;
            document.getElementById("id_wind_speed_max_kmh").value = data.wind_speed_max_kmh;
            document.getElementById("id_precipitation_mm").value = data.precipitation_mm;
            document.getElementById("id_cloud_cover_pct").value = data.cloud_cover_pct;

            // Highlight button temporarily
            presetButtons.forEach(b => b.classList.remove("active", "btn-primary"));
            this.classList.add("btn-primary");
            this.classList.remove("btn-outline-secondary");
            setTimeout(() => {
                this.classList.remove("btn-primary");
                this.classList.add("btn-outline-secondary");
            }, 1000);
        });
    });

    // 2. Reset Button
    const resetBtn = document.getElementById("resetBtn");
    if (resetBtn) {
        resetBtn.addEventListener("click", function () {
            document.getElementById("id_temp_max_c").value = "30.5";
            document.getElementById("id_temp_min_c").value = "19.0";
            document.getElementById("id_humidity_pct").value = "52.0";
            document.getElementById("id_wind_speed_max_kmh").value = "16.5";
            document.getElementById("id_precipitation_mm").value = "0.0";
            document.getElementById("id_cloud_cover_pct").value = "25";
        });
    }

    // 3. Chart.js Recent History or Baseline Trend
    const chartCanvas = document.getElementById("forecastHistoryChart");
    if (chartCanvas && typeof Chart !== "undefined") {
        let historyData = [];
        const rawJsonEl = document.getElementById("historyData");
        if (rawJsonEl && rawJsonEl.textContent) {
            try {
                historyData = JSON.parse(rawJsonEl.textContent);
            } catch (e) {
                historyData = [];
            }
        }

        // Default points if session history is empty
        let labels = [];
        let predictedTemps = [];
        let maxTemps = [];
        let minTemps = [];

        if (historyData.length > 0) {
            // Reverse so oldest of recent is on the left
            const reversed = [...historyData].reverse();
            labels = reversed.map((h, i) => h.time || `Run #${i + 1}`);
            predictedTemps = reversed.map(h => h.prediction);
            maxTemps = reversed.map(h => h.temp_max);
            minTemps = reversed.map(h => h.temp_min);
        } else {
            labels = ["Winter Mean", "Spring Mean", "Summer Peak", "Autumn Cooling", "Current Target"];
            predictedTemps = [14.5, 22.0, 29.8, 22.4, 24.2];
            maxTemps = [19.5, 27.5, 36.0, 27.5, 30.5];
            minTemps = [9.5, 16.5, 23.5, 17.5, 19.0];
        }

        new Chart(chartCanvas, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Predicted Mean Temp (°C)",
                        data: predictedTemps,
                        borderColor: "#0dcaf0",
                        backgroundColor: "rgba(13, 202, 240, 0.15)",
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.35,
                        pointBackgroundColor: "#0dcaf0",
                        pointRadius: 4,
                    },
                    {
                        label: "Input Max Temp (°C)",
                        data: maxTemps,
                        borderColor: "#dc3545",
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 2,
                        tension: 0.35,
                    },
                    {
                        label: "Input Min Temp (°C)",
                        data: minTemps,
                        borderColor: "#0d6efd",
                        borderDash: [4, 4],
                        borderWidth: 1.5,
                        pointRadius: 2,
                        tension: 0.35,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: "top",
                        labels: {
                            color: "#adb5bd",
                            boxWidth: 12,
                            font: { family: "'Inter', sans-serif", size: 11 },
                        },
                    },
                    tooltip: {
                        mode: "index",
                        intersect: false,
                    },
                },
                scales: {
                    x: {
                        ticks: { color: "#6c757d", font: { size: 10 } },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                    },
                    y: {
                        ticks: { color: "#6c757d", font: { size: 10 } },
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        title: { display: true, text: "Temperature (°C)", color: "#6c757d", font: { size: 11 } },
                    },
                },
            },
        });
    }
});

// Clipboard Helper
function copyApiCurl() {
    const code = document.getElementById("curlSampleCode");
    if (code) {
        navigator.clipboard.writeText(code.innerText).then(() => {
            alert("Sample cURL command copied to clipboard!");
        });
    }
}
