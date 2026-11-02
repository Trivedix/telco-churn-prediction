document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("churnForm");
  const resultCard = document.getElementById("resultCard");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => (data[key] = value));

    // Show loading state
    resultCard.innerHTML = `
      <div class="spinner-border text-primary" style="width:3rem;height:3rem;" role="status"></div>
      <h5 class="mt-3 text-muted">Analyzing customer profile...</h5>
    `;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });

      const result = await response.json();

      if (result.error) {
        resultCard.innerHTML = `
          <i class="fa-solid fa-triangle-exclamation result-icon" style="color:#ff4757;"></i>
          <h5 class="text-danger">Something went wrong</h5>
          <p class="text-muted small">${result.error}</p>
        `;
        return;
      }

      renderResult(result);
    } catch (err) {
      resultCard.innerHTML = `
        <i class="fa-solid fa-triangle-exclamation result-icon" style="color:#ff4757;"></i>
        <h5 class="text-danger">Request failed</h5>
        <p class="text-muted small">${err.message}</p>
      `;
    }
  });

  function renderResult(result) {
    const riskClass =
      result.risk_level === "High" ? "risk-high" :
      result.risk_level === "Medium" ? "risk-medium" : "risk-low";

    const icon = result.churn
      ? '<i class="fa-solid fa-triangle-exclamation result-icon" style="color:#ff4757;"></i>'
      : '<i class="fa-solid fa-circle-check result-icon" style="color:#2ed573;"></i>';

    resultCard.innerHTML = `
      ${icon}
      <h4 class="mt-2">${result.churn_label}</h4>

      <div class="progress-ring-wrap">
        <svg width="160" height="160">
          <circle cx="80" cy="80" r="70" stroke="#eee" stroke-width="12" fill="none"></circle>
          <circle id="progressCircle" cx="80" cy="80" r="70" stroke="url(#grad)" stroke-width="12"
                  fill="none" stroke-linecap="round"
                  stroke-dasharray="440" stroke-dashoffset="440"
                  transform="rotate(-90 80 80)"></circle>
          <defs>
            <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stop-color="#6c5ce7"/>
              <stop offset="100%" stop-color="#00cec9"/>
            </linearGradient>
          </defs>
        </svg>
        <div class="progress-percentage">${result.probability}%</div>
      </div>

      <span class="risk-badge ${riskClass}">${result.risk_level} Risk</span>
      <p class="text-muted small mt-3">Estimated probability that this customer will churn.</p>
    `;

    // Animate the ring
    const circle = document.getElementById("progressCircle");
    const circumference = 440;
    const offset = circumference - (result.probability / 100) * circumference;
    setTimeout(() => {
      circle.style.transition = "stroke-dashoffset 1s ease-out";
      circle.style.strokeDashoffset = offset;
    }, 100);
  }
});
