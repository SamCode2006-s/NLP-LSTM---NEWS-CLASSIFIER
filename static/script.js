const textArea = document.getElementById("newsText");
const count = document.getElementById("count");
const button = document.getElementById("predictBtn");
const errorBox = document.getElementById("errorBox");
const result = document.getElementById("result");

textArea.addEventListener("input", () => {
  count.textContent = `${textArea.value.length} / 12000`;
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.style.display = "block";
}

function hideError() {
  errorBox.textContent = "";
  errorBox.style.display = "none";
}

function setPrediction(index, item) {
  document.getElementById(`cat${index}`).textContent = item?.category || "—";
  document.getElementById(`pct${index}`).textContent =
    item ? `${item.confidence.toFixed(2)}%` : "—";
  document.getElementById(`fill${index}`).style.width =
    item ? `${Math.min(item.confidence, 100)}%` : "0%";
}

button.addEventListener("click", async () => {
  const text = textArea.value.trim();

  hideError();

  if (!text) {
    showError("Enter text.");
    textArea.focus();
    return;
  }

  button.disabled = true;
  button.querySelector("span").textContent = "READING";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text })
    });

    const raw = await response.text();

    let data;

    try {
        data = JSON.parse(raw);
    } catch (error) {
        console.error("Raw server response:", raw);
        throw new Error(
            `Server returned an invalid response (${response.status}).`
        );
    }

    if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
    }

    const predictions = data.predictions || [];
    const top = data.top_prediction;

    document.getElementById("topCategory").textContent =
      top?.category || "—";
    document.getElementById("topConfidence").textContent =
      top ? `${top.confidence.toFixed(2)}% confidence` : "—";

    [0, 1, 2].forEach((i) => setPrediction(i, predictions[i]));
    result.classList.add("show");
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "CLASSIFY";
  }
});

textArea.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    button.click();
  }
});
