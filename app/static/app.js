const getElement = (id) => document.getElementById(id);

function showGuidance(articles) {
  const list = getElement("guidance");
  list.replaceChildren();

  if (articles.length === 0) {
    const item = document.createElement("li");
    item.textContent = "No matching policy article was found.";
    list.append(item);
    return;
  }

  for (const article of articles) {
    const item = document.createElement("li");
    const title = document.createElement("strong");
    const snippet = document.createElement("small");
    title.textContent = article.title;
    snippet.textContent = article.snippet;
    item.append(title, document.createElement("br"), snippet);
    list.append(item);
  }
}

function showResult(result) {
  getElement("empty").hidden = true;
  getElement("result").hidden = false;
  getElement("intent").textContent = result.intent_name;

  const priority = getElement("priority");
  priority.textContent = result.urgency === "high" ? "High priority" : "Standard priority";
  priority.className = `priority ${result.urgency}`;

  getElement("confidence").textContent = `${(result.confidence * 100).toFixed(1)}%`;
  getElement("summary").textContent = result.agent_summary;
  getElement("next-step").textContent = result.recommended_next_step;
  getElement("orders").textContent = result.order_ids.length
    ? result.order_ids.join(", ")
    : "No order reference found.";
  getElement("redacted").textContent = result.redacted_message;
  showGuidance(result.knowledge_matches);
}

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => {
    getElement("message").value = button.dataset.message;
  });
}

getElement("ticket-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  const submitButton = getElement("submit");
  const errorMessage = getElement("error");
  errorMessage.textContent = "";
  submitButton.disabled = true;
  submitButton.textContent = "Reviewing…";

  try {
    const response = await fetch("/v1/tickets/analyze", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: getElement("message").value }),
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Ticket review failed.");
    }
    showResult(result);
  } catch (error) {
    errorMessage.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Review ticket";
  }
});
