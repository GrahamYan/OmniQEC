"use strict";

const workflowContent = {
  query: {
    className: "detail-query",
    kicker: "Design specification",
    title: "Start from practical constraints",
    description:
      "The user describes the desired code family, total qubit budget, weight limit, connectivity, and optimization objective.",
    points: [
      "Complete physical-qubit budget, not data qubits alone",
      "Hardware and syndrome-extraction constraints",
      "Target code family and circuit-level objective",
    ],
  },
  orchestrator: {
    className: "detail-orchestrator",
    kicker: "Planning · routing · memory",
    title: "Turn intent into a discovery campaign",
    description:
      "An LLM-based orchestrator translates the request into a structured task, routes candidates through scientific tools, and maintains the evidence accumulated across iterations.",
    points: [
      "Structured objectives and stopping conditions",
      "Unified records for programs, circuits, and decoding results",
      "Self-evolving search context built from prior evidence",
    ],
  },
  fast: {
    className: "detail-fast",
    kicker: "Seconds per candidate",
    title: "Explore broadly with inexpensive screening",
    description:
      "A generative search agent proposes executable construction programs. Code-characterization tools verify validity and rank the resulting codes with inexpensive algebraic quantities.",
    points: [
      "Lifted product, bivariate bicycle, trivariate tricycle, and generalized lifted product",
      "CSS commutation, logical-qubit count, distance, and sparsity checks",
      "Large candidate populations screened before costly simulation",
    ],
  },
  slow: {
    className: "detail-slow",
    kicker: "Minutes to hours per candidate",
    title: "Evaluate the complete implementation",
    description:
      "Selected codes are compiled into syndrome-extraction circuits and evaluated under circuit-level noise with configurable decoders.",
    points: [
      "Ancilla allocation, scheduling, and Stim circuit construction",
      "Detector error model and noisy syndrome sampling",
      "BP+OSD, BP+LSD, and Relay-BP logical-error estimation",
    ],
  },
  feedback: {
    className: "detail-feedback",
    kicker: "Evidence for the next generation",
    title: "Let physical performance reshape the search",
    description:
      "Circuit-level rankings, synthesis records, and decoding evidence return to the orchestrator, guiding new proposals toward practically competitive regions.",
    points: [
      "Promote structures supported by physical evidence",
      "Avoid proxy-optimal but circuit-level suboptimal candidates",
      "Finish with ranked codes, compiled circuits, and decoding performance",
    ],
  },
};

const workflowTabs = document.querySelectorAll("[data-workflow]");
const workflowDetail = document.querySelector("#workflow-detail");
const workflowKicker = document.querySelector("#workflow-kicker");
const workflowTitle = document.querySelector("#workflow-title");
const workflowDescription = document.querySelector("#workflow-description");
const workflowPoints = document.querySelector("#workflow-points");

function setWorkflowStage(stage) {
  const content = workflowContent[stage];
  if (
    !content ||
    !workflowDetail ||
    !workflowKicker ||
    !workflowTitle ||
    !workflowDescription ||
    !workflowPoints
  ) {
    return;
  }

  workflowTabs.forEach((tab) => {
    const isActive = tab.dataset.workflow === stage;
    tab.classList.toggle("active", isActive);
    tab.setAttribute("aria-selected", String(isActive));
  });

  workflowDetail.className = `workflow-detail ${content.className}`;
  workflowKicker.textContent = content.kicker;
  workflowTitle.textContent = content.title;
  workflowDescription.textContent = content.description;
  workflowPoints.replaceChildren(
    ...content.points.map((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      return item;
    }),
  );
}

workflowTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    setWorkflowStage(tab.dataset.workflow);
  });

  tab.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowRight", "ArrowUp", "ArrowLeft"].includes(event.key)) {
      return;
    }

    event.preventDefault();
    const tabs = Array.from(workflowTabs);
    const currentIndex = tabs.indexOf(tab);
    const step = ["ArrowDown", "ArrowRight"].includes(event.key) ? 1 : -1;
    const nextIndex = (currentIndex + step + tabs.length) % tabs.length;
    tabs[nextIndex].focus();
    tabs[nextIndex].click();
  });
});

const revealItems = document.querySelectorAll(".reveal");
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

if (prefersReducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("visible"));
} else {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.08, rootMargin: "0px 0px -30px" },
  );

  revealItems.forEach((item) => revealObserver.observe(item));
}

const assistantWidget = document.querySelector(".assistant-widget");
const assistantPanel = document.querySelector("#assistant-panel");
const assistantToggle = document.querySelector("#assistant-toggle");
const assistantClose = document.querySelector("#assistant-close");
const assistantClear = document.querySelector("#assistant-clear");
const assistantHeroButton = document.querySelector("#open-assistant-hero");
const assistantResourcesButton = document.querySelector("#open-assistant-resources");
const assistantForm = document.querySelector("#assistant-form");
const assistantInput = document.querySelector("#assistant-input");
const assistantMessages = document.querySelector("#assistant-messages");
const assistantStatus = document.querySelector("#assistant-status");
const assistantSend = document.querySelector("#assistant-send");
const assistantCharCount = document.querySelector("#assistant-char-count");
const assistantHistory = [];
const maxAssistantMessageChars = 900;
const localHosts = new Set(["localhost", "127.0.0.1", "::1"]);
const localAssistantEndpoint = "http://127.0.0.1:8787/chat";
const productionAssistantEndpoint = assistantWidget?.dataset.productionEndpoint || "";
const assistantEndpoint = localHosts.has(window.location.hostname)
  ? localAssistantEndpoint
  : productionAssistantEndpoint;
let assistantReady = false;
let assistantBusy = false;

function setAssistantOpen(isOpen) {
  if (!assistantWidget || !assistantPanel || !assistantToggle) return;
  assistantWidget.classList.toggle("open", isOpen);
  assistantToggle.setAttribute("aria-expanded", String(isOpen));
  assistantPanel.setAttribute("aria-hidden", String(!isOpen));

  if (isOpen) {
    window.setTimeout(() => assistantInput?.focus(), 80);
  } else {
    assistantToggle.focus();
  }
}

function setAssistantStatus(message, state = "") {
  if (!assistantStatus) return;
  assistantStatus.textContent = message;
  assistantStatus.className = ["assistant-status", state].filter(Boolean).join(" ");
}

function setAssistantEnabled(isEnabled) {
  assistantReady = isEnabled;
  if (assistantInput) assistantInput.disabled = !isEnabled;
  if (assistantSend) assistantSend.disabled = !isEnabled;
}

function scrollAssistantToBottom() {
  if (!assistantMessages) return;
  assistantMessages.scrollTop = assistantMessages.scrollHeight;
}

function appendAssistantMessage(text, role, sources = []) {
  if (!assistantMessages) return null;
  const article = document.createElement("article");
  article.className = `assistant-message assistant-message-${
    role === "user" ? "user" : "bot"
  }`;

  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);

  if (sources.length) {
    const sourceList = document.createElement("div");
    sourceList.className = "assistant-sources";
    sourceList.setAttribute("aria-label", "Sources used");
    sources.forEach((source) => {
      const chip = document.createElement("span");
      chip.className = "assistant-source";
      chip.textContent = source;
      sourceList.appendChild(chip);
    });
    article.appendChild(sourceList);
  }

  assistantMessages.appendChild(article);
  scrollAssistantToBottom();
  return article;
}

function resetAssistantConversation() {
  assistantHistory.length = 0;
  if (!assistantMessages) return;
  assistantMessages.replaceChildren();
  const welcome = appendAssistantMessage(
    "Conversation cleared. Ask about the slow-fast workflow, code families, experiments, or results in the current manuscript.",
    "assistant",
  );
  welcome?.classList.add("assistant-message-bot");
}

function resolveHealthEndpoint() {
  if (!assistantEndpoint) return "";
  return assistantEndpoint.replace(/\/chat\/?$/, "/health");
}

async function checkAssistantHealth() {
  const healthEndpoint = resolveHealthEndpoint();
  if (!healthEndpoint) {
    setAssistantEnabled(false);
    setAssistantStatus(
      "The assistant backend has not been connected for this environment.",
      "error",
    );
    return;
  }

  try {
    const response = await fetch(healthEndpoint, {
      headers: { Accept: "application/json" },
    });
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    const modeLabel = data.mode === "mock" ? "Local grounded mock" : "Workers AI";
    setAssistantEnabled(true);
    setAssistantStatus(`${modeLabel} ready · Answers use the current manuscript only.`, "ready");
  } catch {
    setAssistantEnabled(false);
    setAssistantStatus(
      "Local assistant is offline. Start the mock service on port 8787 and retry.",
      "error",
    );
  }
}

async function submitAssistantQuestion(question) {
  const message = String(question || "").trim();
  if (!message || !assistantEndpoint || !assistantReady || assistantBusy) return;
  if (message.length > maxAssistantMessageChars) {
    setAssistantStatus(
      `Questions are limited to ${maxAssistantMessageChars} characters.`,
      "error",
    );
    return;
  }

  assistantBusy = true;
  setAssistantEnabled(false);
  appendAssistantMessage(message, "user");

  const pending = appendAssistantMessage("Reading the relevant manuscript sections…", "assistant");
  pending?.classList.add("assistant-message-pending");
  setAssistantStatus("Retrieving grounded context…", "busy");

  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), 30000);

  try {
    const response = await fetch(assistantEndpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: assistantHistory.slice(-6),
      }),
      signal: controller.signal,
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Assistant request failed (${response.status})`);
    }

    const answer = data.answer || "The assistant did not return an answer.";
    const sources = Array.isArray(data.sources) ? data.sources : [];
    pending?.remove();
    appendAssistantMessage(answer, "assistant", sources);
    assistantHistory.push({ role: "user", content: message });
    assistantHistory.push({ role: "assistant", content: answer });
    setAssistantStatus(
      `${data.model === "local-grounded-mock" ? "Local grounded mock" : "Grounded model"} · ${
        sources.length
          ? `${sources.length} source section${sources.length === 1 ? "" : "s"}`
          : "No matching source"
      }`,
      "ready",
    );
  } catch (error) {
    pending?.remove();
    const timedOut = error instanceof DOMException && error.name === "AbortError";
    appendAssistantMessage(
      timedOut
        ? "The request timed out. Please try a shorter question."
        : "I could not reach the local assistant. Check that the mock service is running.",
      "assistant",
    );
    setAssistantStatus(timedOut ? "Request timed out." : "Assistant connection failed.", "error");
  } finally {
    window.clearTimeout(timeoutId);
    assistantBusy = false;
    setAssistantEnabled(true);
    assistantInput?.focus();
  }
}

assistantToggle?.addEventListener("click", () => {
  setAssistantOpen(!assistantWidget.classList.contains("open"));
});

assistantClose?.addEventListener("click", () => setAssistantOpen(false));
assistantHeroButton?.addEventListener("click", () => setAssistantOpen(true));
assistantResourcesButton?.addEventListener("click", () => setAssistantOpen(true));
assistantClear?.addEventListener("click", resetAssistantConversation);

assistantForm?.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = assistantInput?.value.trim() || "";
  if (!question) return;
  if (question.length > maxAssistantMessageChars) {
    setAssistantStatus(
      `Questions are limited to ${maxAssistantMessageChars} characters.`,
      "error",
    );
    assistantInput?.focus();
    return;
  }
  assistantInput.value = "";
  if (assistantCharCount) {
    assistantCharCount.textContent = `0 / ${maxAssistantMessageChars}`;
  }
  await submitAssistantQuestion(question);
});

assistantInput?.addEventListener("input", () => {
  if (assistantInput.value.length > maxAssistantMessageChars) {
    assistantInput.value = assistantInput.value.slice(0, maxAssistantMessageChars);
    setAssistantStatus(
      `Question limited to ${maxAssistantMessageChars} characters.`,
      "error",
    );
  }
  if (assistantCharCount) {
    assistantCharCount.textContent = `${assistantInput.value.length} / ${maxAssistantMessageChars}`;
  }
});

assistantInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    assistantForm?.requestSubmit();
  }
});

document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    const question = button.dataset.question || "";
    setAssistantOpen(true);
    if (assistantInput) {
      assistantInput.value = question;
      assistantInput.dispatchEvent(new Event("input"));
      assistantInput.focus();
    }
  });
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && assistantWidget?.classList.contains("open")) {
    setAssistantOpen(false);
  }
});

setAssistantEnabled(false);
void checkAssistantHealth();
