// ===== Tab switching =====
const tabButtons = document.querySelectorAll(".tab-btn");
const panels = document.querySelectorAll(".panel");

tabButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    tabButtons.forEach((b) => {
      b.classList.remove("active");
      b.setAttribute("aria-selected", "false");
    });
    panels.forEach((p) => p.classList.remove("active"));

    btn.classList.add("active");
    btn.setAttribute("aria-selected", "true");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
  });
});

// ===== Helpers =====
function showLoading(container, message) {
  container.innerHTML = `<div class="loading-state">${message}</div>`;
}

function showError(container, message) {
  container.innerHTML = `<div class="error-state">${message}</div>`;
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "حدث خطأ غير متوقع.");
  }
  return data;
}

// ===== Search =====
const searchForm = document.getElementById("search-form");
const searchResults = document.getElementById("search-results");

searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = document.getElementById("search-query").value.trim();
  if (!query) return;

  showLoading(searchResults, "يجري البحث في الديوان...");

  try {
    const data = await postJSON("/api/search", { query });

    if (!data.results || data.results.length === 0) {
      searchResults.innerHTML = `<div class="empty-state">لم يتم العثور على قصائد مطابقة.</div>`;
      return;
    }

    searchResults.innerHTML = data.results.map((r) => `
      <div class="result-card">
        <div class="result-meta">
          <span>الشاعر: ${r.poet}</span>
          <span>القصيدة: ${r.title}</span>
          ${r.era ? `<span>العصر: ${r.era}</span>` : ""}
        </div>
        <div class="result-poem">${escapeHtml(r.poem_text)}</div>
      </div>
    `).join("");
  } catch (err) {
    showError(searchResults, err.message);
  }
});

// ===== Rewrite =====
const rewriteForm = document.getElementById("rewrite-form");
const rewriteResults = document.getElementById("rewrite-results");

rewriteForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fragment = document.getElementById("rewrite-fragment").value.trim();
  const modified_first_verse = document.getElementById("rewrite-verse").value.trim();
  if (!fragment || !modified_first_verse) return;

  showLoading(rewriteResults, "يجري البحث عن القصيدة الأصلية وإعادة الصياغة...");

  try {
    const data = await postJSON("/api/rewrite", { fragment, modified_first_verse });

    const statusClass = data.consistent ? "status-ok" : "status-warn";
    const statusText = data.consistent
      ? "القافية متسقة في كامل الأبيات."
      : `القافية لم تلتزم بحرف الروي بعد ${data.attempts} محاولات.`;

    rewriteResults.innerHTML = `
      <div class="result-card">
        <div class="result-meta">
          <span>القصيدة الأصلية: ${data.title}</span>
          <span>الشاعر: ${data.poet}</span>
          <span>العصر: ${data.era}</span>
        </div>
        <div class="result-poem">${escapeHtml(modified_first_verse)}\n${escapeHtml(data.generated_text)}</div>
        <div class="result-status ${statusClass}">${statusText}</div>
      </div>
    `;
  } catch (err) {
    showError(rewriteResults, err.message);
  }
});

// ===== Generate =====
const generateForm = document.getElementById("generate-form");
const generateResults = document.getElementById("generate-results");

generateForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const poet = document.getElementById("gen-poet").value.trim();
  const theme = document.getElementById("gen-theme").value.trim();
  const topic = document.getElementById("gen-topic").value.trim();
  if (!poet || !theme || !topic) return;

  showLoading(generateResults, "يجري تأليف القصيدة على منوال الشاعر...");

  try {
    const data = await postJSON("/api/generate", { poet, theme, topic });

    generateResults.innerHTML = `
      <div class="result-card">
        <div class="result-meta">
          <span>الشاعر: ${data.poet}</span>
          <span>الثيم: ${data.theme}</span>
          <span>الموضوع: ${data.topic}</span>
        </div>
        <div class="result-poem">${escapeHtml(data.poem_text)}</div>
      </div>
    `;
  } catch (err) {
    showError(generateResults, err.message);
  }
});

// ===== Utility: prevent HTML injection from generated/retrieved text =====
function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
