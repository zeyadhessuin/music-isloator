"use strict";

const JOBS_KEY = "vocal_def_jobs";
const POLL_MS = 2000;
const jobCards = new Map();

const form = document.getElementById("job-form");
const submitBtn = document.getElementById("submit-btn");
const btnLabel = submitBtn.querySelector(".btn-label");
const btnSpinner = submitBtn.querySelector(".btn-spinner");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("files");
const fileList = document.getElementById("file-list");
const jobsEl = document.getElementById("jobs");
const template = document.getElementById("job-template");
const clearCacheBtn = document.getElementById("clear-cache");

clearCacheBtn.addEventListener("click", async () => {
  if (!confirm("Delete all cached downloads? They will be re-downloaded next time.")) return;
  clearCacheBtn.disabled = true;
  try {
    const res = await fetch("/api/cache", { method: "DELETE" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    alert(`Cache cleared: removed ${data.removed} file(s).`);
  } catch (err) {
    alert("Failed to clear cache:\n" + err.message);
  } finally {
    clearCacheBtn.disabled = false;
  }
});

function trackJob(id) {
  let ids = JSON.parse(localStorage.getItem(JOBS_KEY) || "[]");
  if (!ids.includes(id)) {
    ids.unshift(id);
    localStorage.setItem(JOBS_KEY, JSON.stringify(ids.slice(0, 30)));
  }
}

function fmtTime(ts) {
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function renderJob(job) {
  let card = jobCards.get(job.id);
  if (!card) {
    const frag = template.content.cloneNode(true);
    const el = frag.querySelector(".job-card");
    el.dataset.id = job.id;
    jobsEl.prepend(el);
    card = {
      el,
      id: el.querySelector(".job-id"),
      badge: el.querySelector(".badge"),
      message: el.querySelector(".job-message"),
      bar: el.querySelector(".bar"),
      progress: el.querySelector(".progress"),
      results: el.querySelector(".results"),
      logs: el.querySelector(".logs"),
      toggle: el.querySelector(".toggle-logs"),
    };
    card.toggle.addEventListener("click", () => {
      card.logs.hidden = !card.logs.hidden;
    });
    jobCards.set(job.id, card);
    const empty = jobsEl.querySelector(".empty");
    if (empty) empty.remove();
  }

  card.id.textContent = "#" + job.id;
  card.badge.textContent = job.state;
  card.badge.className = "badge " + job.state;
  card.message.textContent = job.message || "";

  if (job.state === "running" && job.stage === "separating") {
    card.progress.classList.add("indeterminate");
    card.bar.style.width = "";
  } else {
    card.progress.classList.remove("indeterminate");
    card.bar.style.width = (job.progress || 0) + "%";
  }

  if (job.state === "queued") {
    card.progress.classList.add("indeterminate");
  }

  card.results.innerHTML = "";
  if (job.results && job.results.length) {
    for (const r of job.results) {
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = r.url;
      a.textContent = "Download: " + r.name;
      a.setAttribute("download", "");
      li.appendChild(a);
      card.results.appendChild(li);
    }
  }

  if (job.logs && job.logs.length) {
    card.logs.textContent = job.logs.join("\n");
    card.logs.scrollTop = card.logs.scrollHeight;
  }
  if (job.error) {
    card.message.textContent = "Error: " + job.error;
    card.message.style.color = "#ff6b6b";
  }
}

async function refreshJobs() {
  const ids = JSON.parse(localStorage.getItem(JOBS_KEY) || "[]");
  if (!ids.length) return;
  try {
    const res = await fetch("/api/jobs");
    const jobs = await res.json();
    const active = new Set();
    for (const job of jobs) {
      if (ids.includes(job.id)) {
        renderJob(job);
        active.add(job.id);
      }
    }
    for (const id of ids) {
      if (!active.has(id) && !jobCards.has(id)) {
        // finished earlier and cleared from list; fetch individually
      }
    }
  } catch (err) {
    console.warn("Poll failed:", err);
  }
}

async function startJob(formData) {
  submitBtn.disabled = true;
  btnLabel.textContent = "Submitting...";
  btnSpinner.hidden = false;
  try {
    const res = await fetch("/api/jobs", { method: "POST", body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    trackJob(data.job_id);
    await refreshJobs();
  } catch (err) {
    alert("Failed to start job:\n" + err.message);
  } finally {
    submitBtn.disabled = false;
    btnLabel.textContent = "Separate Vocals + Def";
    btnSpinner.hidden = true;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  startJob(fd);
});

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length) {
    fileInput.files = e.dataTransfer.files;
    updateFileList();
  }
});
fileInput.addEventListener("change", updateFileList);

function updateFileList() {
  fileList.innerHTML = "";
  for (const f of fileInput.files) {
    const li = document.createElement("li");
    li.textContent = f.name;
    fileList.appendChild(li);
  }
}

refreshJobs();
setInterval(refreshJobs, POLL_MS);
