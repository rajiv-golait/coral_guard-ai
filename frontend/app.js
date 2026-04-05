(function () {
  "use strict";

  const SAMPLE_JSON = JSON.stringify(
    {
      latitude_degrees: 23.163,
      longitude_degrees: -82.526,
      depth_m: 10,
      turbidity: 0.0287,
      cyclone_frequency: 49.9,
      clim_sst: 301.61,
      ssta: -0.46,
      tsa: -0.8,
      percent_cover: 0,
      date_year: 2005,
    },
    null,
    2
  );

  const $ = (sel) => document.querySelector(sel);

  /** @type {File|null} */
  let droppedFile = null;

  const els = {
    apiBase: $("#api-base"),
    image: $("#image"),
    btnPick: $("#btn-pick-image"),
    dropZone: $("#drop-zone"),
    fileName: $("#file-name"),
    featuresForm: $("#features-form"),
    featuresJson: $("#features-json"),
    btnRun: $("#btn-run"),
    btnSample: $("#btn-fill-sample"),
    btnSpinner: $("#btn-spinner"),
    protocolWarning: $("#protocol-warning"),
    outEmpty: $("#out-empty"),
    outError: $("#out-error"),
    outResults: $("#out-results"),
    bodyCnn: $("#body-cnn"),
    bodyDbscan: $("#body-dbscan"),
    bodyFusion: $("#body-fusion"),
    bodyLlm: $("#body-llm"),
    bodyMeta: $("#body-meta"),
  };

  function apiRoot() {
    const v = (els.apiBase.value || "").trim().replace(/\/+$/, "");
    if (v) return v;
    return "";
  }

  function buildFeaturesFromForm() {
    const fd = new FormData(els.featuresForm);
    const o = {};
    for (const [k, v] of fd.entries()) {
      if (k === "date_year") o[k] = parseInt(String(v), 10);
      else if (k === "percent_cover") o[k] = v === "" ? 0 : parseFloat(String(v));
      else o[k] = parseFloat(String(v));
    }
    return o;
  }

  function getFeaturesJsonString() {
    const raw = (els.featuresJson.value || "").trim();
    if (raw) return raw;
    return JSON.stringify(buildFeaturesFromForm());
  }

  function formatDetail(detail) {
    if (detail == null) return "";
    if (typeof detail === "string") return detail;
    try {
      return JSON.stringify(detail, null, 2);
    } catch {
      return String(detail);
    }
  }

  function renderCnn(cnn) {
    if (!cnn) {
      els.bodyCnn.innerHTML = "<p class='muted'>No CNN output (model missing or error).</p>";
      return;
    }
    const probs = cnn.probabilities || {};
    const rows = Object.entries(probs)
      .map(([cls, p]) => {
        const pct = Math.round(p * 1000) / 10;
        return `<div class="prob-row"><span>${cls}</span><div class="prob-bar" title="${pct}%"><span style="width:${pct}%"></span></div><span>${pct}%</span></div>`;
      })
      .join("");
    els.bodyCnn.innerHTML = `
      <p><span class="badge">${cnn.predicted_class}</span> · confidence <strong>${(cnn.confidence * 100).toFixed(1)}%</strong></p>
      ${rows}
    `;
  }

  function renderDbscan(db) {
    if (!db) {
      els.bodyDbscan.innerHTML = "<p class='muted'>No cluster output (tabular bundle or error).</p>";
      return;
    }
    els.bodyDbscan.innerHTML = `
      <p><span class="badge">${db.cluster_label}</span> · id <strong>${db.cluster_id}</strong></p>
      ${db.detail ? `<p class="muted" style="margin-top:0.5rem;font-size:0.82rem">${escapeHtml(db.detail)}</p>` : ""}
    `;
  }

  function renderFusion(fusion) {
    if (!fusion) {
      els.bodyFusion.innerHTML = "<p class='muted'>No fusion output (model or tabular error).</p>";
      return;
    }
    const v = fusion.predicted_percent_bleaching;
    els.bodyFusion.innerHTML = `
      <p class="badge bleach">Predicted bleaching</p>
      <p style="font-size:1.75rem;font-weight:700;margin:0.35rem 0;">${typeof v === "number" ? v.toFixed(2) : v}%</p>
    `;
  }

  function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function renderLlm(text) {
    if (!text) {
      els.bodyLlm.innerHTML = "<p class='muted'>No report (set GROQ_API_KEY on the server).</p>";
      return;
    }
    els.bodyLlm.textContent = text;
  }

  els.btnPick.addEventListener("click", () => els.image.click());

  els.image.addEventListener("change", () => {
    droppedFile = null;
    const f = els.image.files && els.image.files[0];
    els.fileName.textContent = f ? f.name : "No file selected";
  });

  ["dragenter", "dragover"].forEach((ev) => {
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((ev) => {
    els.dropZone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.dropZone.classList.remove("dragover");
    });
  });

  els.dropZone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      droppedFile = file;
      els.fileName.textContent = file.name + " (dropped)";
    }
  });

  els.btnSample.addEventListener("click", () => {
    els.featuresJson.value = SAMPLE_JSON;
    document.querySelector("#expert-details").open = true;
  });

  els.btnRun.addEventListener("click", async () => {
    const file = droppedFile || (els.image.files && els.image.files[0]);
    if (!file) {
      alert("Choose a reef image first.");
      return;
    }

    let featuresStr;
    try {
      const raw = getFeaturesJsonString();
      const parsed = JSON.parse(raw);
      featuresStr = JSON.stringify(parsed);
    } catch (e) {
      alert("Invalid features JSON: " + (e && e.message ? e.message : e));
      return;
    }

    const root = apiRoot();
    const url = (root || "") + "/predict";

    els.outError.classList.add("hidden");
    els.outError.textContent = "";
    els.outResults.classList.add("hidden");
    els.outEmpty.classList.remove("hidden");
    els.btnRun.disabled = true;
    els.btnSpinner.classList.remove("hidden");

    const fd = new FormData();
    fd.append("image", file);
    fd.append("features", featuresStr);

    try {
      const res = await fetch(url, { method: "POST", body: fd });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        const msg = formatDetail(data.detail) || res.statusText;
        throw new Error(msg);
      }

      els.outEmpty.classList.add("hidden");
      els.outResults.classList.remove("hidden");

      renderCnn(data.cnn);
      renderDbscan(data.dbscan);
      renderFusion(data.fusion);
      renderLlm(data.llm_report);

      const meta = {
        status: data.status,
        meta: data.meta || {},
        errors: data.errors || [],
      };
      els.bodyMeta.textContent = JSON.stringify(meta, null, 2);
    } catch (err) {
      els.outEmpty.classList.add("hidden");
      els.outResults.classList.add("hidden");
      els.outError.classList.remove("hidden");
      els.outError.textContent = err.message || String(err);
    } finally {
      els.btnRun.disabled = false;
      els.btnSpinner.classList.add("hidden");
    }
  });

  if (window.location.protocol === "file:") {
    els.protocolWarning.classList.remove("hidden");
  }

  if (!els.apiBase.value && window.location.origin && window.location.origin !== "null") {
    els.apiBase.placeholder = window.location.origin;
  }
})();
