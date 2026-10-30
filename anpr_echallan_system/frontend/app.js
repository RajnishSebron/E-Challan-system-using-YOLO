/**
 * app.js - E-Challan & ANPR Command Center Client Logic
 * Handles WebSocket alerts, Fetch API calls, Interactive ROI Canvas drawing,
 * Signal Controls, and Modal inspection.
 */

document.addEventListener("DOMContentLoaded", () => {
  // -------------------------------------------------------------------
  // 1. State & Element References
  // -------------------------------------------------------------------
  let soundEnabled = true;
  let audioCtx = null;
  let ws = null;
  let isDrawingRoi = false;
  let roiPoints = []; // [[x1, y1], [x2, y2]]
  let currentSignal = "RED";

  // Header & Controls
  const clockEl = document.getElementById("digitalClock");
  const soundToggleBtn = document.getElementById("soundToggleBtn");
  const soundIcon = document.getElementById("soundIcon");
  const toggleSignalBtn = document.getElementById("toggleSignalBtn");
  const signalPill = document.getElementById("signalPill");
  const signalText = document.getElementById("signalText");
  const testViolationBtn = document.getElementById("testViolationBtn");

  // KPIs
  const kpiVehicles = document.getElementById("kpiVehicles");
  const kpiViolations = document.getElementById("kpiViolations");
  const kpiFines = document.getElementById("kpiFines");
  const kpiAccuracy = document.getElementById("kpiAccuracy");

  // Surveillance & Canvas
  const feedWrapper = document.getElementById("feedWrapper");
  const videoFeed = document.getElementById("videoFeed");
  const roiCanvas = document.getElementById("roiCanvas");
  const ctx = roiCanvas.getContext("2d");
  const drawRoiBtn = document.getElementById("drawRoiBtn");
  const clearRoiBtn = document.getElementById("clearRoiBtn");
  const saveRoiBtn = document.getElementById("saveRoiBtn");

  // Alert Container
  const alertsContainer = document.getElementById("alertsContainer");
  const noAlertsMsg = document.getElementById("noAlertsMsg");
  const wsStatus = document.getElementById("wsStatus");
  const wsText = document.getElementById("wsText");

  // Table & Filters
  const challanTableBody = document.getElementById("challanTableBody");
  const plateSearchInput = document.getElementById("plateSearchInput");
  const violationFilter = document.getElementById("violationFilter");
  const statusFilter = document.getElementById("statusFilter");
  const refreshTableBtn = document.getElementById("refreshTableBtn");

  // Modal
  const challanModal = document.getElementById("challanModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const modalDismissBtn = document.getElementById("modalDismissBtn");
  const modalPrintBtn = document.getElementById("modalPrintBtn");
  const modalPayBtn = document.getElementById("modalPayBtn");

  let currentModalChallanId = null;

  // -------------------------------------------------------------------
  // 2. Real-time Digital Clock
  // -------------------------------------------------------------------
  function updateClock() {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString("en-IN", { hour12: false }) + " IST";
  }
  setInterval(updateClock, 1000);
  updateClock();

  // -------------------------------------------------------------------
  // 3. Synthesized Web Audio Alert Chime
  // -------------------------------------------------------------------
  function playAlertChime() {
    if (!soundEnabled) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === "suspended") {
        audioCtx.resume();
      }

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = "sine";
      // Dual-tone futuristic police/security ping: 880Hz -> 1760Hz
      osc.frequency.setValueAtTime(880, audioCtx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1760, audioCtx.currentTime + 0.15);

      gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.35);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + 0.35);
    } catch (e) {
      console.warn("Audio chime failed:", e);
    }
  }

  soundToggleBtn.addEventListener("click", () => {
    soundEnabled = !soundEnabled;
    soundIcon.textContent = soundEnabled ? "🔊" : "🔇";
    soundToggleBtn.title = soundEnabled ? "Mute Sound" : "Unmute Sound";
  });

  // -------------------------------------------------------------------
  // 4. WebSocket Client for Real-Time Alerts
  // -------------------------------------------------------------------
  function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws/alerts`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      wsText.textContent = "CONNECTED";
      wsStatus.style.color = "var(--accent-emerald)";
      console.log("WebSocket connected to alert stream.");
    };

    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload.event === "NEW_VIOLATION" && payload.data) {
          handleIncomingViolation(payload.data);
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
      }
    };

    ws.onclose = () => {
      wsText.textContent = "RECONNECTING...";
      wsStatus.style.color = "var(--accent-amber)";
      setTimeout(initWebSocket, 3000);
    };

    ws.onerror = (err) => {
      console.error("WebSocket error:", err);
      ws.close();
    };
  }

  function handleIncomingViolation(violation) {
    playAlertChime();

    // Remove placeholder if present
    if (noAlertsMsg) {
      noAlertsMsg.style.display = "none";
    }

    // Create Alert Card
    const card = document.createElement("div");
    card.className = "alert-card";
    
    // Violation Badge Color
    let badgeClass = "badge-red-light";
    if (violation.violation_type.includes("HELMET")) badgeClass = "badge-helmet";
    if (violation.violation_type.includes("TRIPLE")) badgeClass = "badge-triple";

    card.innerHTML = `
      <div class="alert-card-top">
        <span class="alert-badge ${badgeClass}">${violation.violation_type.replace(/_/g, " ")}</span>
        <span class="alert-time">${violation.timestamp.split(" ")[1] || ""}</span>
      </div>
      <div class="alert-plate">${violation.license_plate}</div>
      <div class="alert-card-footer">
        <span>${violation.vehicle_type}</span>
        <span class="fine-amount">₹${violation.fine_amount}</span>
      </div>
    `;

    card.addEventListener("click", () => {
      openChallanModal(violation);
    });

    alertsContainer.prepend(card);

    // Keep max 15 cards
    while (alertsContainer.children.length > 15) {
      alertsContainer.removeChild(alertsContainer.lastChild);
    }

    // Refresh KPIs & Table
    fetchStats();
    fetchChallans();
  }

  // -------------------------------------------------------------------
  // 5. Canvas ROI Interactive Drawing
  // -------------------------------------------------------------------
  function resizeCanvas() {
    roiCanvas.width = feedWrapper.clientWidth;
    roiCanvas.height = feedWrapper.clientHeight;
    drawCanvasOverlay();
  }
  window.addEventListener("resize", resizeCanvas);
  setTimeout(resizeCanvas, 500);

  function drawCanvasOverlay() {
    ctx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);

    if (roiPoints.length > 0) {
      ctx.beginPath();
      ctx.strokeStyle = currentSignal === "RED" ? "#ef4444" : "#10b981";
      ctx.lineWidth = 4;
      ctx.setLineDash([8, 6]);

      ctx.moveTo(roiPoints[0][0], roiPoints[0][1]);
      for (let i = 1; i < roiPoints.length; i++) {
        ctx.lineTo(roiPoints[i][0], roiPoints[i][1]);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Draw point handles
      for (const pt of roiPoints) {
        ctx.beginPath();
        ctx.arc(pt[0], pt[1], 6, 0, Math.PI * 2);
        ctx.fillStyle = "#06b6d4";
        ctx.fill();
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  drawRoiBtn.addEventListener("click", () => {
    isDrawingRoi = true;
    roiPoints = [];
    drawCanvasOverlay();
    drawRoiBtn.textContent = "Click 2 Points on Road";
    drawRoiBtn.classList.add("btn-primary");
  });

  clearRoiBtn.addEventListener("click", () => {
    roiPoints = [];
    isDrawingRoi = false;
    drawRoiBtn.textContent = "Draw Stop Line";
    drawRoiBtn.classList.remove("btn-primary");
    drawCanvasOverlay();
  });

  roiCanvas.addEventListener("click", (e) => {
    if (!isDrawingRoi) return;

    const rect = roiCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    roiPoints.push([Math.round(x), Math.round(y)]);
    drawCanvasOverlay();

    if (roiPoints.length >= 2) {
      isDrawingRoi = false;
      drawRoiBtn.textContent = "Draw Stop Line";
      drawRoiBtn.classList.remove("btn-primary");
    }
  });

  saveRoiBtn.addEventListener("click", async () => {
    if (roiPoints.length < 2) {
      alert("Please mark at least 2 points to define the stop line.");
      return;
    }

    // Scale canvas coordinates to video resolution (1280x720)
    const scaleX = 1280 / roiCanvas.width;
    const scaleY = 720 / roiCanvas.height;

    const scaledLine = roiPoints.map(p => [Math.round(p[0] * scaleX), Math.round(p[1] * scaleY)]);

    try {
      const res = await fetch("/api/roi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stop_line: scaledLine })
      });
      const data = await res.json();
      if (data.success) {
        saveRoiBtn.textContent = "Saved ✓";
        setTimeout(() => { saveRoiBtn.textContent = "Save ROI"; }, 1500);
      }
    } catch (err) {
      console.error("Failed to save ROI:", err);
    }
  });

  // -------------------------------------------------------------------
  // 6. Signal Switcher (RED / GREEN)
  // -------------------------------------------------------------------
  toggleSignalBtn.addEventListener("click", async () => {
    const nextSignal = currentSignal === "RED" ? "GREEN" : "RED";
    try {
      const res = await fetch("/api/signal", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ signal: nextSignal })
      });
      const data = await res.json();
      if (data.success) {
        currentSignal = nextSignal;
        updateSignalUI(currentSignal);
        drawCanvasOverlay();
      }
    } catch (err) {
      console.error("Signal toggle error:", err);
    }
  });

  function updateSignalUI(signal) {
    currentSignal = signal;
    signalText.textContent = `SIGNAL: ${signal}`;
    if (signal === "RED") {
      signalPill.className = "signal-badge signal-red";
    } else {
      signalPill.className = "signal-badge signal-green";
    }
  }

  // -------------------------------------------------------------------
  // 7. Developer Trigger Test Violation
  // -------------------------------------------------------------------
  testViolationBtn.addEventListener("click", async () => {
    const types = ["NO_HELMET", "RED_LIGHT_VIOLATION", "TRIPLE_RIDING"];
    const chosen = types[Math.floor(Math.random() * types.length)];
    try {
      testViolationBtn.disabled = true;
      testViolationBtn.textContent = "Sending...";
      await fetch(`/api/trigger-test-violation?violation_type=${chosen}`, { method: "POST" });
    } catch (err) {
      console.error("Error triggering test violation:", err);
    } finally {
      testViolationBtn.disabled = false;
      testViolationBtn.textContent = "+ Test Violation";
    }
  });

  // -------------------------------------------------------------------
  // 8. Fetch Statistics & Dashboard KPIs
  // -------------------------------------------------------------------
  async function fetchStats() {
    try {
      const res = await fetch("/api/stats");
      const data = await res.json();

      kpiVehicles.textContent = data.total_vehicles.toLocaleString();
      kpiViolations.textContent = data.total_violations.toLocaleString();
      kpiFines.textContent = `₹${data.total_fines.toLocaleString()}`;
      kpiAccuracy.textContent = `${data.ocr_accuracy}%`;

      if (data.traffic_signal && data.traffic_signal !== currentSignal) {
        updateSignalUI(data.traffic_signal);
      }
    } catch (err) {
      console.warn("Could not fetch stats:", err);
    }
  }
  setInterval(fetchStats, 4000);
  fetchStats();

  // -------------------------------------------------------------------
  // 9. Fetch Challans & Table Rendering
  // -------------------------------------------------------------------
  async function fetchChallans() {
    const plate = plateSearchInput.value.trim();
    const vType = violationFilter.value;
    const status = statusFilter.value;

    let url = `/api/violations?limit=50&offset=0`;
    if (plate) url += `&plate=${encodeURIComponent(plate)}`;
    if (vType !== "ALL") url += `&violation_type=${encodeURIComponent(vType)}`;
    if (status !== "ALL") url += `&status=${encodeURIComponent(status)}`;

    try {
      const res = await fetch(url);
      const data = await res.json();
      renderChallanTable(data.violations || []);
    } catch (err) {
      console.error("Failed to fetch challan records:", err);
    }
  }

  function renderChallanTable(challans) {
    challanTableBody.innerHTML = "";

    if (challans.length === 0) {
      challanTableBody.innerHTML = `
        <tr>
          <td colspan="9" style="text-align: center; color: var(--text-dim); padding: 2rem;">
            No violation challans found matching current filters.
          </td>
        </tr>
      `;
      return;
    }

    for (const c of challans) {
      const tr = document.createElement("tr");

      const isPaid = c.status === "PAID";
      const statusClass = isPaid ? "status-paid" : "status-pending";

      tr.innerHTML = `
        <td class="font-mono">${c.challan_no}</td>
        <td class="font-mono">${c.timestamp}</td>
        <td class="font-mono"><strong>${c.license_plate}</strong></td>
        <td>${c.vehicle_type}</td>
        <td><span class="alert-badge badge-helmet">${c.violation_type.replace(/_/g, " ")}</span></td>
        <td class="fine-amount">₹${c.fine_amount}</td>
        <td>
          <img src="${c.plate_crop_path || c.evidence_frame_path}" class="thumb-preview" alt="Snap" onerror="this.src='/static/sample_plate.jpg'">
        </td>
        <td><span class="status-badge ${statusClass}">${c.status}</span></td>
        <td>
          <button class="btn btn-secondary btn-sm inspect-btn" data-id="${c.id}">View</button>
        </td>
      `;

      // Attach row event listener
      tr.querySelector(".inspect-btn").addEventListener("click", () => {
        openChallanModal(c);
      });

      challanTableBody.appendChild(tr);
    }
  }

  // Filter Event Listeners
  plateSearchInput.addEventListener("input", debounce(fetchChallans, 350));
  violationFilter.addEventListener("change", fetchChallans);
  statusFilter.addEventListener("change", fetchChallans);
  refreshTableBtn.addEventListener("click", fetchChallans);
  fetchChallans();

  function debounce(fn, ms) {
    let timer = null;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }

  // -------------------------------------------------------------------
  // 10. Official E-Challan Inspection Modal & Print
  // -------------------------------------------------------------------
  function openChallanModal(c) {
    currentModalChallanId = c.id;

    document.getElementById("modalChallanNo").textContent = c.challan_no;
    document.getElementById("modalChallanDate").textContent = c.timestamp;
    document.getElementById("modalEvidenceFrame").src = c.evidence_frame_path || "/static/sample_evidence.jpg";
    document.getElementById("modalPlateCrop").src = c.plate_crop_path || "/static/sample_plate.jpg";
    document.getElementById("modalPlateText").textContent = c.license_plate;
    document.getElementById("modalRegNo").textContent = c.license_plate;
    document.getElementById("modalVehicleClass").textContent = c.vehicle_type;
    document.getElementById("modalViolationType").textContent = c.violation_type.replace(/_/g, " ");
    document.getElementById("modalFine").textContent = `₹${c.fine_amount}`;
    document.getElementById("modalLocation").textContent = c.location || "NH-48 Junction, Sector 21";
    document.getElementById("modalStatus").textContent = c.status;
    document.getElementById("modalConfidence").textContent = `${Math.round((c.confidence || 0.95) * 100)}%`;

    if (c.status === "PAID") {
      modalPayBtn.style.display = "none";
    } else {
      modalPayBtn.style.display = "inline-flex";
    }

    challanModal.classList.add("active");
  }

  function closeModal() {
    challanModal.classList.remove("active");
    currentModalChallanId = null;
  }

  closeModalBtn.addEventListener("click", closeModal);
  modalDismissBtn.addEventListener("click", closeModal);
  challanModal.addEventListener("click", (e) => {
    if (e.target === challanModal) closeModal();
  });

  modalPrintBtn.addEventListener("click", () => {
    window.print();
  });

  modalPayBtn.addEventListener("click", async () => {
    if (!currentModalChallanId) return;

    try {
      const res = await fetch(`/api/violations/${currentModalChallanId}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: "PAID" })
      });
      const data = await res.json();
      if (data.success) {
        document.getElementById("modalStatus").textContent = "PAID";
        modalPayBtn.style.display = "none";
        fetchStats();
        fetchChallans();
      }
    } catch (err) {
      console.error("Error updating challan status:", err);
    }
  });

  // Initialize WebSockets
  initWebSocket();
});

