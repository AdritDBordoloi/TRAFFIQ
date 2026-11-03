/**
 * TRAFFIQ Exhibition Web Dashboard JavaScript
 * Polling, real-time telemetry, animations, and interactive demo triggers.
 * 100% Vanilla JS, zero dependencies, offline capable.
 */

(function () {
  'use strict';

  // State trackers
  let lastEventId = 0;
  let isConnected = true;
  let violationFlashTimer = null;
  let smsToastTimer = null;
  const recentChallans = [];

  // DOM Elements
  const el = {
    clock: document.getElementById('clockDisplay'),
    fps: document.getElementById('fpsDisplay'),
    videoFeed: document.getElementById('videoFeed'),
    emergencyBanner: document.getElementById('emergencyBanner'),
    screenFlash: document.getElementById('screenBorderFlash'),
    reconnectingBadge: document.getElementById('reconnectingBadge'),
    
    // Signal
    signalLight: document.getElementById('signalLight'),
    signalIcon: document.getElementById('signalIcon'),
    signalStateText: document.getElementById('signalStateText'),
    signalCountdown: document.getElementById('signalCountdown'),
    signalLockBadge: document.getElementById('signalLockBadge'),

    // Counters
    counterVehicles: document.getElementById('counterVehicles'),
    counterViolations: document.getElementById('counterViolations'),
    counterEmergencies: document.getElementById('counterEmergencies'),
    counterFines: document.getElementById('counterFines'),

    // Challan
    challanBody: document.getElementById('challanBody'),
    challanPlateImg: document.getElementById('challanPlateImg'),
    platePlaceholder: document.getElementById('platePlaceholder'),
    challanPlateText: document.getElementById('challanPlateText'),
    challanOwnerText: document.getElementById('challanOwnerText'),
    challanModelText: document.getElementById('challanModelText'),
    challanOffenseText: document.getElementById('challanOffenseText'),
    challanFineText: document.getElementById('challanFineText'),
    challanHistory: document.getElementById('challanHistoryContainer'),

    // SMS Toast
    smsToast: document.getElementById('smsNotificationToast'),
    smsToastBody: document.getElementById('smsToastBody'),

    // Buttons
    btnToggleSignal: document.getElementById('btnToggleSignal'),
    btnSimulateViolation: document.getElementById('btnSimulateViolation'),
    btnSimulateAmbulance: document.getElementById('btnSimulateAmbulance'),
    btnResetCounters: document.getElementById('btnResetCounters'),

    // Pipeline Stages
    stageCamera: document.getElementById('stageCamera'),
    stageDetect: document.getElementById('stageDetect'),
    stageTrack: document.getElementById('stageTrack'),
    stageViolation: document.getElementById('stageViolation'),
    stageOcr: document.getElementById('stageOcr'),
    stageChallan: document.getElementById('stageChallan'),
  };

  // --------------------------------------------------------------------------
  // 1. Digital Clock
  // --------------------------------------------------------------------------
  function updateClock() {
    const now = new Date();
    el.clock.textContent = now.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true,
    });
  }
  setInterval(updateClock, 1000);
  updateClock();

  // --------------------------------------------------------------------------
  // 2. Telemetry Polling (/api/state)
  // --------------------------------------------------------------------------
  async function pollState() {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      
      setConnectionStatus(true);
      renderState(data);
    } catch (err) {
      setConnectionStatus(false);
    }
  }

  function setConnectionStatus(connected) {
    if (isConnected !== connected) {
      isConnected = connected;
      if (connected) {
        el.reconnectingBadge.classList.add('hidden');
      } else {
        el.reconnectingBadge.classList.remove('hidden');
      }
    }
  }

  function renderState(data) {
    // 1. Counters
    const s = data.stats || {};
    el.counterVehicles.textContent = s.vehicles !== undefined ? s.vehicles.toLocaleString() : '0';
    el.counterViolations.textContent = s.violations !== undefined ? s.violations.toLocaleString() : '0';
    el.counterEmergencies.textContent = s.emergencies !== undefined ? s.emergencies.toLocaleString() : '0';
    el.counterFines.textContent = '₹' + (s.total_fines !== undefined ? s.total_fines.toLocaleString() : '0');
    el.fps.textContent = 'FPS: ' + (s.fps ? s.fps.toFixed(1) : '0.0');

    // 2. Traffic Signal
    const sig = data.signal || {};
    const state = (sig.state || 'GREEN').toUpperCase();
    const timeRem = Math.ceil(sig.time_remaining || 0);

    el.signalCountdown.textContent = String(timeRem).padStart(2, '0');
    el.signalStateText.textContent = `PHASE: ${state}`;

    el.signalLight.classList.remove('green', 'red', 'yellow');
    if (state === 'RED') {
      el.signalLight.classList.add('red');
      el.signalIcon.innerHTML = '&#10008;';
    } else if (state === 'YELLOW') {
      el.signalLight.classList.add('yellow');
      el.signalIcon.innerHTML = '&#9888;';
    } else {
      el.signalLight.classList.add('green');
      el.signalIcon.innerHTML = '&#10004;';
    }

    // 3. Emergency Lock State
    if (data.emergency_active || sig.is_emergency_locked) {
      el.signalLockBadge.classList.remove('hidden');
      el.emergencyBanner.classList.remove('hidden');
      el.screenFlash.classList.add('flash-green');
    } else {
      el.signalLockBadge.classList.add('hidden');
      el.emergencyBanner.classList.add('hidden');
      el.screenFlash.classList.remove('flash-green');
    }
  }

  // --------------------------------------------------------------------------
  // 3. Event Polling (/api/events?since=ID)
  // --------------------------------------------------------------------------
  async function pollEvents() {
    try {
      const res = await fetch(`/api/events?since=${lastEventId}`);
      if (!res.ok) return;
      const events = await res.json();

      if (Array.isArray(events) && events.length > 0) {
        for (const ev of events) {
          if (ev.id > lastEventId) {
            lastEventId = ev.id;
            handleEvent(ev);
          }
        }
      }
    } catch (e) {
      // Handled by state poller
    }
  }

  function handleEvent(ev) {
    if (ev.type === 'violation') {
      handleViolationEvent(ev.payload || {});
    } else if (ev.type === 'emergency') {
      handleEmergencyEvent(ev.payload || {});
    }
  }

  function handleViolationEvent(payload) {
    // 1. Red Border Screen Flash (2 seconds)
    el.screenFlash.classList.remove('flash-red');
    void el.screenFlash.offsetWidth; // Trigger reflow
    el.screenFlash.classList.add('flash-red');
    if (violationFlashTimer) clearTimeout(violationFlashTimer);
    violationFlashTimer = setTimeout(() => {
      el.screenFlash.classList.remove('flash-red');
    }, 2000);

    // 2. Pipeline Breadcrumb Flash
    flashPipelineStages();

    // 3. Update Latest Challan Card
    if (payload.plate_url) {
      el.challanPlateImg.src = payload.plate_url;
      el.challanPlateImg.classList.remove('hidden');
      el.platePlaceholder.classList.add('hidden');
    }

    el.challanPlateText.textContent = payload.plate || 'UNKNOWN';
    el.challanOwnerText.textContent = payload.owner || 'Registered Owner';
    el.challanModelText.textContent = payload.model || 'Motor Vehicle';
    el.challanOffenseText.textContent = 'Red Light Signal Disobedience';
    el.challanFineText.textContent = `PENALTY: INR ${(payload.fine || 1000).toLocaleString()}`;

    // Animate Challan Body Slide-in
    el.challanBody.classList.remove('slide-in');
    void el.challanBody.offsetWidth;
    el.challanBody.classList.add('slide-in');

    // 4. Update History Strip
    if (payload.plate) {
      recentChallans.unshift(payload.plate);
      if (recentChallans.length > 3) recentChallans.pop();
      renderHistoryStrip();
    }

    // 5. Show Simulated SMS Toast
    const maskedContact = payload.contact && payload.contact.length > 6 
      ? payload.contact.slice(0, 5) + '***' + payload.contact.slice(-2)
      : '+91 98*** 43210';
    
    showSmsToast(`e-Challan #${payload.challan_id || 1} sent to ${maskedContact} (Fine: INR ${(payload.fine || 1000).toLocaleString()}).`);
  }

  function renderHistoryStrip() {
    if (recentChallans.length === 0) {
      el.challanHistory.innerHTML = '<span class="history-empty">No recent infractions</span>';
      return;
    }
    el.challanHistory.innerHTML = recentChallans
      .map(p => `<span class="history-chip">${p}</span>`)
      .join('');
  }

  function handleEmergencyEvent(payload) {
    if (payload.status === 'active') {
      el.emergencyBanner.classList.remove('hidden');
      el.screenFlash.classList.add('flash-green');
      showSmsToast('EMERGENCY: Green Corridor Priority Engaged for Ambulance.');
    } else {
      el.emergencyBanner.classList.add('hidden');
      el.screenFlash.classList.remove('flash-green');
    }
  }

  function flashPipelineStages() {
    const stages = [
      el.stageCamera,
      el.stageDetect,
      el.stageTrack,
      el.stageViolation,
      el.stageOcr,
      el.stageChallan
    ];

    stages.forEach((st, idx) => {
      setTimeout(() => {
        st.classList.add('flash');
        setTimeout(() => st.classList.remove('flash'), 350);
      }, idx * 120);
    });
  }

  function showSmsToast(message) {
    el.smsToastBody.textContent = message;
    el.smsToast.classList.remove('hidden');

    if (smsToastTimer) clearTimeout(smsToastTimer);
    smsToastTimer = setTimeout(() => {
      el.smsToast.classList.add('hidden');
    }, 4500);
  }

  // --------------------------------------------------------------------------
  // 4. Interactive Demo Buttons
  // --------------------------------------------------------------------------
  el.btnToggleSignal.addEventListener('click', async () => {
    try {
      await fetch('/api/signal/toggle', { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  });

  el.btnSimulateViolation.addEventListener('click', async () => {
    try {
      await fetch('/api/demo/violation', { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  });

  el.btnSimulateAmbulance.addEventListener('click', async () => {
    try {
      await fetch('/api/demo/ambulance', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration: 15.0 })
      });
    } catch (e) {
      console.error(e);
    }
  });

  el.btnResetCounters.addEventListener('click', async () => {
    try {
      await fetch('/api/reset', { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  });

  // --------------------------------------------------------------------------
  // 5. Initial Challan Load
  // --------------------------------------------------------------------------
  async function loadInitialChallans() {
    try {
      const res = await fetch('/api/challans?limit=5');
      if (!res.ok) return;
      const challans = await res.json();

      if (Array.isArray(challans) && challans.length > 0) {
        const latest = challans[0];
        el.challanPlateText.textContent = latest.plate_number;
        el.challanOwnerText.textContent = latest.owner_name;
        el.challanModelText.textContent = latest.vehicle_model || 'Motor Vehicle';
        el.challanOffenseText.textContent = latest.violation_type || 'Red Light Violation';
        el.challanFineText.textContent = `PENALTY: INR ${(latest.fine_amount || 1000).toLocaleString()}`;
        
        if (latest.plate_url) {
          el.challanPlateImg.src = latest.plate_url;
          el.challanPlateImg.classList.remove('hidden');
          el.platePlaceholder.classList.add('hidden');
        }

        // Populate history strip
        challans.slice(0, 3).forEach(c => {
          if (c.plate_number && !recentChallans.includes(c.plate_number)) {
            recentChallans.push(c.plate_number);
          }
        });
        renderHistoryStrip();
      }
    } catch (e) {
      console.error(e);
    }
  }

  // --------------------------------------------------------------------------
  // 6. Polling Loops Initialization
  // --------------------------------------------------------------------------
  loadInitialChallans();
  setInterval(pollState, 500);
  setInterval(pollEvents, 500);

})();
