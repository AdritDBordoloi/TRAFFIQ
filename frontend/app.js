/**
 * TRAFFIQ dashboard: polls the backend for telemetry and violations, and handles
 * the video source, signal and demo controls. Vanilla JS with no dependencies.
 */

(function () {
  'use strict';

  const POLL_INTERVAL_MS = 500;
  const MAX_VIOLATION_ROWS = 3;
  const DEFAULT_FINE = 1000;
  const WEBCAM_OPTION = 'camera:0';

  let lastEventId = 0;
  // Newest first: the first row fills the "Latest violation" panel, the rest go under "Earlier".
  let violations = [];

  const byId = (id) => document.getElementById(id);
  const el = {
    connection: byId('connection'),
    connectionText: byId('connectionText'),
    fps: byId('fpsDisplay'),
    detect: byId('detectDisplay'),

    sourceSelect: byId('sourceSelect'),
    sourceStatus: byId('sourceStatus'),
    btnUploadVideo: byId('btnUploadVideo'),
    videoFileInput: byId('videoFileInput'),

    btnDemoMenu: byId('btnDemoMenu'),
    demoMenu: byId('demoMenu'),
    btnSimulateViolation: byId('btnSimulateViolation'),
    btnSimulateAmbulance: byId('btnSimulateAmbulance'),
    btnResetCounters: byId('btnResetCounters'),

    emergencyAlert: byId('emergencyAlert'),
    emergencyTitle: byId('emergencyTitle'),

    counterVehicles: byId('counterVehicles'),
    counterViolations: byId('counterViolations'),
    counterEmergencies: byId('counterEmergencies'),
    counterFines: byId('counterFines'),

    signalMode: byId('signalMode'),
    signalState: byId('signalState'),
    signalCountdownWrap: byId('signalCountdownWrap'),
    signalCountdown: byId('signalCountdown'),
    btnToggleSignal: byId('btnToggleSignal'),

    latestTime: byId('latestTime'),
    latestEmpty: byId('latestEmpty'),
    latestBody: byId('latestBody'),
    latestPhoto: byId('latestPhoto'),
    latestPlate: byId('latestPlate'),
    latestFine: byId('latestFine'),
    latestOwner: byId('latestOwner'),
    earlierSection: byId('earlierSection'),
    earlierList: byId('earlierList'),
  };

  const formatCount = (n) => (n || 0).toLocaleString('en-IN');
  const formatFine = (n) => '₹' + (n || DEFAULT_FINE).toLocaleString('en-IN');
  // Timestamps arrive as "YYYY-MM-DD HH:MM:SS"; only the time is shown.
  const formatTime = (timestamp) => String(timestamp || '').split(' ').pop();

  async function post(url, body) {
    const options = { method: 'POST' };
    if (body !== undefined) {
      options.headers = { 'Content-Type': 'application/json' };
      options.body = JSON.stringify(body);
    }
    const res = await fetch(url, options);
    return res.json();
  }

  // --------------------------------------------------------------------------
  // Telemetry
  // --------------------------------------------------------------------------
  function setConnected(connected) {
    el.connection.classList.toggle('is-offline', !connected);
    el.connectionText.textContent = connected ? 'Live' : 'Reconnecting…';
    el.fps.hidden = !connected;
    el.detect.hidden = !connected;
  }

  function renderState(data) {
    const stats = data.stats || {};
    el.counterVehicles.textContent = formatCount(stats.vehicles);
    el.counterViolations.textContent = formatCount(stats.violations);
    el.counterEmergencies.textContent = formatCount(stats.emergencies);
    el.counterFines.textContent = '₹' + formatCount(stats.total_fines);
    el.fps.textContent = `${Math.round(stats.fps || 0)} FPS`;
    el.detect.textContent = `Detection ${Math.round(stats.detect_fps || 0)}/s`;

    const signal = data.signal || {};
    const state = (signal.state || 'GREEN').toUpperCase();
    const locked = Boolean(data.emergency_active || signal.is_emergency_locked);

    el.signalState.textContent = state;
    el.signalState.dataset.state = state;
    el.signalMode.textContent = locked ? 'Emergency hold' : 'Automatic cycle';
    el.signalCountdownWrap.hidden = locked;
    el.signalCountdown.textContent = String(Math.ceil(signal.time_remaining || 0)).padStart(2, '0');
    el.btnToggleSignal.disabled = locked;
    el.btnToggleSignal.textContent = locked ? 'Locked during corridor' : 'Toggle signal';

    el.emergencyAlert.hidden = !locked;
    if (locked) {
      const trackId = signal.emergency_track_id;
      el.emergencyTitle.textContent = trackId === undefined || trackId === null
        ? 'Ambulance: green corridor active'
        : `Ambulance #${trackId}: green corridor active`;
    }
  }

  async function pollState() {
    try {
      const res = await fetch('/api/state');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      renderState(await res.json());
      setConnected(true);
    } catch (e) {
      setConnected(false);
    }
  }

  // --------------------------------------------------------------------------
  // Violations
  // --------------------------------------------------------------------------
  function renderViolations() {
    const latest = violations[0];
    el.latestEmpty.hidden = Boolean(latest);
    el.latestBody.hidden = !latest;
    el.latestTime.textContent = latest ? latest.time : '';

    if (latest) {
      el.latestPlate.textContent = latest.plate;
      el.latestFine.textContent = formatFine(latest.fine);
      el.latestOwner.textContent = [latest.owner, latest.model].filter(Boolean).join(' · ');
      el.latestPhoto.hidden = !latest.image;
      if (latest.image) {
        el.latestPhoto.src = latest.image;
      } else {
        el.latestPhoto.removeAttribute('src');
      }
    }

    const earlier = violations.slice(1, MAX_VIOLATION_ROWS);
    el.earlierSection.hidden = earlier.length === 0;
    el.earlierList.replaceChildren(...earlier.map((row) => {
      const item = document.createElement('li');
      const cells = [
        ['earlier-time', row.time],
        ['earlier-plate', row.plate],
        ['', formatFine(row.fine)],
      ];
      cells.forEach(([className, text]) => {
        const cell = document.createElement('span');
        cell.className = className;
        cell.textContent = text;
        item.appendChild(cell);
      });
      return item;
    }));
  }

  function addViolation(timestamp, payload) {
    violations.unshift({
      time: formatTime(timestamp),
      plate: payload.plate || 'UNKNOWN',
      owner: payload.owner,
      model: payload.model,
      fine: payload.fine,
      image: payload.image_url,
    });
    violations = violations.slice(0, MAX_VIOLATION_ROWS);
    renderViolations();
  }

  async function pollEvents() {
    try {
      const res = await fetch(`/api/events?since=${lastEventId}`);
      if (!res.ok) return;
      const events = await res.json();
      events.forEach((ev) => {
        if (ev.id <= lastEventId) return;
        lastEventId = ev.id;
        if (ev.type === 'violation') addViolation(ev.timestamp, ev.payload || {});
      });
    } catch (e) {
      // pollState reports the connection problem
    }
  }

  async function loadViolations() {
    try {
      const res = await fetch(`/api/challans?limit=${MAX_VIOLATION_ROWS}`);
      if (!res.ok) return;
      const challans = await res.json();
      // A live event may already have arrived; those rows are newer, so keep them.
      if (violations.length > 0) return;
      violations = challans.map((c) => ({
        time: formatTime(c.timestamp),
        plate: c.plate_number,
        owner: c.owner_name,
        model: c.vehicle_model,
        fine: c.fine_amount,
        image: c.image_url,
      }));
      renderViolations();
    } catch (e) {
      console.error(e);
    }
  }

  // --------------------------------------------------------------------------
  // Video source and upload
  // --------------------------------------------------------------------------
  function addSourceOption(value, label) {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    el.sourceSelect.appendChild(option);
  }

  async function loadSources() {
    try {
      const res = await fetch('/api/source');
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();

      el.sourceSelect.replaceChildren();
      addSourceOption(WEBCAM_OPTION, 'Webcam');
      data.videos.forEach((name) => addSourceOption(`video:${name}`, name));

      const current = data.current || {};
      if (current.kind === 'video') {
        // A video passed with --source may live outside the uploads folder
        if (!data.videos.includes(current.label)) {
          addSourceOption(`video:${current.label}`, current.label);
        }
        el.sourceSelect.value = `video:${current.label}`;
      } else {
        el.sourceSelect.value = WEBCAM_OPTION;
      }
    } catch (e) {
      el.sourceStatus.textContent = 'Could not load video sources.';
    }
  }

  el.sourceSelect.addEventListener('change', async () => {
    const value = el.sourceSelect.value;
    const body = value === WEBCAM_OPTION
      ? { camera: 0 }
      : { video: value.slice('video:'.length) };
    el.sourceStatus.textContent = 'Switching…';
    try {
      const data = await post('/api/source', body);
      el.sourceStatus.textContent = data.success ? '' : (data.message || 'Could not switch source.');
    } catch (e) {
      el.sourceStatus.textContent = 'Could not switch source.';
    }
    loadSources();
  });

  el.btnUploadVideo.addEventListener('click', () => el.videoFileInput.click());

  el.videoFileInput.addEventListener('change', () => {
    const file = el.videoFileInput.files[0];
    if (!file) return;

    const form = new FormData();
    form.append('video', file);
    el.btnUploadVideo.disabled = true;
    el.sourceSelect.disabled = true;

    // XMLHttpRequest rather than fetch: it reports upload progress, which matters for large videos
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        el.sourceStatus.textContent = `Uploading ${Math.round((e.loaded / e.total) * 100)}%`;
      }
    };
    const finish = (message) => {
      el.sourceStatus.textContent = message;
      el.btnUploadVideo.disabled = false;
      el.sourceSelect.disabled = false;
      el.videoFileInput.value = '';
      loadSources();
    };
    xhr.onload = () => {
      const data = xhr.response || {};
      finish(data.success ? '' : (data.message || `Upload failed (${xhr.status}).`));
    };
    xhr.onerror = () => finish('Upload failed. Check that the dashboard is still running.');
    el.sourceStatus.textContent = 'Uploading 0%';
    xhr.send(form);
  });

  // --------------------------------------------------------------------------
  // Signal and demo controls
  // --------------------------------------------------------------------------
  function setDemoMenuOpen(open) {
    el.demoMenu.hidden = !open;
    el.btnDemoMenu.setAttribute('aria-expanded', String(open));
  }

  el.btnDemoMenu.addEventListener('click', (e) => {
    e.stopPropagation();
    setDemoMenuOpen(el.demoMenu.hidden);
  });
  document.addEventListener('click', () => setDemoMenuOpen(false));
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !el.demoMenu.hidden) {
      setDemoMenuOpen(false);
      el.btnDemoMenu.focus();
    }
  });

  function bindAction(button, action) {
    button.addEventListener('click', async () => {
      try {
        await action();
      } catch (e) {
        console.error(e);
      }
    });
  }

  bindAction(el.btnToggleSignal, () => post('/api/signal/toggle'));
  bindAction(el.btnSimulateViolation, () => post('/api/demo/violation'));
  bindAction(el.btnSimulateAmbulance, () => post('/api/demo/ambulance', { duration: 15.0 }));
  bindAction(el.btnResetCounters, () => post('/api/reset'));

  // --------------------------------------------------------------------------
  // Start
  // --------------------------------------------------------------------------
  renderViolations();
  loadSources();
  loadViolations();
  pollState();
  setInterval(pollState, POLL_INTERVAL_MS);
  setInterval(pollEvents, POLL_INTERVAL_MS);
})();
