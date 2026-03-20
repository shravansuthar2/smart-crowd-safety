// ========== CONFIGURATION ==========
const API_URL = "http://localhost:8000/api";
let currentFile = null;
let originalImageData = null;  // Store original CLEAN image (no annotations)
let allAlerts = [];
let currentVideoJobId = null;
let videoPollingInterval = null;

// ========== INIT ==========
document.addEventListener("DOMContentLoaded", () => {
    setupClock();
    setupUploadArea();
    setupVideoUpload();
    loadAlerts();
    loadMissingPersons();

    // Auto-refresh alerts every 10 seconds
    setInterval(loadAlerts, 10000);

    // Update "last updated" every minute
    setInterval(() => {
        document.getElementById("lastUpdated").textContent = new Date().toLocaleString();
    }, 60000);
});

// ========== CLOCK ==========
function setupClock() {
    function updateClock() {
        const now = new Date();
        document.getElementById("currentTime").textContent = now.toLocaleTimeString("en-IN", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit"
        });
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// ========== DRAG & DROP UPLOAD ==========
function setupUploadArea() {
    const uploadArea = document.getElementById("uploadArea");
    if (!uploadArea) return;

    uploadArea.addEventListener("click", () => {
        document.getElementById("videoUpload").click();
    });

    uploadArea.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "#3b82f6";
        uploadArea.style.background = "rgba(59, 130, 246, 0.05)";
    });

    uploadArea.addEventListener("dragleave", () => {
        uploadArea.style.borderColor = "#1e293b";
        uploadArea.style.background = "transparent";
    });

    uploadArea.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = "#1e293b";
        uploadArea.style.background = "transparent";

        const files = e.dataTransfer.files;
        if (files.length > 0 && files[0].type.startsWith("video/")) {
            uploadVideo(files[0]);
        } else {
            showToast("Invalid File", "Please drop a video file (.mp4, .avi, .mov)", "warning");
        }
    });
}

// ========== FILE UPLOAD ==========
function setupImageUpload() {
    document.getElementById("imageUpload").addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            currentFile = e.target.files[0];
            displayUploadedImage(currentFile);
        }
    });
}

function displayUploadedImage(file) {
    const reader = new FileReader();
    reader.onload = (ev) => {
        const img = document.getElementById("liveFrame");
        img.src = ev.target.result;
        img.style.display = "block";
        document.getElementById("videoOverlay").style.display = "none";

        // Save original CLEAN image data (before any annotations)
        originalImageData = ev.target.result;

        // Reset detection results
        hideDetectionResults();
        hideBadge();

        // Auto-run all detections
        runAutoDetection();
    };
    reader.readAsDataURL(file);
    showToast("Image Uploaded", "Running all detections automatically...", "info");
}

// ========== AUTO DETECTION ==========
async function runAutoDetection() {
    const statusBar = document.getElementById("autoDetectStatus");
    statusBar.style.display = "block";

    // Reset all steps
    resetSteps();

    // Step 1: Crowd Density
    setStepStatus("stepCrowd", "running");
    await detectCrowd();
    setStepStatus("stepCrowd", "done");

    // Step 2: Emergency Detection
    setStepStatus("stepEmergency", "running");
    await detectEmergency();
    setStepStatus("stepEmergency", "done");

    // Step 4: Heatmap
    setStepStatus("stepHeatmap", "running");
    await generateHeatmap();
    setStepStatus("stepHeatmap", "done");

    showToast("Analysis Complete", "All detections finished", "success");
}

function setStepStatus(stepId, status) {
    const step = document.getElementById(stepId);
    step.className = `detect-step ${status}`;
}

function resetSteps() {
    ["stepCrowd", "stepEmergency", "stepHeatmap"].forEach(id => {
        document.getElementById(id).className = "detect-step";
    });
}

// ========== API HELPER ==========
async function apiCall(endpoint, formData) {
    if (!currentFile) {
        showToast("No Image", "Please upload an image first", "warning");
        return null;
    }

    try {
        const fd = formData || new FormData();
        if (!formData) {
            fd.append("file", currentFile);
        }

        const res = await fetch(`${API_URL}${endpoint}`, {
            method: "POST",
            body: fd,
        });

        if (!res.ok) {
            throw new Error(`Server error: ${res.status}`);
        }

        return await res.json();
    } catch (err) {
        console.error(`API Error (${endpoint}):`, err);
        showToast("Connection Error", "Cannot connect to backend. Make sure the server is running.", "error");
        return null;
    }
}

// ========== CROWD DETECTION ==========
async function detectCrowd() {
    const data = await apiCall("/detect/crowd");
    if (!data) return;

    // Update stats
    document.getElementById("crowdCount").textContent = data.count;
    updateCrowdBar(data.count, data.threshold);

    // Show annotated image
    showAnnotatedImage(data.annotated_image);

    // Show results
    const severity = data.is_overcrowded ? "danger" : (data.count > data.threshold * 0.7 ? "warning" : "safe");
    showDetectionResults([
        { label: "People Detected", value: data.count, status: severity },
        { label: "Threshold", value: data.threshold, status: "safe" },
        { label: "Status", value: data.is_overcrowded ? "OVERCROWDED" : "Normal", status: severity },
        { label: "Density", value: `${Math.round((data.count / data.threshold) * 100)}%`, status: severity },
    ]);

    // Show badge
    if (data.is_overcrowded) {
        showBadge("OVERCROWDED", "danger");
        addAlertToUI("Crowd Overcrowding", `Count: ${data.count} exceeds threshold of ${data.threshold}`, "critical", "crowd_density");
        showToast("Overcrowding Alert", `${data.count} people detected!`, "error");
    } else {
        showBadge(`${data.count} People`, "safe");
        showToast("Crowd Scan Complete", `${data.count} people detected`, "success");
    }
}

function updateCrowdBar(count, threshold) {
    const percent = Math.min((count / threshold) * 100, 100);
    document.getElementById("crowdBar").style.width = `${percent}%`;
}

// ========== PICKPOCKET DETECTION ==========
async function detectPickpocket() {
    const data = await apiCall("/detect/pickpocket");
    if (!data) return;

    showAnnotatedImage(data.annotated_image);

    if (data.suspicious) {
        showBadge("SUSPICIOUS ACTIVITY", "danger");
        showDetectionResults([
            { label: "Status", value: "SUSPICIOUS", status: "danger" },
            { label: "Alerts", value: data.alerts.length, status: "danger" },
        ]);
        addAlertToUI("Pickpocket Suspected", `${data.alerts.length} suspicious activity detected`, "high", "pickpocket");
        showToast("Suspicious Activity", "Possible pickpocket behavior detected!", "warning");
    } else {
        showBadge("No Suspicious Activity", "safe");
        showDetectionResults([
            { label: "Status", value: "All Clear", status: "safe" },
        ]);
        showToast("Scan Complete", "No suspicious activity detected", "success");
    }
}

// ========== EMERGENCY DETECTION ==========
async function detectEmergency() {
    const data = await apiCall("/detect/emergency");
    if (!data) return;

    showAnnotatedImage(data.annotated_image);

    const hasFire = data.fire_detected;
    const hasSmoke = data.smoke_detected;
    const hasFall = data.fall_detected;
    const hasFight = data.fight_detected;
    const hasEmergency = hasFire || hasFall || hasFight;

    showDetectionResults([
        { label: "Fire Detection", value: hasFire ? "FIRE!" : "None", status: hasFire ? "danger" : "safe" },
        { label: "Smoke Detection", value: hasSmoke ? "SMOKE!" : "None", status: hasSmoke ? "warning" : "safe" },
        { label: "Fall Detection", value: hasFall ? "DETECTED" : "None", status: hasFall ? "danger" : "safe" },
        { label: "Fight Detection", value: hasFight ? "DETECTED" : "None", status: hasFight ? "danger" : "safe" },
        { label: "Overall Status", value: hasEmergency ? "EMERGENCY" : "Normal", status: hasEmergency ? "danger" : "safe" },
    ]);

    if (hasFire) {
        showBadge("FIRE DETECTED", "danger");
        addAlertToUI("Fire Detected", "Fire/flames detected - evacuate immediately", "critical", "emergency");
        showToast("FIRE!", "Fire detected - immediate evacuation needed!", "error");
    }
    if (hasFall) {
        showBadge("FALL DETECTED", "danger");
        addAlertToUI("Fall Detected", "Person fallen - immediate help needed", "critical", "emergency");
        showToast("Emergency!", "Fall detected!", "error");
    }
    if (hasFight) {
        showBadge("FIGHT DETECTED", "danger");
        addAlertToUI("Fight Detected", "Physical altercation in progress", "critical", "emergency");
        showToast("Emergency!", "Fight detected!", "error");
    }
    if (!hasEmergency) {
        showBadge("All Clear", "safe");
        showToast("Emergency Check", "No emergencies detected", "success");
    }
}

// ========== HEATMAP ==========
async function generateHeatmap() {
    const data = await apiCall("/detect/heatmap");
    if (!data) return;

    const heatmapImg = document.getElementById("heatmapImage");
    heatmapImg.src = `data:image/jpeg;base64,${data.heatmap}`;
    heatmapImg.style.display = "block";
    document.getElementById("heatmapEmpty").style.display = "none";

    showToast("Heatmap Generated", "Crowd density heatmap is ready", "info");
}

// ========== VIDEO PROCESSING ==========

function setupVideoUpload() {
    document.getElementById("videoUpload").addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            uploadVideo(e.target.files[0]);
        }
    });
}

async function uploadVideo(file) {
    window._notifiedPersons = new Set(); // Reset face match notifications
    const maxSize = 100 * 1024 * 1024; // 100MB limit
    if (file.size > maxSize) {
        showToast("File Too Large", "Video must be under 100MB", "warning");
        return;
    }

    showToast("Uploading Video", `${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`, "info");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`${API_URL}/detect/video?skip_frames=3`, {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (data.error) {
            showToast("Error", data.error, "error");
            return;
        }

        currentVideoJobId = data.job_id;
        showToast("Processing Started", `${data.total_frames} frames, ~${data.duration}s video`, "success");

        // Show video status panel
        const statusPanel = document.getElementById("videoStatus");
        statusPanel.style.display = "block";
        document.getElementById("videoInfo").textContent =
            `File: ${file.name} | Duration: ${data.duration}s | Resolution: ${data.resolution} | Frames: ${data.total_frames}`;

        // Hide image overlay, show video processing
        document.getElementById("videoOverlay").style.display = "none";

        // Start polling for progress
        startVideoPolling(data.job_id);

    } catch (err) {
        showToast("Upload Failed", "Cannot connect to backend", "error");
    }
}

function startVideoPolling(jobId) {
    // Clear any existing polling
    if (videoPollingInterval) clearInterval(videoPollingInterval);

    videoPollingInterval = setInterval(async () => {
        try {
            const res = await fetch(`${API_URL}/detect/video/status/${jobId}`);
            const data = await res.json();

            // Update progress bar
            document.getElementById("videoProgressFill").style.width = `${data.progress}%`;
            document.getElementById("videoProgressText").textContent = `${data.progress}%`;

            // Update live preview frame
            if (data.latest_frame) {
                const img = document.getElementById("liveFrame");
                img.src = `data:image/jpeg;base64,${data.latest_frame}`;
                img.style.display = "block";
            }

            // Save original CLEAN frame (no annotations) for face search
            if (data.original_frame) {
                originalImageData = `data:image/jpeg;base64,${data.original_frame}`;
            }

            // Update live heatmap
            if (data.current_heatmap) {
                const heatmapImg = document.getElementById("heatmapImage");
                heatmapImg.src = `data:image/jpeg;base64,${data.current_heatmap}`;
                heatmapImg.style.display = "block";
                document.getElementById("heatmapEmpty").style.display = "none";
            }

            // Update crowd count LIVE with current frame's people count
            if (data.current_people !== undefined) {
                document.getElementById("crowdCount").textContent = data.current_people;
            }

            // Show LIVE face match alerts
            if (data.face_matches && data.face_matches.length > 0) {
                data.face_matches.forEach(name => {
                    // Only show toast once per person (avoid spam)
                    if (!window._notifiedPersons) window._notifiedPersons = new Set();
                    if (!window._notifiedPersons.has(name)) {
                        window._notifiedPersons.add(name);
                        showToast("PERSON FOUND!", `${name} detected in video at frame ${data.current_frame}`, "error");
                        addAlertToUI("Missing Person Found!", `${name} detected in video frame ${data.current_frame}`, "critical", "missing_person");
                    }
                });
            }

            // Update info
            const faceInfo = data.face_matches && data.face_matches.length > 0
                ? ` | FOUND: ${data.face_matches.join(", ")}`
                : "";
            document.getElementById("videoInfo").textContent =
                `Frame: ${data.current_frame}/${data.total_frames} | People: ${data.current_people || 0} | Status: ${data.status}${faceInfo}`;

            // Completed
            if (data.status === "completed") {
                clearInterval(videoPollingInterval);
                videoPollingInterval = null;
                showVideoResults(data);
                showToast("Video Complete", "All frames processed!", "success");
            }

            // Error
            if (data.status === "error") {
                clearInterval(videoPollingInterval);
                videoPollingInterval = null;
                showToast("Processing Error", data.error || "Unknown error", "error");
            }

        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 2000); // Poll every 2 seconds
}

function showVideoResults(data) {
    const stats = data.stats;

    // Show stats grid
    const statsDiv = document.getElementById("videoStats");
    statsDiv.style.display = "grid";
    const personsFound = stats.persons_found || [];
    const faceFrames = stats.face_match_frames || 0;

    statsDiv.innerHTML = `
        <div class="video-stat-item">
            <div class="stat-val">${stats.processed_frames}</div>
            <div class="stat-lbl">Frames Processed</div>
        </div>
        <div class="video-stat-item">
            <div class="stat-val">${stats.avg_people_per_frame}</div>
            <div class="stat-lbl">Avg People/Frame</div>
        </div>
        <div class="video-stat-item">
            <div class="stat-val">${stats.max_people_in_frame}</div>
            <div class="stat-lbl">Max People</div>
        </div>
        <div class="video-stat-item">
            <div class="stat-val red">${stats.overcrowded_frames}</div>
            <div class="stat-lbl">Overcrowded Frames</div>
        </div>
        <div class="video-stat-item">
            <div class="stat-val red">${stats.emergency_frames}</div>
            <div class="stat-lbl">Emergency Frames</div>
        </div>
        <div class="video-stat-item">
            <div class="stat-val ${faceFrames > 0 ? 'green' : ''}">${faceFrames}</div>
            <div class="stat-lbl">Face Match Frames</div>
        </div>
        ${personsFound.length > 0 ? `
        <div class="video-stat-item" style="grid-column: span 2">
            <div class="stat-val green">${personsFound.join(", ")}</div>
            <div class="stat-lbl">Persons Found in Video</div>
        </div>
        ` : ''}
    `;

    // Show timeline (only alert frames)
    if (data.timeline && data.timeline.length > 0) {
        const timelineDiv = document.getElementById("videoTimeline");
        timelineDiv.style.display = "block";

        const alertFrames = data.timeline.filter(f => f.overcrowded || f.fire || f.fall || f.fight || (f.face_match && f.face_match.length > 0));
        const normalCount = data.timeline.length - alertFrames.length;

        let html = `<div class="timeline-item normal-row"><span class="timeline-time">---</span><span class="timeline-event">${normalCount} normal frames</span></div>`;

        alertFrames.forEach(f => {
            let events = [];
            if (f.overcrowded) events.push(`Overcrowded (${f.people} people)`);
            if (f.fire) events.push("FIRE detected");
            if (f.fall) events.push("Fall detected");
            if (f.fight) events.push("Fight detected");
            if (f.face_match && f.face_match.length > 0) events.push(`FOUND: ${f.face_match.join(", ")}`);

            html += `<div class="timeline-item alert-row">
                <span class="timeline-time">${f.time}s</span>
                <span class="timeline-event">${events.join(" | ")}</span>
            </div>`;
        });

        timelineDiv.innerHTML = html;
    }

    // Show download button
    document.getElementById("btnDownloadVideo").style.display = "inline-flex";

    // Update crowd count with the LAST frame's people count (matches what's shown on screen)
    if (data.timeline && data.timeline.length > 0) {
        const lastFrame = data.timeline[data.timeline.length - 1];
        document.getElementById("crowdCount").textContent = lastFrame.people;
    } else {
        document.getElementById("crowdCount").textContent = stats.avg_people_per_frame;
    }
}

async function downloadVideo() {
    if (!currentVideoJobId) return;

    showToast("Downloading", "Preparing processed video...", "info");

    try {
        const res = await fetch(`${API_URL}/detect/video/download/${currentVideoJobId}`);

        if (!res.ok) {
            showToast("Error", "Video not ready", "error");
            return;
        }

        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `processed_video.mp4`;
        a.click();
        URL.revokeObjectURL(url);

        showToast("Downloaded", "Video saved!", "success");
    } catch (err) {
        showToast("Error", "Download failed", "error");
    }
}

// ========== MISSING PERSONS ==========
function previewPersonPhoto(input) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const preview = document.getElementById("personPhotoPreview");
            preview.src = e.target.result;
            preview.style.display = "block";
        };
        reader.readAsDataURL(input.files[0]);
    }
}

async function registerPerson() {
    const name = document.getElementById("personName").value.trim();
    const details = document.getElementById("personDetails").value.trim();
    const photo = document.getElementById("personPhoto").files[0];

    if (!name) {
        showToast("Missing Info", "Please enter the person's name", "warning");
        return;
    }
    if (!photo) {
        showToast("Missing Photo", "Please upload a photo of the person", "warning");
        return;
    }

    const formData = new FormData();
    formData.append("name", name);
    formData.append("details", details);
    formData.append("photo", photo);

    try {
        const res = await fetch(`${API_URL}/persons/register`, {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        if (data.success) {
            showToast("Person Registered", `${name} has been registered as missing`, "success");
            // Reset form
            document.getElementById("personName").value = "";
            document.getElementById("personDetails").value = "";
            document.getElementById("personPhoto").value = "";
            document.getElementById("personPhotoPreview").style.display = "none";
            loadMissingPersons();
        } else {
            showToast("Registration Failed", data.message, "error");
        }
    } catch (err) {
        showToast("Connection Error", "Cannot connect to backend", "error");
    }
}

async function searchPerson() {
    showToast("Searching", "Scanning for missing persons...", "info");

    let formData = new FormData();

    // PRIORITY 1: Use original CLEAN image (no annotations drawn on it)
    if (currentFile && currentFile.type && currentFile.type.startsWith("image/")) {
        formData.append("file", currentFile);
    }
    // PRIORITY 2: Use saved original image data (before auto-detection drew boxes)
    else if (originalImageData) {
        const response = await fetch(originalImageData);
        const blob = await response.blob();
        formData.append("file", blob, "original_frame.jpg");
    }
    // PRIORITY 3: Capture from displayed frame (last resort, may have annotations)
    else {
        const liveFrame = document.getElementById("liveFrame");
        if (!liveFrame.src || liveFrame.style.display === "none" || !liveFrame.naturalWidth) {
            showToast("No Image", "Upload an image or process a video first", "warning");
            return;
        }

        const canvas = document.createElement("canvas");
        canvas.width = liveFrame.naturalWidth;
        canvas.height = liveFrame.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(liveFrame, 0, 0);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
        formData.append("file", blob, "current_frame.jpg");
    }

    try {
        const res = await fetch(`${API_URL}/persons/search`, {
            method: "POST",
            body: formData,
        });
        const data = await res.json();

        // Show annotated image with face boxes if available
        if (data.annotated_image) {
            showAnnotatedImage(data.annotated_image);
        }

        if (data.count > 0) {
            data.matches.forEach((m) => {
                addAlertToUI(
                    "Missing Person Found!",
                    `${m.person_name} - Confidence: ${(m.confidence * 100).toFixed(1)}%`,
                    "critical",
                    "missing_person"
                );
            });
            showToast("Match Found!", `${data.count} missing person(s) identified`, "error");
        } else {
            showToast("No Match", "No missing persons found in this frame", "info");
        }
    } catch (err) {
        showToast("Connection Error", "Cannot connect to backend", "error");
    }
}

async function loadMissingPersons() {
    try {
        const res = await fetch(`${API_URL}/persons/list`);
        const data = await res.json();

        document.getElementById("missingCount").textContent = data.persons.length;
        document.getElementById("missingBadge").textContent = `${data.persons.length} registered`;

        const grid = document.getElementById("personsList");
        if (data.persons.length === 0) {
            grid.innerHTML = '<div class="empty-state small"><p>No missing persons registered</p></div>';
            return;
        }

        grid.innerHTML = data.persons.map((p) => `
            <div class="person-card">
                <img src="${p.image_url || 'data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%231e293b%22 width=%22100%22 height=%22100%22/><text fill=%22%23475569%22 font-size=%2240%22 x=%2250%22 y=%2260%22 text-anchor=%22middle%22>?</text></svg>'}" alt="${p.name}">
                <h4>${p.name}</h4>
                <p>${p.details || 'No details'}</p>
                <span class="person-status ${p.status === 'found' ? 'status-found' : 'status-missing'}">
                    ${p.status === 'found' ? 'Found' : 'Missing'}
                </span>
                <button class="btn btn-sm btn-danger" onclick="deletePerson('${p.name}')" style="margin-top:8px">
                    Delete
                </button>
            </div>
        `).join("");
    } catch (err) {
        console.error("Could not load missing persons:", err);
    }
}

async function deletePerson(name) {
    try {
        const res = await fetch(`${API_URL}/persons/delete/${name}`, { method: "DELETE" });
        const data = await res.json();
        showToast("Deleted", data.message, "success");
        loadMissingPersons();
    } catch (err) {
        showToast("Error", "Could not delete person", "error");
    }
}

// ========== ALERTS ==========
async function loadAlerts() {
    try {
        const res = await fetch(`${API_URL}/alerts/?limit=30`);
        const data = await res.json();

        allAlerts = data.alerts || [];
        const activeCount = allAlerts.filter(a => a.status === "active").length;
        document.getElementById("activeAlerts").textContent = activeCount;

        renderAlerts(allAlerts);
        document.getElementById("lastUpdated").textContent = new Date().toLocaleString();
    } catch (err) {
        // Backend not running — silent fail, user sees "Connection Error" on actions
        console.error("Could not load alerts");
    }
}

function filterAlerts() {
    const filter = document.getElementById("alertFilter").value;
    if (filter === "all") {
        renderAlerts(allAlerts);
    } else {
        renderAlerts(allAlerts.filter(a => a.type === filter));
    }
}

function renderAlerts(alerts) {
    const list = document.getElementById("alertsList");

    if (!alerts || alerts.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <div class="empty-icon">i</div>
                <p>No alerts yet</p>
                <span>Alerts will appear here when detected</span>
            </div>`;
        return;
    }

    list.innerHTML = alerts.map((a) => `
        <div class="alert-item alert-${a.severity}" data-type="${a.type}">
            <h4>
                <span class="severity-badge severity-${a.severity}">${a.severity}</span>
                ${getAlertTitle(a.type)}
            </h4>
            <p>${formatAlertDetails(a.details)}</p>
            <div class="alert-meta">
                <span class="alert-time">${formatTime(a.timestamp)}</span>
                <div class="alert-actions">
                    ${a.status === 'active' ? `
                        <button class="btn btn-sm btn-outline" onclick="acknowledgeAlert('${a.id}')">Acknowledge</button>
                        <button class="btn btn-sm btn-primary" onclick="resolveAlert('${a.id}')">Resolve</button>
                    ` : `
                        <span class="severity-badge" style="background:#14532d;color:#22c55e">${a.status}</span>
                    `}
                </div>
            </div>
        </div>
    `).join("");
}

// Add alert to UI without server (for immediate feedback)
function addAlertToUI(title, description, severity, type) {
    const alert = {
        id: "local-" + Date.now(),
        type: type,
        severity: severity,
        details: description,
        timestamp: new Date().toISOString(),
        status: "active"
    };

    allAlerts.unshift(alert);
    const activeCount = allAlerts.filter(a => a.status === "active").length;
    document.getElementById("activeAlerts").textContent = activeCount;
    renderAlerts(allAlerts);
}

async function acknowledgeAlert(alertId) {
    if (alertId.startsWith("local-")) {
        // Local alert — just update UI
        const alert = allAlerts.find(a => a.id === alertId);
        if (alert) alert.status = "acknowledged";
        renderAlerts(allAlerts);
        return;
    }

    try {
        await fetch(`${API_URL}/alerts/${alertId}/acknowledge`, { method: "PUT" });
        showToast("Alert Acknowledged", "", "info");
        loadAlerts();
    } catch (err) {
        showToast("Error", "Could not acknowledge alert", "error");
    }
}

async function resolveAlert(alertId) {
    if (alertId.startsWith("local-")) {
        const alert = allAlerts.find(a => a.id === alertId);
        if (alert) alert.status = "resolved";
        renderAlerts(allAlerts);
        return;
    }

    try {
        await fetch(`${API_URL}/alerts/${alertId}/resolve`, { method: "PUT" });
        showToast("Alert Resolved", "", "success");
        loadAlerts();
    } catch (err) {
        showToast("Error", "Could not resolve alert", "error");
    }
}

async function clearAllAlerts() {
    try {
        await fetch(`${API_URL}/alerts/clear`, { method: "DELETE" });
        allAlerts = [];
        document.getElementById("activeAlerts").textContent = "0";
        renderAlerts([]);
        showToast("Cleared", "All alerts removed", "success");
    } catch (err) {
        // Also clear local alerts
        allAlerts = [];
        document.getElementById("activeAlerts").textContent = "0";
        renderAlerts([]);
    }
}

// ========== UI HELPERS ==========
function showAnnotatedImage(base64) {
    const img = document.getElementById("liveFrame");
    img.src = `data:image/jpeg;base64,${base64}`;
    img.style.display = "block";
    document.getElementById("videoOverlay").style.display = "none";
}

function showDetectionResults(rows) {
    const container = document.getElementById("detectionResults");
    const content = document.getElementById("resultsContent");

    content.innerHTML = rows.map(r => `
        <div class="result-row">
            <span class="result-label">${r.label}</span>
            <span class="result-value ${r.status}">${r.value}</span>
        </div>
    `).join("");

    container.style.display = "block";
}

function hideDetectionResults() {
    document.getElementById("detectionResults").style.display = "none";
}

function showBadge(text, type) {
    const badge = document.getElementById("detectionBadge");
    const badgeText = document.getElementById("badgeText");
    badgeText.textContent = text;
    badge.className = `detection-badge badge-${type}`;
    badge.style.display = "block";
}

function hideBadge() {
    document.getElementById("detectionBadge").style.display = "none";
}

function getAlertTitle(type) {
    const titles = {
        crowd_density: "Crowd Overcrowding",
        pickpocket: "Suspicious Activity",
        emergency: "Emergency Detected",
        missing_person: "Missing Person Found",
    };
    return titles[type] || type;
}

function formatAlertDetails(details) {
    if (!details) return "No details";
    if (typeof details === "string") return details;
    return Object.entries(details).map(([k, v]) => `${k}: ${v}`).join(" | ");
}

function formatTime(timestamp) {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    const now = new Date();
    const diff = Math.floor((now - date) / 1000);

    if (diff < 60) return "Just now";
    if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
    return date.toLocaleDateString();
}

// ========== TOAST NOTIFICATIONS ==========
function showToast(title, message, type = "info") {
    const container = document.getElementById("toastContainer");

    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <h4>${title}</h4>
            ${message ? `<p>${message}</p>` : ""}
        </div>
        <button class="toast-close" onclick="this.parentElement.remove()">x</button>
    `;

    container.appendChild(toast);

    // Auto remove after 4 seconds
    setTimeout(() => {
        toast.style.animation = "slideOut 0.3s ease forwards";
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}
