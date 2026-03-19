// ========== FIREBASE CONFIGURATION ==========
// Replace these values with your Firebase project config
// Get from: Firebase Console > Project Settings > General > Your apps > Web app

const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
};

// ========== FIREBASE REAL-TIME LISTENER ==========
// This file sets up real-time alert listening from Firestore
// So alerts appear INSTANTLY on dashboard without polling

// NOTE: To enable real-time Firebase, add these scripts to index.html <head>:
//
// <script src="https://www.gstatic.com/firebasejs/10.7.1/firebase-app-compat.js"></script>
// <script src="https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore-compat.js"></script>
//
// Then uncomment the code below:

/*
// Initialize Firebase
firebase.initializeApp(firebaseConfig);
const db = firebase.firestore();

// Listen for new alerts in real-time
function setupRealtimeAlerts() {
    db.collection("alerts")
        .where("status", "==", "active")
        .orderBy("timestamp", "desc")
        .limit(30)
        .onSnapshot((snapshot) => {
            snapshot.docChanges().forEach((change) => {
                if (change.type === "added") {
                    const alert = { id: change.doc.id, ...change.doc.data() };

                    // Show toast for new alert
                    showToast(
                        getAlertTitle(alert.type),
                        formatAlertDetails(alert.details),
                        alert.severity === "critical" ? "error" : "warning"
                    );

                    // Play alert sound for critical
                    if (alert.severity === "critical") {
                        playAlertSound();
                    }
                }
            });

            // Update alert count
            const activeCount = snapshot.size;
            document.getElementById("activeAlerts").textContent = activeCount;
        });
}

// Listen for missing person matches
function setupMissingPersonListener() {
    db.collection("alerts")
        .where("type", "==", "missing_person")
        .where("status", "==", "active")
        .onSnapshot((snapshot) => {
            snapshot.docChanges().forEach((change) => {
                if (change.type === "added") {
                    const data = change.doc.data();
                    showToast(
                        "MISSING PERSON FOUND!",
                        `${data.details?.person_name} identified with ${(data.details?.confidence * 100).toFixed(1)}% confidence`,
                        "error"
                    );
                    playAlertSound();
                }
            });
        });
}

// Alert sound
function playAlertSound() {
    try {
        const audio = new AudioContext();
        const oscillator = audio.createOscillator();
        const gain = audio.createGain();
        oscillator.connect(gain);
        gain.connect(audio.destination);
        oscillator.frequency.value = 800;
        oscillator.type = "sine";
        gain.gain.value = 0.3;
        oscillator.start();
        setTimeout(() => {
            oscillator.stop();
            audio.close();
        }, 300);
    } catch (e) {
        // Audio not supported
    }
}

// Start listeners when page loads
document.addEventListener("DOMContentLoaded", () => {
    setupRealtimeAlerts();
    setupMissingPersonListener();
});
*/

// ========== PLACEHOLDER (works without Firebase) ==========
// The dashboard works fine without Firebase using REST API polling in app.js
// Firebase adds REAL-TIME updates (instant alerts without refresh)
//
// To enable:
// 1. Create Firebase project at console.firebase.google.com
// 2. Add web app in Project Settings
// 3. Copy config values above
// 4. Add Firebase SDK scripts to index.html
// 5. Uncomment the code above
console.log("Firebase config loaded. Real-time listeners disabled (using REST polling).");
