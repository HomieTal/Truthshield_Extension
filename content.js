// This is the content script that runs on web pages

console.log("TruthShield content script loaded");

// Listen for messages from the background script or popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    console.log("Content script received message:", message);
    
    if (message.action === "getSelectedText") {
        const selectedText = window.getSelection().toString();
        sendResponse({text: selectedText});
    }
    
    return true; // Will respond asynchronously
});

// Track selected text
let lastSelectedText = "";

document.addEventListener("mouseup", () => {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText && selectedText.length > 0) {
        lastSelectedText = selectedText;
    }
});

// Add custom context menu via right-click
document.addEventListener("contextmenu", (event) => {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText && selectedText.length > 0) {
        lastSelectedText = selectedText;
    }
});