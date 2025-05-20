chrome.runtime.onInstalled.addListener(() => {
    // Create context menu item
    chrome.contextMenus.create({
        id: 'analyzeWithTruthShield',
        title: 'Analyze with TruthShield',
        contexts: ['selection'] // Only show when text is selected
    });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
    if (info.menuItemId === 'analyzeWithTruthShield' && info.selectionText) {
        // Send message to popup with selected text
        chrome.runtime.sendMessage({
            action: 'analyzeText',
            text: info.selectionText
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error('Error sending message:', chrome.runtime.lastError);
                return;
            }
            console.log('Response from popup:', response);
        });

        // Open the popup if not already open
        chrome.action.openPopup();
    }
});

// Listen for messages to open the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'openPopup') {
        chrome.action.openPopup();
        sendResponse({ status: 'popup opened' });
    }
});