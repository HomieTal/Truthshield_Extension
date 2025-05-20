document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    if (!tabs.length || !tabContents.length) {
        console.error('Tabs or tab contents not found in DOM');
        return;
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const tabContent = document.getElementById(tab.dataset.tab);
            if (tabContent) {
                tabContent.classList.add('active');
            } else {
                console.error(`Tab content for ${tab.dataset.tab} not found`);
            }
        });
    });

    // Load settings
    chrome.storage.local.get(['autoAiAnalysis', 'minConfidence', 'apiKey'], (data) => {
        const autoAiCheckbox = document.getElementById('auto-ai-analysis');
        const minConfidenceInput = document.getElementById('min-confidence');
        const confidenceValue = document.getElementById('confidence-value');
        const apiKeyInput = document.getElementById('api-key');

        if (autoAiCheckbox && minConfidenceInput && confidenceValue && apiKeyInput) {
            autoAiCheckbox.checked = data.autoAiAnalysis !== false;
            minConfidenceInput.value = data.minConfidence || 75;
            confidenceValue.textContent = data.minConfidence || 75;
            apiKeyInput.value = data.apiKey || '';
        } else {
            console.error('Settings elements not found in DOM');
        }
    });

    // Update confidence value display
    const minConfidenceInput = document.getElementById('min-confidence');
    if (minConfidenceInput) {
        minConfidenceInput.addEventListener('input', (e) => {
            const confidenceValue = document.getElementById('confidence-value');
            if (confidenceValue) {
                confidenceValue.textContent = e.target.value;
            }
        });
    }

    // Save settings
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    if (saveSettingsBtn) {
        saveSettingsBtn.addEventListener('click', () => {
            const settings = {
                autoAiAnalysis: document.getElementById('auto-ai-analysis')?.checked,
                minConfidence: parseInt(document.getElementById('min-confidence')?.value) || 75,
                apiKey: document.getElementById('api-key')?.value || ''
            };
            chrome.storage.local.set(settings, () => {
                const savedMessage = document.getElementById('settings-saved');
                if (savedMessage) {
                    savedMessage.classList.remove('hidden');
                    setTimeout(() => savedMessage.classList.add('hidden'), 2000);
                }
            });
        });
    }

    // Analyze text function
    async function analyzeText(newsText) {
        if (!newsText) {
            alert('Please enter or select news text to analyze.');
            return;
        }

        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        if (!loading || !results) {
            console.error('Loading or results element not found');
            return;
        }

        loading.classList.remove('hidden');
        results.classList.add('hidden');

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${document.getElementById('api-key')?.value || ''}`
                },
                body: JSON.stringify({ 
                    news_text: newsText, 
                    auto_ai_analysis: document.getElementById('auto-ai-analysis')?.checked 
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            loading.classList.add('hidden');
            results.classList.remove('hidden');

            // Update results
            const verdictResult = document.getElementById('verdict-result');
            const confidence = document.getElementById('confidence');
            const mlResult = document.getElementById('ml-result');
            const aiResult = document.getElementById('ai-result');
            const analysisText = document.getElementById('analysis-text');

            if (verdictResult && confidence && mlResult && aiResult && analysisText) {
                verdictResult.textContent = data.verdict;
                verdictResult.className = `verdict ${data.verdict === 'Real News' ? 'real' : 'fake'}`;
                confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                mlResult.textContent = `${data.rf_result} (${(data.rf_confidence * 100).toFixed(2)}%) / ${data.lr_result} (${(data.lr_confidence * 100).toFixed(2)}%)`;
                aiResult.textContent = data.ai_result || 'Not performed';
                analysisText.innerHTML = data.analysis_details.replace(/\n/g, '<br>');
            } else {
                console.error('Result display elements not found');
            }

            // Save to history
            chrome.storage.local.get(['history'], (result) => {
                const history = result.history || [];
                history.unshift({
                    text: newsText,
                    verdict: data.verdict,
                    confidence: data.confidence,
                    timestamp: new Date().toLocaleString()
                });
                chrome.storage.local.set({ history: history.slice(0, 50) }, updateHistory);
            });
        } catch (error) {
            loading.classList.add('hidden');
            alert(`Error analyzing text: ${error.message}`);
            console.error('Analysis error:', error);
        }
    }

    // Analyze button click
    const analyzeBtn = document.getElementById('analyze-btn');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', () => {
            const newsText = document.getElementById('news-text')?.value.trim();
            analyzeText(newsText);
        });
    }

    // Ask question
    const askQuestionBtn = document.getElementById('ask-question-btn');
    if (askQuestionBtn) {
        askQuestionBtn.addEventListener('click', () => {
            const questionSection = document.getElementById('question-section');
            if (questionSection) {
                questionSection.classList.toggle('hidden');
            }
        });
    }

    const submitQuestionBtn = document.getElementById('submit-question-btn');
    if (submitQuestionBtn) {
        submitQuestionBtn.addEventListener('click', async () => {
            const question = document.getElementById('question-text')?.value.trim();
            const newsText = document.getElementById('news-text')?.value.trim();
            if (!question || !newsText) {
                alert('Please enter a question and ensure news text is provided.');
                return;
            }

            try {
                const response = await fetch('http://localhost:5000/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${document.getElementById('api-key')?.value || ''}`
                    },
                    body: JSON.stringify({ news_text: newsText, question })
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const data = await response.json();
                const questionAnswer = document.getElementById('question-answer');
                if (questionAnswer) {
                    questionAnswer.textContent = data.answer;
                    questionAnswer.classList.remove('hidden');
                }
            } catch (error) {
                alert(`Error submitting question: ${error.message}`);
                console.error('Question submission error:', error);
            }
        });
    }

    // Save result
    const saveResultBtn = document.getElementById('save-result-btn');
    if (saveResultBtn) {
        saveResultBtn.addEventListener('click', () => {
            const newsText = document.getElementById('news-text')?.value.trim();
            const verdict = document.getElementById('verdict-result')?.textContent;
            const confidence = document.getElementById('confidence')?.textContent;

            if (newsText && verdict && confidence) {
                chrome.storage.local.get(['history'], (result) => {
                    const history = result.history || [];
                    history.unshift({
                        text: newsText,
                        verdict,
                        confidence,
                        timestamp: new Date().toLocaleString()
                    });
                    chrome.storage.local.set({ history: history.slice(0, 50) }, updateHistory);
                });
            }
        });
    }

    // Update history display
    function updateHistory() {
        chrome.storage.local.get(['history'], (result) => {
            const historyList = document.getElementById('history-list');
            if (!historyList) {
                console.error('History list element not found');
                return;
            }
            historyList.innerHTML = '';
            const history = (result.history || []).slice(0, 5); // Limit to last 5 entries

            if (history.length === 0) {
                historyList.innerHTML = '<p class="empty-message">No saved analyses yet.</p>';
                return;
            }

            history.forEach(item => {
                const div = document.createElement('div');
                div.className = 'history-item';
                div.innerHTML = `
                    <p><strong>${item.verdict}</strong> (${item.confidence})</p>
                    <p>${item.text.substring(0, 100)}${item.text.length > 100 ? '...' : ''}</p>
                    <p class="timestamp">${item.timestamp}</p>
                `;
                historyList.appendChild(div);
            });
        });
    }

    // About modal
    const aboutLink = document.getElementById('about-link');
    if (aboutLink) {
        aboutLink.addEventListener('click', (e) => {
            e.preventDefault();
            const aboutModal = document.getElementById('about-modal');
            if (aboutModal) {
                aboutModal.classList.remove('hidden');
            }
        });
    }

    const closeModal = document.querySelector('.close-modal');
    if (closeModal) {
        closeModal.addEventListener('click', () => {
            const aboutModal = document.getElementById('about-modal');
            if (aboutModal) {
                aboutModal.classList.add('hidden');
            }
        });
    }

    // Context menu integration
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
        if (request.action === 'analyzeText' && request.text) {
            const newsText = request.text.trim();
            const newsTextArea = document.getElementById('news-text');
            const analyzeTabBtn = document.querySelector('.tab-btn[data-tab="analyze"]');
            const analyzeTabContent = document.getElementById('analyze');

            if (!newsTextArea || !analyzeTabBtn || !analyzeTabContent) {
                console.error('Required DOM elements for context menu action not found');
                alert('Error: Extension UI elements not found.');
                sendResponse({ status: 'error', message: 'UI elements not found' });
                return;
            }

            // Load selected text into textarea and trigger analysis
            newsTextArea.value = newsText;
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            analyzeTabBtn.classList.add('active');
            analyzeTabContent.classList.add('active');

            // Ensure popup is visible and focused
            chrome.runtime.sendMessage({ action: 'openPopup' });

            // Trigger analysis
            analyzeText(newsText);
            sendResponse({ status: 'success', message: 'Text loaded and analysis triggered' });
        } else {
            console.error('Invalid or missing text in context menu request:', request);
            alert('Error: No text selected for analysis.');
            sendResponse({ status: 'error', message: 'No text selected' });
        }
    });

    // Initial history load
    updateHistory();
});