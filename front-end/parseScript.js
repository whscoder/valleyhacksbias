async function parseText() {
    const userInput = document.getElementById('userInput').value;
    const outputArea = document.getElementById('raw-output');
    const resultsSection = document.getElementById('results-area');

    if (!userInput) return alert("Please enter text or a URL");

    resultsSection.style.display = 'block';
    outputArea.innerText = "Processing... Please wait.";

    let finalRawText = "";

    // (Extraction logic remains the same)
    if (userInput.startsWith("http")) {
        try {
            const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(userInput)}`;
            const response = await fetch(proxyUrl);
            const data = await response.json();
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = data.contents;
            const paragraphs = tempDiv.querySelectorAll('p');
            finalRawText = Array.from(paragraphs).map(p => p.innerText).join('\n\n');
        } catch (error) {
            finalRawText = "Error fetching the link.";
        }
    } else {
        finalRawText = userInput;
    }

    outputArea.innerText = finalRawText; // Troubleshooting area

    // --- NEW: SEND DATA TO BACKEND ---
    const backendUrl = "http://127.0.0.1:8000/analyze"; // Your FastAPI URL
    
    // Create the JSON object
    const dataToSend = {
        raw_text: finalRawText,
        source: userInput.startsWith("http") ? "url" : "manual_paste"
    };

    try {
        const backendResponse = await fetch(backendUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(dataToSend) // This "converts" the object to a string
        });

        const result = await backendResponse.json();
        console.log("Response from Python:", result);
    } catch (err) {
        console.error("Backend error:", err);
    }
}