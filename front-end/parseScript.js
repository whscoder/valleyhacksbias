console.log("IT WORKS!")
async function parseText() {
    const userInput = document.getElementById('userInput').value;
    const outputArea = document.getElementById('parsed-output');
    const resultsSection = document.getElementById('results-area');

    if (!userInput) return alert("Please enter text or a URL");

    // Show the troubleshooting area
    resultsSection.style.display = 'block';
    outputArea.innerText = "Processing... Please wait.";

    // VARIABLE: This will hold our final raw text
    let finalRawText = "";

    if (userInput.startsWith("http")) {
        // --- CASE 1: IT'S A LINK ---
        try {
            // We use a proxy to bypass security blocks (CORS)
            const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(userInput)}`;
            const response = await fetch(proxyUrl);
            const data = await response.json();
            
            // This creates a temporary element to turn HTML into readable text
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = data.contents;
            
            // Extract text from paragraphs only (to avoid navbars/ads)
            const paragraphs = tempDiv.querySelectorAll('p');
            finalRawText = Array.from(paragraphs).map(p => p.innerText).join('\n\n');

            if (!finalRawText) finalRawText = "Could not extract text. Site might be protected.";

        } catch (error) {
            finalRawText = "Error fetching the link. Check your connection or the URL.";
        }
    } else {
        // --- CASE 2: IT'S RAW TEXT ---
        finalRawText = userInput;
    }

    // --- DISPLAY FOR TROUBLESHOOTING ---
    // The variable 'finalRawText' now holds all your data
    outputArea.innerText = finalRawText;
    console.log("Variable 'finalRawText' value:", finalRawText);
}