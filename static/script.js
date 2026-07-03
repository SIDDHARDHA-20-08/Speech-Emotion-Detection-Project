async function PredictEmotion() {
    const fileInput = document.getElementById("audioFile");
    const file = fileInput.files[0];
    const emotionText = document.getElementById("emotion");
    const errorText = document.getElementById("error");

    if (!file) {
        alert("Please select an audio file first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    emotionText.textContent = "Processing...";
    errorText.textContent = "";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            emotionText.textContent = "Detected Emotion: " + data.emotion;
        } else {
            errorText.textContent = "Error: " + (data.error || "Unknown error");
        }
    } catch (err) {
        errorText.textContent = "Error: " + err.message;
    }
}




