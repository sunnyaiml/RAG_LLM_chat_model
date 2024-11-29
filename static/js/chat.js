document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("fileInput");
    const dropArea = document.getElementById("dropArea");
    const trainButton = document.getElementById("trainBtn");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const sendButton = document.getElementById("sendBtn");
    const messageInput = document.getElementById("messageInput");
    const chatBody = document.getElementById("chatBody");
    const menuToggle = document.querySelector(".menu-toggle");
    const sidebar = document.querySelector(".sidebar");

    const API_BASE_URL = "http://127.0.0.1:5000"; // Backend URL

    // Handle file drag and drop
    ["dragover", "dragenter"].forEach(eventName => {
        dropArea.addEventListener(eventName, (event) => {
            event.preventDefault();
            dropArea.classList.add("dragging");
        });
    });

    ["dragleave", "dragend"].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove("dragging");
        });
    });

    dropArea.addEventListener("drop", (event) => {
        event.preventDefault();
        dropArea.classList.remove("dragging");
        const files = event.dataTransfer.files;
        uploadFiles(files);
    });

    dropArea.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (event) => {
        const files = event.target.files;
        uploadFiles(files);
    });

    // Upload files to the server
    async function uploadFiles(files) {
        if (files.length === 0) {
            alert("No file selected.");
            return;
        }

        const formData = new FormData();
        formData.append("file", files[0]);

        try {
            const response = await fetch(`${API_BASE_URL}/upload`, {
                method: "POST",
                body: formData,
                mode: "cors",
            });

            if (!response.ok) {
                throw new Error("File upload failed");
            }

            const data = await response.json();
            console.log("Upload Response:", data);
            alert(data.message || "File uploaded successfully");
        } catch (error) {
            console.error("Error uploading file:", error);
            alert("An error occurred while uploading the file.");
        }
    }

    // Train the model
    trainButton.addEventListener("click", async () => {
        trainButton.disabled = true;
        loadingIndicator.style.display = "block";

        try {
            const response = await fetch(`${API_BASE_URL}/trained`, {
                method: "POST",
                mode: "cors",
            });

            if (!response.ok) {
                throw new Error(`Training failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Train Response:", data);
            alert(data.message || "Model trained successfully");
        } catch (error) {
            console.error("Error during training:", error);
            alert("An error occurred during training.");
        } finally {
            loadingIndicator.style.display = "none";
            trainButton.disabled = false;
        }
    });

    // Ask a question
    sendButton.addEventListener("click", async () => {
        const question = messageInput.value.trim();
        if (!question) {
            alert("Please type a question.");
            return;
        }

        // Display the user's question in the chat
        const messageElement = document.createElement("div");
        messageElement.className = "message sent";
        messageElement.textContent = question;
        chatBody.appendChild(messageElement);
        messageInput.value = "";

        try {
            const response = await fetch(`${API_BASE_URL}/ask`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question }),
                mode: "cors",
            });

            if (!response.ok) {
                throw new Error(`Failed to get an answer: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            console.log("Ask Response:", data);

            // Display the answer in the chat body
            const answerElement = document.createElement("div");
            answerElement.className = "message received";
            answerElement.textContent = data.answer || "Sorry, no answer found.";
            chatBody.appendChild(answerElement);

            // Scroll to the bottom of the chat
            chatBody.scrollTop = chatBody.scrollHeight;
        } catch (error) {
            console.error("Error getting answer:", error);

            // Display an error message if something goes wrong
            const errorElement = document.createElement("div");
            errorElement.className = "message error";
            errorElement.textContent = "Error: Unable to process your request, please try again.";
            chatBody.appendChild(errorElement);

            // Scroll to the bottom of the chat
            chatBody.scrollTop = chatBody.scrollHeight;
        }
    });

    // Optional: Allow sending question with Enter key
    messageInput.addEventListener("keypress", (event) => {
        if (event.key === "Enter") {
            sendButton.click();
        }
    });

    // Menu toggle functionality
    menuToggle.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });
});
