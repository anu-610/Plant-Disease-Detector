# Crop Protector 🌿

## Introduction

Crop Protector is a web-based, multimodal AI tool designed to help farmers and gardeners easily diagnose plant diseases. By leveraging a powerful computer vision model, users can upload an image of a plant leaf and receive an instant, accurate diagnosis. The application then uses the Gemini generative AI to provide a clear, actionable solution, making expert agricultural advice accessible to everyone.

This project was developed for the IIT Mandi iHub & HCI Foundation Multimodal AI Hackathon.

## How to Run Locally

Follow these steps to set up and run the application on your local machine.

### Prerequisites

* Python3 (tested version 3.12)
* Git

---

### Step 1: Clone the Repository

Open your terminal and clone this repository to your local machine.

```bash
git clone https://github.com/anu-610/Plant-Disease-Detector.git
cd Plant-Disease-Detector
```

### Step 2: Create and Activate a Virtual Environment
It is better to use a virtual environment

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Run the application
```bash
python app.py
```

<b>Once the server is running, open web browser and navigate to https://127.0.0.1:5001</b>
