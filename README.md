
# PDF Answer Key Generator

This script automates the creation of detailed, textbook-quality HTML answer keys for PDF worksheets. It leverages the Google Gemini 2.5 Pro model to analyze and solve problems from worksheets, using a provided set of reference PDF documents (like textbooks or notes) as a knowledge base.

## Features

*   **AI-Powered Solutions**: Utilizes Google's Gemini 2.5 Pro to generate step-by-step solutions.
*   **Context-Aware**: Uses a folder of "training" PDFs as a primary knowledge source.
*   **Rich HTML Output**: Creates clean, professional, and easy-to-read HTML answer keys.
*   **Interactive Diagrams**: Generates 2D diagrams with D3.js and 3D visualizations with Plotly.js.
*   **Parallel Processing**: Processes multiple worksheets concurrently to save time.
*   **Session Management**: Caches uploaded training files to avoid re-uploading them.

---

## Installation & Usage Guide

### 1. Obtain Your Google API Key
You need a Gemini API key to use this script. You can get one for free from Google AI Studio.

*   **Step 1:** Go to the [Google AI Studio](https://aistudio.google.com/) website.
*   **Step 2:** Sign in with your Google account.
*   **Step 3:** Click on the **"Get API key"** button, usually found in the top-left corner.
*   **Step 4:** Click **"Create API key"**. This will likely prompt you to create a new Google Cloud project if you don't have one already. The process is quick and guided.
*   **Step 5:** Once the key is generated, click the copy icon to copy it to your clipboard.

> **Important:** Treat your API key like a password! Do not share it publicly or commit it to your Git repository.

### 2. Installation
**A. Download The Repo**

You can either clone the repository using git or download it as a ZIP file.
```bash
git clone https://github.com/your-username/your-repository-name.git
```
Or, click the `Code` button on the GitHub page and select `Download ZIP`. Unzip the file to your desired location.

**B. Install the Required Python Packages**

Open your terminal or command prompt and run the following command to install the necessary libraries directly.
```bash
pip install google-generativeai python-dotenv
```
*Note: The `pathlib` library is part of the standard Python library (since Python 3.4) and does not require a separate installation.*

**C. Add Your API Key**

In the folder you just downloaded, find the `.env` file. Open it and replace `YOUR_API_KEY` with the actual key you copied from Google AI Studio.

```
# .env file content
GOOGLE_API_KEY=PASTE_YOUR_KEY_HERE
```

### 3. Folder Structure
Before running the script, place your files into the correct folders as shown below.

```
your-repository-name/
├── training_materials/       <-- Place your reference PDFs (textbooks, notes) here
│   ├── textbook1.pdf
│   └── chapter5_notes.pdf
├── worksheets_to_solve/      <-- Place the worksheet PDFs you want to process here
│   ├── homework1.pdf
│   └── quiz3.pdf
├── process_pdfs.py           <-- The main script
├── generate.bat              <-- The script runner
└── .env                      <-- Your environment file with the API key
```

### 4. Run the Script
Simply run the **`generate.bat`** file by double-clicking it.

A command window will open and show the progress. Please be patient as the script uploads the training files and processes each worksheet. The final `.key.html` answer keys will appear in the `worksheets_to_solve` folder once they are complete.
