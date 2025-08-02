# PDF Answer Key Generator

This script uses the Google Gemini 2.5 Pro model to automate the creation of detailed, textbook-quality HTML answer keys from PDF worksheets. It uses a provided set of reference PDFs (e.g., textbooks, notes) as a knowledge base to solve problems.

A key feature is its page-by-page processing. The script splits a multi-page worksheet and feeds it to the AI one page at a time, allowing it to build a single, cohesive answer key for large documents without losing context.

### Core Features

*   **AI-Powered Solutions**: Generates step-by-step solutions using Gemini 2.5 Pro.
*   **Context-Aware**: Uses a folder of "training" PDFs as a primary knowledge source.
*   **Rich HTML Output**: Creates clean, professional, and readable HTML answer keys.
*   **Interactive Diagrams**: Automatically generates 2D diagrams with D3.js and mandatory 3D visualizations with Plotly.js for relevant problems.
*   **Parallel Processing**: Processes multiple worksheet files concurrently to save time.
*   **Session Caching**: Avoids re-uploading training files on subsequent runs.
*   **Resumable & Multi-Key**: Supports interruption and can use multiple API keys.

---

## Setup and Usage Guide

### 1. Prerequisites

*   **Python 3.x**
*   **Google Gemini API Key**:
    *   Visit [Google AI Studio](https://aistudio.google.com/) to create a free API key.
    *   Treat your API key like a password. Do not share it publicly.

### 2. Installation

**A. Download the Code**

Clone the repository or download the ZIP file and unzip it.

**B. Install Required Packages**

Open a terminal and run the following command to install the necessary libraries:
```bash
pip install google-generativeai python-dotenv pypdf
```

**C. Configure Your API Key**

In the project folder, create a file named `.env` and add your API key to it:
```
# .env file content
GOOGLE_API_KEY="KEY HERE"
```
You can add multiple keys for rotation (`GOOGLE_API_KEY_1`, `GOOGLE_API_KEY_2`, etc.) which is recommended.

### 3. Folder Structure

Arrange your files in the following structure before running the script:

```
your-project-folder/
├── training_materials/       <-- Place reference PDFs (textbooks, notes) here
├── worksheets_to_solve/      <-- Place worksheet PDFs to be solved here
├── process_pdfs.py           <-- The main Python script
└── .env                      <-- Your environment file with the API key
```

### 4. Run the Script

Execute the script from your terminal, pointing it to your material and worksheet folders.

**Basic Command:**
```bash
python process_pdfs.py --training-folder training_data --worksheets-folder my_worksheets --max-workers 1
```

**Command-Line Arguments:**

*   `--training-folder` **(Required)**: Path to the folder with reference PDFs.
*   `--worksheets-folder` **(Required)**: Path to the folder with worksheets to be solved.
*   `--max-workers` (Optional): Number of worksheets to process in parallel. Defaults to `4`.

**Example with all arguments:**
```bash
python process_pdfs.py --training-folder "./training_materials" --worksheets-folder "./worksheets_to_solve" --max-workers 4
```
### A NOTE ON THE NEW SCRIPT:
* The script is now built for concurrency. Using more workers is better, especially with multiple API keys. It still takes around 8 mins an answer key, but you can do many at once.
* The script will show its progress in the terminal. The final `.key.html` answer keys will be saved in the `worksheets_to_solve` folder. It now automatically saves progress and will resume if you stop and start it again.
