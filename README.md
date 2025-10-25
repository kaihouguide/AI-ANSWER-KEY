

AI Answer Keys

This script leverages the Google Gemini 2.5 Pro model to automate the creation of detailed, textbook-quality HTML answer keys from PDF worksheets. It uses a provided set of reference PDFs (e.g., textbooks, lecture notes) as a contextual knowledge base to inform its problem-solving process.

A key feature is its robust page-by-page processing. The script intelligently splits a multi-page worksheet and feeds it to the AI one page at a time. This allows it to build a single, cohesive answer key for large documents without losing context, while also saving progress after each page. If the script is interrupted, it can resume exactly where it left off.

### Core Features

*   **High-Quality AI Solutions**: Generates detailed, step-by-step solutions using the powerful **Gemini 2.5 Pro** model.
*   **Context-Aware**: Uses a folder of "training" PDFs as a primary knowledge source for methods and notation.
*   **Rich HTML Output**: Creates clean, professional, and readable HTML answer keys ready for web or print.
*   **Interactive Diagrams**: Automatically generates 2D diagrams with **D3.js** and mandatory 3D visualizations with **Plotly.js** for relevant problems involving geometry, forces, or data plotting.
*   **Parallel Processing**: Processes multiple worksheet files concurrently, significantly reducing total processing time.
*   **Resumable & Fault-Tolerant**: Automatically saves progress after each page. If the script is stopped or fails, it will resume from the last completed page on the next run.
*   **Multi-API Key Support**: Rotates through multiple API keys automatically if one reaches its rate limit, ensuring continuous operation.
*   **Intelligent Rate Limiting**: A smart, token-aware rate limiter prevents API errors by managing both request frequency and token count per minute.
*   **Session Caching**: Avoids the need to re-upload training files on subsequent runs, saving time and bandwidth.

---

## Setup and Usage Guide

### 1. Prerequisites

*   **Python 3.x**
*   **Google Gemini API Key**:
    *   Visit [Google AI Studio](https://aistudio.google.com/) to create a free API key.
    *   Treat your API key like a password. Do not share it publicly or commit it to version control.

### 2. Installation

**A. Download the Code**

Clone the repository or download the `process_pdfs.py` script to a new project folder.

**B. Install Required Packages**

Open a terminal in your project folder and run the following command to install the necessary Python libraries:
```bash
pip install google-generativeai python-dotenv pypdf
```
install the new api
```bash
pip install google-genai
```
**C. Configure Your API Key(s)**

In the same folder as the script, create a file named `.env`. Add your API key to this file. For best results, add multiple keys. The script will automatically rotate through them as needed.

```
# .env file content

# Primary API Key
GOOGLE_API_KEY="YOUR_PRIMARY_API_KEY_HERE"

# Additional keys for rotation (optional but recommended)
GOOGLE_API_KEY_1="YOUR_SECOND_API_KEY_HERE"
GOOGLE_API_KEY_2="YOUR_THIRD_API_KEY_HERE"
# ...and so on.
```

### 3. Folder Structure

Arrange your files in the following structure before running the script:

```
your-project-folder/
├── training_materials/       <-- Place reference PDFs (textbooks, notes) here
│   ├── textbook_chapter_1.pdf
│   └── class_notes.pdf
├── worksheets_to_solve/      <-- Place worksheet PDFs to be solved here
│   ├── homework_1.pdf
│   └── final_exam_review.pdf
├── process_pdfs.py           <-- The main Python script
└── .env                      <-- Your environment file with API key(s)
```

### 4. Run the Script

Execute the script from your terminal, pointing it to your training and worksheet folders. The script will show its progress in the terminal.

**The final `.key.html` answer keys will be generated in the same folder as your worksheets (`worksheets_to_solve/` in the example above).**

**Basic Command:**
```bash
python process_pdfs.py --training-folder training_materials --worksheets-folder worksheets_to_solve
```

**Command-Line Arguments:**

*   `--training-folder` **(Required)**: The path to the folder containing your reference PDF documents.
*   `--worksheets-folder` **(Required)**: The path to the folder containing the worksheet PDFs you want to solve.
*   `--max-workers` (Optional): The number of worksheet files to process in parallel. The default is `4`. It is recommended to start with a lower number.

**Example with all arguments:**
```bash
python process_pdfs.py --training-folder "./training_materials" --worksheets-folder "./worksheets_to_solve" --max-workers 1
```

### Important Notes on the New Script:

*   **Managing Concurrency**: The script is built for concurrency, but using fewer workers is often better to avoid hitting API rate limits, especially if you have a limited number of API keys. Processing one file at a time (`--max-workers 1`) is the safest way to ensure completion without interruption.
*   **Automatic Resuming**: The script creates a `.progress.json` file for any worksheet it's actively processing. If you stop and restart the script, it will automatically detect these files and resume from the last successfully completed page, so no work is lost. These progress files are deleted automatically upon successful completion of a worksheet.
