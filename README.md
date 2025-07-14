

# PDF Answer Key Generator

This script automates the creation of detailed, textbook-quality HTML answer keys from PDF worksheets. It leverages the Google Gemini 2.5 Pro model to analyze and solve problems from worksheets, using a provided set of reference PDF documents (like textbooks or notes) as a knowledge base.

The key feature of this script is its **page-by-page processing**. It splits a worksheet into individual pages, feeding them to the AI one by one. This iterative approach allows the model to build a single, cohesive answer key for long and complex documents without losing context.

## Features

*   **AI-Powered Solutions**: Utilizes Google's Gemini 2.5 Pro to generate step-by-step solutions.
*   **Iterative Page Processing**: Intelligently splits worksheet PDFs and processes them page by page to build a single, complete HTML file, ensuring scalability for large documents.
*   **Context-Aware**: Uses a folder of "training" PDFs as a primary knowledge source for the AI.
*   **Rich HTML Output**: Creates clean, professional, and easy-to-read HTML answer keys using a modern, readable template.
*   **Interactive Diagrams**: Automatically generates 2D diagrams with D3.js and mandatory 3D visualizations with Plotly.js for relevant problems.
*   **Parallel Processing**: Processes multiple worksheet files concurrently to save significant time.
*   **Session Management**: Caches uploaded training files on Google's servers to avoid re-uploading them every time you run the script, saving bandwidth and time.

---

## Installation & Usage Guide

### 1. Obtain Your Google API Key
You need a Gemini API key to use this script. You can get one for free from Google AI Studio.

*   **Step 1:** Go to the [Google AI Studio](https://aistudio.google.com/) website.
*   **Step 2:** Sign in with your Google account.
*   **Step 3:** Click on the **"Get API key"** button, usually found in the top-left corner.
*   **Step 4:** Click **"Create API key"**. This will likely prompt you to create a new Google Cloud project if you don't have one already. The process is quick and guided.
*   **Step 5:** Once the key is generated, click the copy icon to copy it to your clipboard.

> **Important:** Treat your API key like a password! Do not share it publicly or commit it to a Git repository.

### 2. Installation
**A. Download The Repository**

You can either clone the repository using git or download it as a ZIP file.
```bash
git clone https://github.com/your-username/your-repository-name.git
```
Or, click the `Code` button on the GitHub page and select `Download ZIP`. Unzip the file to your desired location.

**B. Install the Required Python Packages**

Open your terminal or command prompt and run the following command to install the necessary libraries. This project requires `pypdf` for splitting the worksheet PDFs.
```bash
pip install google-generativeai python-dotenv pypdf
```

**C. Add Your API Key**

In the project folder, create or edit the file named `.env`. Add the API key you copied from Google AI Studio.

```
# .env file content
GOOGLE_API_KEY=PASTE_YOUR_KEY_HERE
```

### 3. Folder Structure
Before running the script, place your files into the correct folders. The script will look for these specific folder names.

```
your-project-folder/
├── training_materials/       <-- Place your reference PDFs (textbooks, notes) here
│   ├── textbook1.pdf
│   └── chapter5_notes.pdf
├── worksheets_to_solve/      <-- Place the worksheet PDFs you want to process here
│   ├── homework1.pdf
│   └── quiz3.pdf
├── process_pdfs.py           <-- The main Python script
└── .env                      <-- Your environment file with the API key
```

### 4. Run the Script
The script is run from the command line and accepts arguments to specify the folders and the number of parallel workers.

Open your terminal or command prompt and navigate to the project folder. Use the following command:

```bash
python process_pdfs.py --training-folder "./training_materials" --worksheets-folder "./worksheets_to_solve"
```

**Command-Line Arguments:**

*   `--training-folder` (Required): Path to the folder with your reference PDFs.
*   `--worksheets-folder` (Required): Path to the folder with the worksheets you want to solve.
*   `--max-workers` (Optional): The maximum number of worksheets to process in parallel. Defaults to `5`. You can adjust this based on your computer's capability.

**Example with all arguments:**
```bash
python process_pdfs.py --training-folder "./training_materials" --worksheets-folder "./worksheets_to_solve" --max-workers 3
```

A command window will open and show the progress. The script will first handle uploading any new training files and then begin processing each worksheet. The final `.key.html` answer keys will appear in the `worksheets_to_solve` folder once they are complete.
