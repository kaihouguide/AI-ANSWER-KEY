**Obtain Your Google API Key:**
    You need a Gemini API key to use this script. You can get one for free from Google AI Studio.

    *   **Step 1:** Go to the [Google AI Studio](https://aistudio.google.com/) website.
    *   **Step 2:** Sign in with your Google account.
    *   **Step 3:** Click on the **"Get API key"** button, usually found in the top-left corner.
    *   **Step 4:** Click **"Create API key"**. This will likely prompt you to create a new Google Cloud project if you don't have one already. The process is quick and guided.
    *   **Step 5:** Once the key is generated, click the copy icon to copy it to your clipboard.

    > **Important:** Treat your API key like a password! Do not share it publicly or commit it to your Git repository.

### Installation

1.  **Download The Repo:**
  

2.  **Install the Required Python Packages:**
    Open your terminal or command prompt and run the following command to install the necessary libraries directly.
    ```bash
    pip install google-generativeai python-dotenv
    ```
    *Note: The `pathlib` library is part of the standard Python library (since 3.4+) and does not require a separate installation.*

3.  **In The `.env` file** replace `YOUR_API_KEY` with your actual key:
  

### Folder Structure

Before running the script, organize your files as follows:

```
your-repository-name/
├── training_materials/       <-- Place your reference PDFs (textbooks, notes) here
│   ├── textbook1.pdf
│   └── chapter5_notes.pdf
├── worksheets_to_solve/      <-- Place the worksheet PDFs you want to process here
│   ├── homework1.pdf
│   └── quiz3.pdf
├── process_pdfs.py           <-- The main script
└── .env                      <-- Your environment file with the API key
```
### Run generate.bat and wait 
