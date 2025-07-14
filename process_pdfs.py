import google.generativeai as genai
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time
import concurrent.futures
import json
import pypdf
import tempfile
from typing import List

# --- ANSI Color Codes for Rich Console Output ---
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_RED = '\033[91m'
C_BOLD = '\033[1m'
C_END = '\033[0m'

# --- Session Management ---
SESSION_FILE = ".pdf_process_session.json"

def format_time(seconds):
    """Converts seconds into a human-readable string 'X min, Y sec'."""
    if seconds < 60:
        return f"{seconds:.1f} sec"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)} min, {int(seconds)} sec"

# ===============================================================
# START: MODIFIED PROMPT & HTML TEMPLATE
# ===============================================================
def get_system_prompt():
    """
    Generates the initial system prompt that sets the context for the entire conversational task.
    The HTML template is simplified as structural insertion is now used.
    """
    prompt = f"""
You are a premier expert in mathematics and physics, tasked with creating a professional, textbook-quality HTML answer key from a provided worksheet. Your output must be flawless in its accuracy, pedagogy, and visual presentation. You will be given the worksheet page by page.

You have been provided with a set of files to inform your answers. Your primary knowledge source for methods, theorems, and notational conventions MUST be these provided documents. You do not need to state which PDF you used for a specific method.

**Core Task: Recreate the Worksheet with Impeccable Solutions, Page by Page**
Your most important goal is to SOLVE THE ENTIRE EXAM, remaking it with the highest possible accuracy and clarity. You must complete every single problem.

1.  **First Page:** When you receive the first page, you MUST generate the complete, initial HTML document, including the `<head>`, `<style>`, and opening `<body>` tags, using the exact template provided below. Then, solve all problems on that first page.
2.  **Subsequent Pages:** For all following pages, you will be given the new page and the HTML you've generated so far. Your task is to **APPEND** the solutions for the new page to the existing content. **DO NOT** repeat the HTML header, styles, or `<body>` tag. Only generate the new problem/solution blocks (e.g., starting from `<hr><h3>Problem X</h3>...`).
3.  **Preserve Structure:** Replicate the original structure and numbering as closely as possible within the professional HTML format.
4.  **Transcribe the Problem:** For every problem, transcribe the question exactly as it appears into a `<div class="problem-statement">`.
5.  **Provide the Solution:** Write your full, detailed, step-by-step solution in the `<div class="solution">` that immediately follows.

**Crucial Directive for 3D Geometry:**
For ANY problem involving the calculation of volume, surface area, or the visualization of surfaces, planes, or curves in 3D space, generating a Plotly.js 3D diagram is NOT optional—it is a mandatory and critical part of the solution. The purpose is to provide visual intuition alongside the analytical solution. For a problem like "Find the volume bounded by y = x^2, z = 0, and y + z = 4," a 3D plot showing these surfaces is required.

**Explanation Style: Crystal Clear, Pedagogical, and Professional**
Your explanations must be exceptionally clear, easy to follow, and structured like a perfect example from a university textbook.

*   **1. State the Strategy:** Begin with a concise summary of the method you will use.
*   **2. Justify Key Choices:** Briefly explain why the chosen method is appropriate.
*   **3. Show All Calculation Steps:** Meticulously show every step of the calculation.
*   **4. Guide the Reader:** Use short, declarative sentences to introduce each major part of the calculation.
*   **5. Define All Variables:** Any new variables or coordinate systems must be clearly defined before use.

**Visuals & Diagrams: Textbook Quality is Mandatory (D3.js for 2D, Plotly.js for 3D)**
Follow all previous instructions regarding the high quality and mandatory inclusion of D3.js and Plotly.js diagrams. The setup steps (defining scales, data arrays, layouts, etc.) remain critical.

**INDEPENDENT PROBLEM SOLVING:**
You MUST IGNORE any pre-existing answers in the provided files. Generate all solutions from scratch.

**HTML TEMPLATE (USE THIS EXACTLY FOR THE FIRST PAGE):**
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Answer Key: {{WORKSHEET_NAME}}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Source+Code+Pro:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --font-sans: 'Inter', 'Helvetica Neue', sans-serif;
            --font-mono: 'Source Code Pro', monospace;
            --text-color: #333;
            --bg-color: #fdfdfd;
            --primary-color: #4A90E2;
            --border-color: #e0e0e0;
            --accent-bg-light: #f5f7fa;
            --accent-bg-dark: #e9eef5;
            --success-color: #2E7D32;
            --success-bg: #E8F5E9;
        }}
        body {{ font-family: var(--font-sans); line-height: 1.7; margin: 0; background-color: var(--bg-color); color: var(--text-color); }}
        .container {{ max-width: 900px; margin: 40px auto; background: #fff; padding: 20px 40px; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 8px 30px rgba(0,0,0,0.05); }}
        h2, h3 {{ font-family: var(--font-sans); font-weight: 700; color: #111; border-bottom: 1px solid var(--border-color); padding-bottom: 12px; margin-top: 50px; }}
        h2 {{ text-align: center; font-size: 2.25em; margin-top: 0; margin-bottom: 20px; color: #000; }}
        h3 {{ font-size: 1.75em; color: var(--primary-color); }}
        .problem-statement {{ background-color: var(--accent-bg-light); padding: 25px; border-left: 5px solid var(--primary-color); border-radius: 8px; margin: 30px 0; font-size: 1.1em; }}
        .solution {{ padding: 10px 5px; }}
        .diagram-container {{ text-align: center; margin: 40px auto; padding: 25px; background-color: #fff; border: 1px solid var(--border-color); border-radius: 8px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.04)); }}
        .d3-chart {{ margin: auto; width: 100%; max-width: 650px; }}
        .diagram-caption {{ font-size: 0.95em; font-style: italic; color: #666; margin-top: 20px; }}
        .d3-chart text, .axis text {{ font-family: var(--font-sans); font-size: 12px; }}
        .grid .tick {{ stroke: lightgray; opacity: 0.7; }}
        .grid path {{ stroke-width: 0; }}
        hr {{ border: 0; height: 1px; background-color: var(--border-color); margin: 70px 0; }}
        code {{ font-family: var(--font-mono); background-color: var(--accent-bg-dark); padding: 3px 6px; border-radius: 4px; font-size: 0.95em; }}
        .final-answer {{ font-weight: 700; color: var(--success-color); background-color: var(--success-bg); padding: 4px 8px; border-radius: 6px; display: inline-block; border: 1px solid #a5d6a7; }}
    </style>
</head>
<body>
<div class="container">
    <h2>Answer Key for {{WORKSHEET_NAME}}</h2>
</div>
</body>
</html>
"""
    return prompt

def get_first_page_prompt(worksheet_name):
    """Generates the user prompt for the first page."""
    return (
        f"This is the first page of the exam '{worksheet_name}'. "
        "Please start solving the problems on this page. "
        "Generate the full HTML document based on the system instructions, replacing '{{WORKSHEET_NAME}}' with the actual worksheet name."
    )

def get_next_page_prompt():
    """Generates the user prompt for subsequent pages."""
    return (
        "Excellent work. Here is the next page of the exam. "
        "Below is also the HTML content you have generated so far. "
        "Please solve the problems on this new page and **APPEND** only the new HTML content for these problems. "
        "**DO NOT** repeat the <!DOCTYPE>, <head>, <style>, or opening <body> tags. Start directly with the content for the new problems (e.g., `<hr><h3>...`)."
    )

def configure_ai():
    """Loads API key from .env file and configures the Generative AI client."""
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"{C_RED}Google API Key not found. Please set the GOOGLE_API_KEY in your .env file.{C_END}")
    genai.configure(api_key=api_key)

def save_session(training_pdf_paths, uploaded_files):
    """Saves the server-side file IDs to a session file for future reuse."""
    session_data = [
        {"local_name": path.name, "server_id": file.name}
        for path, file in zip(training_pdf_paths, uploaded_files)
    ]
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(session_data, f, indent=4)
        print(f"{C_GREEN}[i] Session data updated in {SESSION_FILE}.{C_END}")
    except Exception as e:
        print(f"{C_YELLOW}[WARNING] Could not save session data: {e}{C_END}")

def upload_files_with_retry(file_paths, max_retries=3):
    uploaded_files = []
    for path in file_paths:
        for attempt in range(max_retries):
            try:
                print(f" - Uploading: {C_YELLOW}{path.name}{C_END}...")
                uploaded_file = genai.upload_file(path=path, display_name=path.name)
                uploaded_files.append(uploaded_file)
                break
            except Exception as e:
                print(f" {C_RED}[!] Attempt {attempt + 1} failed for {path.name}: {e}{C_END}")
                if attempt + 1 == max_retries:
                    print(f" {C_RED}[!] Failed to upload {path.name} after {max_retries} attempts.{C_END}")
                    raise
                time.sleep(2 ** attempt)
    return uploaded_files

def split_pdf(pdf_path: Path, output_dir: Path) -> List[Path]:
    """Splits a PDF into single-page PDFs and returns their paths."""
    print(f"[*] Splitting '{pdf_path.name}' into individual pages...")
    page_paths = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            writer = pypdf.PdfWriter()
            writer.add_page(page)
            page_output_path = output_dir / f"{pdf_path.stem}_page_{i+1}.pdf"
            with open(page_output_path, "wb") as f:
                writer.write(f)
            page_paths.append(page_output_path)
        print(f"  -> {C_GREEN}Split into {len(page_paths)} pages.{C_END}")
        return page_paths
    except Exception as e:
        print(f"  -> {C_RED}Failed to split PDF: {e}{C_END}")
        raise

def process_single_worksheet(ws_path, training_files, model):
    """
    Processes a single worksheet PDF by splitting it into pages and feeding them
    iteratively to the AI model to build a single, cohesive HTML answer key.
    """
    item_start_time = time.time()
    ws_name = ws_path.name
    output_path = ws_path.with_suffix('.key.html')

    if output_path.exists():
        return 'skipped', ws_name, "Output file already exists."

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            page_paths = split_pdf(ws_path, temp_path)

            if not page_paths:
                return 'failed', ws_name, "PDF could not be split or is empty."

            system_prompt = get_system_prompt()
            chat = model.start_chat(history=[
                {'role': 'user', 'parts': [system_prompt] + training_files},
                {'role': 'model', 'parts': ["Understood. I am ready. Please provide the first page of the worksheet."]}
            ])

            accumulated_html = ""
            total_pages = len(page_paths)

            for i, page_path in enumerate(page_paths):
                print(f"[*] Processing page {C_YELLOW}{i+1}/{total_pages}{C_END} for {C_BOLD}{ws_name}{C_END}...")

                page_file = upload_files_with_retry([page_path])[0]

                if i == 0:
                    prompt = get_first_page_prompt(ws_name)
                    message_parts = [prompt, page_file]
                else:
                    prompt = get_next_page_prompt()
                    message_parts = [prompt, accumulated_html, page_file]
                
                try:
                    response = chat.send_message(message_parts)
                    if not response.parts:
                        return 'failed', ws_name, f"Model returned an empty response on page {i+1}. This may be due to a safety filter."
                    page_html_content = response.text.strip()
                except Exception as api_error:
                    return 'failed', ws_name, f"API Error on page {i+1}: {api_error}"
                
                if page_html_content.startswith("```html"):
                    page_html_content = page_html_content.removeprefix("```html").strip()
                if page_html_content.endswith("```"):
                    page_html_content = page_html_content.removesuffix("```").strip()

                if i == 0:
                    if "<!DOCTYPE html>" not in page_html_content or "</body>" not in page_html_content:
                        return 'failed', ws_name, f"Model did not produce a valid full HTML document on the first page. Snippet: {page_html_content[:500]}"
                    accumulated_html = page_html_content.replace('{{WORKSHEET_NAME}}', ws_name)
                else:
                    insertion_point = '</div>\n</body>'
                    if insertion_point not in accumulated_html:
                        insertion_point = '</div></body>'
                        if insertion_point not in accumulated_html:
                            return 'failed', ws_name, f"Structural error: Could not find the insertion point ('</div></body>') in the HTML generated so far."
                    
                    replacement_chunk = page_html_content + '\n' + insertion_point
                    accumulated_html = accumulated_html.replace(insertion_point, replacement_chunk)

        output_path.write_text(accumulated_html, encoding='utf-8')
        
        item_duration = time.time() - item_start_time
        return 'success', ws_name, item_duration

    except Exception as e:
        return 'failed', ws_name, str(e)


def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Generate HTML answer keys for worksheets by processing PDFs page-by-page using parallel processing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--training-folder", type=str, required=True, help="Path to the folder with 'training' or 'textbook' PDFs.")
    parser.add_argument("--worksheets-folder", type=str, required=True, help="Path to the folder with 'worksheet' PDFs to be solved.")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worksheets to process in parallel.")
    args = parser.parse_args()

    try:
        configure_ai()
    except ValueError as e:
        print(e)
        return

    training_path = Path(args.training_folder)
    worksheets_path = Path(args.worksheets_folder)

    if not training_path.is_dir() or not worksheets_path.is_dir():
        print(f"{C_RED}[ERROR] One or both provided paths are not valid directories.{C_END}")
        return

    print(f"\n{C_BLUE}{C_BOLD}--- Phase 1: Managing Training Material ---{C_END}")
    
    training_pdf_paths = list(training_path.glob("*.pdf"))
    training_files = []

    if not training_pdf_paths:
        print(f"{C_YELLOW}[WARNING] No training PDFs found. The AI will use its general knowledge.{C_END}")
    else:
        session_files = {}
        if Path(SESSION_FILE).exists():
            try:
                with open(SESSION_FILE, 'r') as f:
                    session_data = json.load(f)
                    session_files = {item['local_name']: item['server_id'] for item in session_data}
            except (json.JSONDecodeError, IOError) as e:
                print(f"{C_YELLOW}[WARNING] Could not read session file '{SESSION_FILE}': {e}. Treating all files as new.{C_END}")

        current_filenames = {p.name for p in training_pdf_paths}
        session_filenames = set(session_files.keys())
        new_file_names = current_filenames - session_filenames
        existing_file_names = current_filenames.intersection(session_filenames)
        
        paths_to_upload = [p for p in training_pdf_paths if p.name in new_file_names]
        reused_files = []
        
        if existing_file_names:
            print(f"[*] Found {len(existing_file_names)} existing file(s) in session. Verifying...")
            verified_count = 0
            for name in sorted(list(existing_file_names)):
                server_id = session_files[name]
                try:
                    reused_file = genai.get_file(name=server_id)
                    reused_files.append(reused_file)
                    verified_count += 1
                except Exception:
                    paths_to_upload.append(training_path / name)
            if verified_count > 0:
                print(f"  -> {C_GREEN}Successfully verified and reused {verified_count} file(s).{C_END}")

        if paths_to_upload:
            print(f"[*] Found {len(paths_to_upload)} new or unverified file(s) to upload.")
            try:
                newly_uploaded_files = upload_files_with_retry(paths_to_upload)
                training_files = reused_files + newly_uploaded_files
                path_map = {p.name: p for p in training_pdf_paths}
                final_local_paths = [path_map[f.display_name] for f in training_files if f.display_name in path_map]
                save_session(final_local_paths, training_files)
                print(f"  -> {C_GREEN}Successfully uploaded {len(newly_uploaded_files)} and updated session.{C_END}")
            except Exception as e:
                print(f"{C_RED}[FATAL] Could not upload training files. Aborting. Error: {e}{C_END}")
                return
        else:
            training_files = reused_files

    print(f"\n{C_BLUE}{C_BOLD}--- Phase 2: Processing Worksheets in Parallel (max {args.max_workers} at a time) ---{C_END}")
    worksheet_pdf_paths = list(worksheets_path.glob("*.pdf"))
    if not worksheet_pdf_paths:
        print("No worksheet PDFs found to process.")
        return

    # Corrected Model Name as per user's image
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    processing_times, success_count, skipped_count, failed_count = [], 0, 0, 0
    total_worksheets = len(worksheet_pdf_paths)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_ws = {executor.submit(process_single_worksheet, ws_path, training_files, model): ws_path for ws_path in worksheet_pdf_paths}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_ws)):
            ws_path = future_to_ws[future]
            print(f"[{i+1}/{total_worksheets}] Finalizing result for {C_BOLD}{ws_path.name}{C_END}...")
            try:
                status, ws_name, result = future.result()
                if status == 'success':
                    success_count += 1
                    processing_times.append(result)
                    print(f"  -> {C_GREEN}[SUCCESS] {C_BOLD}{ws_name}{C_END} fully processed in {format_time(result)}.")
                elif status == 'skipped':
                    skipped_count += 1
                    print(f"  -> {C_BLUE}[SKIPPED] {C_BOLD}{ws_name}{C_END}. Reason: {result}")
                elif status == 'failed':
                    failed_count += 1
                    print(f"  -> {C_RED}[FAILED]  {C_BOLD}{ws_name}{C_END}. Reason: {result}")
            except Exception as exc:
                failed_count += 1
                print(f"  -> {C_RED}[ERROR]   {C_BOLD}{ws_path.name}{C_END} generated an unexpected exception: {exc}")

    total_duration = time.time() - script_start_time
    print(f"\n{C_BLUE}{C_BOLD}{'-'*25} All Tasks Completed {'-'*25}{C_END}")
    print(f"Summary for {total_worksheets} total worksheet(s) found:")
    print(f"  - {C_GREEN}Successfully Processed: {success_count}{C_END}")
    print(f"  - {C_BLUE}Skipped (already exists): {skipped_count}{C_END}")
    print(f"  - {C_RED}Failed: {failed_count}{C_END}")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average time per processed worksheet: {format_time(avg_time)}")
        
    print(f"{C_BOLD}Total execution time: {format_time(total_duration)}{C_END}")

if __name__ == "__main__":
    main()
