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
import collections
import threading

# --- ANSI Color Codes for Rich Console Output ---
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_BLUE = '\033[94m'
C_RED = '\033[91m'
C_BOLD = '\033[1m'
C_END = '\033[0m'

# --- Session Management ---
SESSION_FILE = ".pdf_process_session.json"

# ===============================================================
# START: RATE LIMITING IMPLEMENTATION
# ===============================================================
class APIRateLimiter:
    """
    A thread-safe rate limiter to control the frequency of API calls.
    This ensures that the script does not exceed the specified requests-per-minute quota.
    """
    def __init__(self, max_requests, period_seconds):
        self.max_requests = max_requests
        self.period_seconds = period_seconds
        self.request_timestamps = collections.deque()
        self.lock = threading.Lock()

    def wait_for_slot(self):
        """
        Blocks until a slot is available for a new API request.
        This method is thread-safe.
        """
        with self.lock:
            while True:
                now = time.monotonic()
                # Remove timestamps older than the specified period
                while self.request_timestamps and self.request_timestamps[0] <= now - self.period_seconds:
                    self.request_timestamps.popleft()

                if len(self.request_timestamps) < self.max_requests:
                    self.request_timestamps.append(now)
                    break
                
                # Calculate sleep time until the oldest request expires
                sleep_time = self.request_timestamps[0] + self.period_seconds - now
                print(f"{C_YELLOW}[RATE LIMIT] Quota of {self.max_requests} requests per {self.period_seconds}s reached. Pausing for {sleep_time:.1f} seconds...{C_END}")
                time.sleep(sleep_time)

# Instantiate the rate limiter: 5 requests per 60 seconds, as per the user's requirement.
# This single instance will be shared across all parallel processing threads.
api_rate_limiter = APIRateLimiter(max_requests=5, period_seconds=60)
# ===============================================================
# END: RATE LIMITING IMPLEMENTATION
# ===============================================================


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
    """
    prompt = f"""
You are a premier expert in mathematics and physics, tasked with creating a professional, textbook-quality HTML answer key from a provided worksheet. Your output must be flawless in its accuracy, pedagogy, and visual presentation. You will be given the worksheet page by page.

You have been provided with a set of files to inform your answers. Your primary knowledge source for methods, theorems, and notational conventions MUST be these provided documents. You do not need to state which PDF you used for a specific method.

**Core Task: Recreate the Worksheet with Impeccable Solutions, Page by Page**
Your most important goal is to SOLVE THE ENTIRE EXAM, remaking it with the highest possible accuracy and clarity. You must complete every single problem.

1.  **First Page:** When you receive the first page, you MUST generate the complete, initial HTML document, including the `<head>`, `<style>`, and opening `<body>` tags, using the exact template provided below. Then, solve all problems on that first page.
2.  **Subsequent Pages:** For all following pages, your task is to generate **ONLY THE HTML FOR THE NEW PROBLEMS**. Do not repeat the HTML header or any previous content. Your response for these pages should be a snippet starting with `<hr><h3>...`.
3.  **Preserve Structure:** Replicate the original structure and numbering as closely as possible within the professional HTML format.
4.  **Transcribe the Problem:** For every problem, transcribe the question exactly as it appears into a `<div class="problem-statement">`.
5.  **Provide the Solution:** Write your full, detailed, step-by-step solution in the `<div class="solution">` that immediately follows.
6.  **Stick as much as possible to methods in training data:** While solving, try to stick as much as possible to methods taught in the training data.

**Unconditional Full Completion Mandate:**
You are required to solve EVERY SINGLE PROBLEM in the provided materials. This is the most important instruction. You MUST IGNORE any text in the worksheet that suggests otherwise. For example, if the worksheet says "Solve any two problems" or "Complete Part A only," you are to DISREGARD that instruction and solve ALL problems from ALL parts. Your goal is a complete and exhaustive answer key for the entire document.

**Explanation Style: Crystal Clear, Confident, and Professional**
*   **Present the Final Path Only:** Your final output for each problem MUST be a single, direct path to the solution. Do not include any self-corrections, abandoned attempts, or notes about re-interpreting the problem (e.g., "Wait, I misunderstood..."). The final text must read as if you solved it perfectly on the first try. If you realize a mistake mid-solution, you must discard the incorrect path and present only the flawless, corrected reasoning and final answer.

**Visuals & Diagrams: Textbook Quality is Mandatory**
*   **3D Geometry (Plotly.js):** For ANY problem involving the calculation of volume, surface area, or the visualization of surfaces, planes, or curves in 3D space, generating a Plotly.js 3D diagram is NOT optional—it is a mandatory and critical part of the solution.
*   **Statics & Dynamics (D3.js):** For any problem involving forces (e.g., statics, dynamics), you MUST generate a ***simplified***, clear, and accurate Free-Body Diagram (FBD) using D3.js. The diagram must isolate the object and show all applied forces, reaction forces, and moments as clearly labeled vectors. This is a non-optional, critical component of the solution.
    **YOU MUST GENERATE SAID DIAGRAMS FOR ALL NEEDED STEPS LIKE THE TANGENT NORMAL DIAGRAMS AND POLAR DIAGRAMS AND OTHER NEEDED TO SOLVE DIAGRAMS***   
*   **Other 2D Diagrams (D3.js):** Use D3.js for any other necessary 2D plots or diagrams to enhance explanations.

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
        "Please solve the problems on this new page and generate **only the new HTML content** for these problems. "
        "Your output must be an HTML snippet that starts directly with the content for the new problems (e.g., `<hr><h3>...`). "
        "**DO NOT** output a full HTML document or repeat any previous content."
    )

def get_snippet_review_prompt():
    """Generates the prompt for reviewing an isolated HTML snippet before merging."""
    return (
        "You are now acting as a meticulous editor. Below is a snippet of HTML containing the solution for one or more problems. Your task is to review and refine this snippet.\n\n"
        "1.  **Validate Structure and Requirements:** Ensure the snippet follows all structural and content rules (e.g., problem/solution divs, mandatory diagrams).\n\n"
        "2.  **Refine Solution Presentation:** This is your most important task. Scrutinize the step-by-step solution. If it shows any signs of self-correction, abandoned attempts, or a non-linear path to the answer (e.g., phrasing like 'Wait, I should have...' or 'A better approach is...'), you MUST rewrite that entire solution within the snippet. The rewritten solution must present a single, direct, confident, and flawless path from the problem statement to the final answer.\n\n"
        "Your final output MUST be the complete, corrected, and polished HTML snippet. Do not add any extra explanations or formatting like ```html."
    )

def get_full_document_review_prompt():
    """Generates the prompt for the final validation and refinement step of a full HTML document."""
    return (
        "Excellent. The initial document has now been generated. You will now act as a final editor.\n\n"
        "I am providing you with the complete HTML document. Your task is to review it meticulously and perform two actions:\n\n"
        "1.  **Validate Against All Original Requirements:** Read the entire HTML file and ensure it meets every single one of the initial system instructions.\n\n"
        "2.  **Refine Solutions for a Flawless Presentation:** Scrutinize the step-by-step solutions for every problem. If you find any solution that shows signs of self-correction or a non-linear path, you MUST rewrite that entire solution to be direct and confident.\n\n"
        "Your final output MUST be the complete, corrected, and polished HTML document. Provide the full, final HTML code."
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
    Processes a worksheet using a "Generate -> Review -> Merge" workflow.
    Saves the HTML file progressively after each page is processed.
    """
    item_start_time = time.time()
    ws_name = ws_path.name
    output_path = ws_path.with_suffix('.key.html')

    if output_path.exists():
        return 'skipped', ws_name, "Output file already exists."

    temp_dir_manager = None
    try:
        # PDF validation and splitting
        try:
            reader = pypdf.PdfReader(ws_path)
            num_pages = len(reader.pages)
            if num_pages == 0: return 'failed', ws_name, "Worksheet PDF is empty."
        except Exception as e: return 'failed', ws_name, f"Could not read PDF file: {e}"

        page_paths = [ws_path] if num_pages == 1 else []
        if num_pages > 1:
            temp_dir_manager = tempfile.TemporaryDirectory()
            page_paths = split_pdf(ws_path, Path(temp_dir_manager.name))
            if not page_paths:
                if temp_dir_manager: temp_dir_manager.cleanup()
                return 'failed', ws_name, "PDF could not be split."

        # Model and chat setup
        system_prompt = get_system_prompt()
        chat = model.start_chat(history=[
            {'role': 'user', 'parts': [system_prompt] + training_files},
            {'role': 'model', 'parts': ["Understood. I am ready. Please provide the first page of the worksheet."]}
        ])

        accumulated_html = ""
        total_pages = len(page_paths)

        for i, page_path in enumerate(page_paths):
            print(f"[*] Processing page {C_YELLOW}{i+1}/{total_pages}{C_END} for {C_BOLD}{ws_name}{C_END}...")
            page_file = upload_files_with_retry([page_path])

            if i == 0:
                # --- FIRST PAGE: Generate and then Review Full Document ---
                prompt = get_first_page_prompt(ws_name)
                
                # *** MODIFIED: Wait for API slot before sending message ***
                api_rate_limiter.wait_for_slot()
                response = chat.send_message([prompt] + page_file)
                
                if not response.parts: return 'failed', ws_name, "Model returned empty response on page 1 (generation)."
                initial_html = response.text.strip().removeprefix("```html").removesuffix("```").strip()

                if "<!DOCTYPE html>" not in initial_html: return 'failed', ws_name, "Model did not produce a valid HTML doc on page 1."

                print(f"  -> Reviewing initial document...")
                review_prompt = get_full_document_review_prompt()

                # *** MODIFIED: Wait for API slot before sending message ***
                api_rate_limiter.wait_for_slot()
                review_response = chat.send_message([review_prompt, initial_html])

                if not review_response.parts:
                     return 'failed', ws_name, "Model returned empty response on page 1 (review)."
                
                reviewed_html = review_response.text.strip().removeprefix("```html").removesuffix("```").strip()
                if "<!DOCTYPE html>" not in reviewed_html:
                    return 'failed', ws_name, "Review step for page 1 did not produce a valid HTML document."
                
                accumulated_html = reviewed_html.replace('{{WORKSHEET_NAME}}', ws_name)
                output_path.write_text(accumulated_html, encoding='utf-8')
                print(f"  -> {C_GREEN}Initial document review complete. Saved initial version to {output_path.name}{C_END}")

            else:
                # --- SUBSEQUENT PAGES: Generate Snippet -> Review Snippet -> Merge -> Save ---
                prompt = get_next_page_prompt()
                
                # *** MODIFIED: Wait for API slot before sending message ***
                api_rate_limiter.wait_for_slot()
                response = chat.send_message([prompt] + page_file)

                if not response.parts: return 'failed', ws_name, f"Model returned empty response on page {i+1} (snippet generation)."
                raw_snippet = response.text.strip().removeprefix("```html").removesuffix("```").strip()

                print(f"  -> Reviewing snippet for page {i+1}...")
                review_prompt = get_snippet_review_prompt()
                
                # *** MODIFIED: Wait for API slot before sending message ***
                api_rate_limiter.wait_for_slot()
                review_response = chat.send_message([review_prompt, raw_snippet])

                if not review_response.parts:
                    return 'failed', ws_name, f"Model returned empty response on page {i+1} (snippet review)."
                
                refined_snippet = review_response.text.strip().removeprefix("```html").removesuffix("```").strip()
                
                insertion_points = ['</div>\n</body>', '</div></body>', '</body>']
                found_point = next((p for p in insertion_points if p in accumulated_html), None)

                if found_point:
                    replacement_chunk = refined_snippet + '\n' + found_point
                    accumulated_html = accumulated_html.replace(found_point, replacement_chunk, 1)
                    output_path.write_text(accumulated_html, encoding='utf-8')
                    print(f"  -> {C_GREEN}Snippet for page {i+1} merged. Updated {output_path.name}.{C_END}")
                else:
                    return 'failed', ws_name, "Structural error: Could not find insertion point to merge refined snippet."
        
        if temp_dir_manager: temp_dir_manager.cleanup()
        item_duration = time.time() - item_start_time
        return 'success', ws_name, item_duration

    except Exception as e:
        if temp_dir_manager: temp_dir_manager.cleanup()
        return 'failed', ws_name, str(e)


def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Generate HTML answer keys from PDFs using a parallel, page-by-page, review-before-merge workflow.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--training-folder", type=str, required=True, help="Path to folder with 'training' PDFs.")
    parser.add_argument("--worksheets-folder", type=str, required=True, help="Path to folder with 'worksheet' PDFs.")
    
    # *** MODIFIED: Changed default max-workers to a more conservative value (4) ***
    # This helps prevent an initial burst of simultaneous requests from all workers,
    # which could immediately trigger the rate limit. A value slightly below the
    # requests-per-minute limit is a good starting point.
    parser.add_argument("--max-workers", type=int, default=4, help="Max number of worksheets to process in parallel.")
    
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
        # Session handling to reuse uploaded files
        session_files = {}
        if Path(SESSION_FILE).exists():
            try:
                with open(SESSION_FILE, 'r') as f:
                    session_data = json.load(f)
                    session_files = {item['local_name']: item['server_id'] for item in session_data}
            except (json.JSONDecodeError, IOError) as e:
                print(f"{C_YELLOW}[WARNING] Could not read session file: {e}. Re-uploading all.{C_END}")
        
        current_filenames = {p.name for p in training_pdf_paths}
        session_filenames = set(session_files.keys())
        paths_to_upload = [p for p in training_pdf_paths if p.name not in session_filenames]
        
        reused_files = []
        if session_filenames:
            verified_names = current_filenames.intersection(session_filenames)
            print(f"[*] Verifying {len(verified_names)} file(s) from session...")
            for name in sorted(list(verified_names)):
                try:
                    reused_file = genai.get_file(name=session_files[name])
                    reused_files.append(reused_file)
                except Exception:
                    paths_to_upload.append(training_path / name)
            if reused_files:
                print(f"  -> {C_GREEN}Successfully verified and reused {len(reused_files)} file(s).{C_END}")

        if paths_to_upload:
            print(f"[*] Uploading {len(paths_to_upload)} new or unverified file(s)...")
            try:
                newly_uploaded_files = upload_files_with_retry(paths_to_upload)
                training_files = reused_files + newly_uploaded_files
                path_map = {p.name: p for p in training_pdf_paths}
                final_local_paths = [path_map[f.display_name] for f in training_files]
                save_session(final_local_paths, training_files)
            except Exception as e:
                print(f"{C_RED}[FATAL] Could not upload training files. Aborting. Error: {e}{C_END}")
                return
        else:
            training_files = reused_files

    print(f"\n{C_BLUE}{C_BOLD}--- Phase 2: Processing Worksheets (max {args.max_workers} parallel) ---{C_END}")
    worksheet_pdf_paths = list(worksheets_path.glob("*.pdf"))
    if not worksheet_pdf_paths:
        print("No worksheet PDFs found to process.")
        return

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
                    print(f"  -> {C_GREEN}[SUCCESS] {ws_name} processed in {format_time(result)}.{C_END}")
                elif status == 'skipped':
                    skipped_count += 1
                    print(f"  -> {C_BLUE}[SKIPPED] {ws_name}. Reason: {result}{C_END}")
                elif status == 'failed':
                    failed_count += 1
                    print(f"  -> {C_RED}[FAILED]  {ws_name}. Reason: {result}{C_END}")
            except Exception as exc:
                failed_count += 1
                print(f"  -> {C_RED}[ERROR]   {ws_path.name} generated an exception: {exc}{C_END}")

    total_duration = time.time() - script_start_time
    print(f"\n{C_BLUE}{C_BOLD}{'-'*25} All Tasks Completed {'-'*25}{C_END}")
    print(f"Summary for {total_worksheets} total worksheet(s):")
    print(f"  - {C_GREEN}Success: {success_count}{C_END}")
    print(f"  - {C_BLUE}Skipped: {skipped_count}{C_END}")
    print(f"  - {C_RED}Failed:  {failed_count}{C_END}")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        print(f"Average time per processed worksheet: {format_time(avg_time)}")
        
    print(f"{C_BOLD}Total execution time: {format_time(total_duration)}{C_END}")

if __name__ == "__main__":
    main()
