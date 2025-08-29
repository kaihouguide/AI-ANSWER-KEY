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
from typing import List, Dict, Any, Tuple
import collections
import threading
import re
import google.api_core.exceptions

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
# START: TOKEN-AWARE RATE LIMITER (UNCHANGED)
# ===============================================================
class APITokenRateLimiter:
    """
    A thread-safe rate limiter to control API calls based on both the
    number of requests and the total number of tokens per minute.
    """
    def __init__(self, max_requests, max_tokens, period_seconds):
        self.max_requests_per_period = max_requests
        self.max_tokens_per_period = max_tokens
        self.period_seconds = period_seconds
        self.history = collections.deque()
        self.lock = threading.Lock()

    def wait_for_slot(self, upcoming_token_count: int):
        with self.lock:
            if upcoming_token_count > self.max_tokens_per_period:
                raise ValueError(
                    f"Request with {upcoming_token_count} tokens exceeds the "
                    f"per-minute limit of {self.max_tokens_per_period}. "
                    "This request must be split into smaller parts."
                )
            while True:
                now = time.monotonic()
                while self.history and self.history[0][0] <= now - self.period_seconds:
                    self.history.popleft()
                
                current_requests = len(self.history)
                current_tokens = sum(item[1] for item in self.history)

                can_proceed = (
                    current_requests < self.max_requests_per_period and
                    (current_tokens + upcoming_token_count) <= self.max_tokens_per_period
                )

                if can_proceed:
                    self.history.append((now, upcoming_token_count))
                    break 
                
                oldest_timestamp, _ = self.history[0]
                sleep_time = (oldest_timestamp + self.period_seconds) - now + 0.1
                
                reason = "request limit" if current_requests >= self.max_requests_per_period else "token limit"
                print(
                    f"{C_YELLOW}[RATE LIMIT] {reason.capitalize()} reached. "
                    f"Pausing for {sleep_time:.1f} seconds...{C_END}"
                )
                time.sleep(sleep_time)

api_rate_limiter = APITokenRateLimiter(
    max_requests=5, 
    max_tokens=240000, 
    period_seconds=60
)
# ===============================================================
# END: TOKEN-AWARE RATE LIMITER
# ===============================================================

def format_time(seconds):
    if seconds < 60: return f"{seconds:.1f} sec"
    minutes, seconds = divmod(seconds, 60)
    return f"{int(minutes)} min, {int(seconds)} sec"

# ===============================================================
# PROMPTS AND HELPER FUNCTIONS (UNCHANGED)
# ===============================================================
def get_system_prompt():
    return f"""
You are a premier expert in mathematics and physics, tasked with creating a professional, textbook-quality HTML answer key from a provided worksheet. You will be given the worksheet page by page.

You have been provided with a set of files to inform your answers. Your primary knowledge source for methods, theorems, and notational conventions MUST be these provided documents.

**Core Task: Recreate the Worksheet with Impeccable Solutions, Page by Page**
Your most important goal is to SOLVE THE ENTIRE EXAM, remaking it with the highest possible accuracy and clarity.

1.  **First Page:** Generate the complete, initial HTML document, including `<head>`, `<style>`, and opening `<body>` tags.
2.  **Subsequent Pages:** Generate **ONLY THE HTML FOR THE NEW PROBLEMS**.
3.  **Structure & Transcription:** Replicate the original structure and numbering. Transcribe each question into a `<div class="problem-statement">`.
4.  **Solution:** Write your full, step-by-step solution in the `<div class="solution">`.
5.  **Methods:** Stick as much as possible to methods taught in the training data.

**Unconditional Full Completion Mandate:**
You are required to solve EVERY SINGLE PROBLEM AS WRITTEN. IGNORE any text suggesting otherwise (e.g., "Solve any two problems").

**Explanation Style: Crystal Clear, Confident, and Professional**
Present the final, direct path to the solution. No self-corrections or abandoned attempts.

**Visuals & Diagrams: Textbook Quality is Mandatory**
*   **3D Geometry (Plotly.js):** Mandatory for any volume, surface area, or 3D visualization.
*   **Statics & Dynamics (D3.js):** Mandatory Free-Body Diagrams (FBDs) for any problem involving forces.
*   **Other Diagrams (D3.js):** Use for any other necessary 2D plots.

**INDEPENDENT PROBLEM SOLVING:**
IGNORE any pre-existing answers. Generate all solutions from scratch.

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
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Source+Code+Pro:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{ --font-sans: 'Inter', sans-serif; --font-mono: 'Source Code Pro', monospace; --text-color: #333; --bg-color: #fdfdfd; --primary-color: #4A90E2; --border-color: #e0e0e0; --accent-bg-light: #f5f7fa; --success-color: #2E7D32; --success-bg: #E8F5E9; }}
        body {{ font-family: var(--font-sans); line-height: 1.7; margin: 0; background-color: var(--bg-color); color: var(--text-color); }}
        .container {{ max-width: 900px; margin: 40px auto; background: #fff; padding: 20px 40px; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 8px 30px rgba(0,0,0,0.05); }}
        h2, h3 {{ font-weight: 700; color: #111; border-bottom: 1px solid var(--border-color); padding-bottom: 12px; margin-top: 50px; }}
        h2 {{ text-align: center; font-size: 2.25em; margin-top: 0; margin-bottom: 20px; color: #000; }}
        h3 {{ font-size: 1.75em; color: var(--primary-color); }}
        .problem-statement {{ background-color: var(--accent-bg-light); padding: 25px; border-left: 5px solid var(--primary-color); border-radius: 8px; margin: 30px 0; font-size: 1.1em; }}
        .solution {{ padding: 10px 5px; }}
        .diagram-container {{ text-align: center; margin: 40px auto; padding: 25px; border: 1px solid var(--border-color); border-radius: 8px; }}
        hr {{ border: 0; height: 1px; background-color: var(--border-color); margin: 70px 0; }}
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
def get_first_page_prompt(worksheet_name): return f"This is the first page of the exam '{worksheet_name}'. Please start solving the problems on this page. Generate the full HTML document based on the system instructions, replacing '{{WORKSHEET_NAME}}' with the actual worksheet name."
def get_next_page_prompt(): return "Excellent work. Here is the next page of the exam. Please solve the problems on this new page and generate **only the new HTML content** for these problems. Your output must be an HTML snippet that starts directly with the content for the new problems (e.g., `<hr><h3>...`)."
def get_snippet_review_prompt(): return "You are now acting as a meticulous editor. Review and refine the following HTML snippet. Your most important task is to rewrite any solution that shows self-correction or a non-linear path, presenting a single, direct, confident path to the answer. Your final output must be the complete, polished HTML snippet."
def get_full_document_review_prompt(): return "You are a final editor. Review the complete HTML document. Ensure it meets all initial requirements and refine any solutions to be direct and confident. Your final output MUST be the complete, corrected, and polished HTML document."

# ===============================================================
# START: MODIFICATION - MULTI-KEY FILE MANAGER (UNCHANGED)
# ===============================================================
def load_api_keys() -> List[str]:
    """Loads one or more API keys from the .env file."""
    load_dotenv(override=True)
    keys = []
    if os.getenv("GOOGLE_API_KEY"): keys.append(os.getenv("GOOGLE_API_KEY"))
    i = 1
    while True:
        if os.getenv(f"GOOGLE_API_KEY_{i}"): keys.append(os.getenv(f"GOOGLE_API_KEY_{i}")); i += 1
        else: break
    return keys

class ApiKeyFileManager:
    """Manages training file uploads and verification on a per-API-key basis."""
    def __init__(self):
        self.file_cache: Dict[str, List[Any]] = {}
        self.lock = threading.Lock()

    def get_training_files(self, api_key: str, training_pdf_paths: List[Path]) -> List[Any]:
        with self.lock:
            if api_key in self.file_cache:
                print(f"{C_BLUE}[CACHE] Using cached training files for API key ending in '...{api_key[-4:]}'.{C_END}")
                return self.file_cache[api_key]

            print(f"{C_BLUE}[SETUP] Initializing and verifying training files for API key '...{api_key[-4:]}'.{C_END}")
            genai.configure(api_key=api_key)
            
            if not training_pdf_paths:
                self.file_cache[api_key] = []
                return []

            session_files = self._load_session()
            current_filenames = {p.name for p in training_pdf_paths}
            paths_to_upload = [p for p in training_pdf_paths if p.name not in session_files.keys()]
            reused_files = self._verify_session_files(session_files, current_filenames, training_pdf_paths)
            
            reused_filenames = {f.display_name for f in reused_files}
            paths_to_upload.extend([p for p in training_pdf_paths if p.name in session_files.keys() and p.name not in reused_filenames])

            newly_uploaded_files = []
            if paths_to_upload:
                print(f"[*] Uploading {len(paths_to_upload)} new or unverified file(s) for this key...")
                try:
                    newly_uploaded_files = self._upload_files_with_retry(paths_to_upload)
                except Exception as e:
                    print(f"{C_RED}[FATAL] Could not upload training files for key '...{api_key[-4:]}'. Error: {e}{C_END}")
                    raise

            all_files = reused_files + newly_uploaded_files
            self.file_cache[api_key] = all_files
            
            path_map = {p.name: p for p in training_pdf_paths}
            final_local_paths = [path_map[f.display_name] for f in all_files]
            self._save_session(final_local_paths, all_files)

            print(f"{C_GREEN}Setup complete for key '...{api_key[-4:]}'.{C_END}")
            return all_files

    def _load_session(self) -> Dict[str, str]:
        if not Path(SESSION_FILE).exists(): return {}
        try:
            with open(SESSION_FILE, 'r') as f: return {item['local_name']: item['server_id'] for item in json.load(f)}
        except (json.JSONDecodeError, IOError): return {}

    def _verify_session_files(self, session_files, current_filenames, all_paths):
        reused_files = []
        verified_names = current_filenames.intersection(session_files.keys())
        if not verified_names: return []
        
        print(f"[*] Verifying {len(verified_names)} file(s) from session for this key...")
        for name in sorted(list(verified_names)):
            try:
                reused_files.append(genai.get_file(name=session_files[name]))
            except Exception:
                print(f"  -> {C_YELLOW}Verification failed for {name}. It will be re-uploaded.{C_END}")
        if reused_files: print(f"  -> {C_GREEN}Successfully verified and reused {len(reused_files)} file(s).{C_END}")
        return reused_files

    def _upload_files_with_retry(self, file_paths, max_retries=3):
        uploaded = []
        for path in file_paths:
            for attempt in range(max_retries):
                try:
                    print(f" - Uploading: {C_YELLOW}{path.name}{C_END}...")
                    uploaded.append(genai.upload_file(path=path, display_name=path.name))
                    break
                except Exception as e:
                    if attempt + 1 == max_retries: raise
                    time.sleep(2 ** attempt)
        return uploaded

    def _save_session(self, training_pdf_paths, uploaded_files):
        session_data = [{"local_name": path.name, "server_id": file.name} for path, file in zip(training_pdf_paths, uploaded_files)]
        existing_data = {item['server_id']: item for item in (json.load(open(SESSION_FILE)) if Path(SESSION_FILE).exists() else [])}
        for item in session_data: existing_data[item['server_id']] = item
        try:
            with open(SESSION_FILE, "w") as f: json.dump(list(existing_data.values()), f, indent=4)
        except Exception as e: print(f"{C_YELLOW}[WARNING] Could not save session data: {e}{C_END}")

def save_progress(progress_file: Path, page_index: int, html_content: str):
    try:
        progress_file.write_text(json.dumps({"last_completed_page_index": page_index, "accumulated_html": html_content}, indent=4), encoding='utf-8')
    except Exception as e: print(f"{C_YELLOW}[WARNING] Could not save progress to {progress_file.name}: {e}{C_END}")

def load_progress(progress_file: Path) -> Tuple[int, str]:
    if not progress_file.exists(): return -1, ""
    try:
        state = json.loads(progress_file.read_text(encoding='utf-8'))
        print(f"{C_BLUE}[RESUME] Progress file found. Resuming from page {state.get('last_completed_page_index', -1) + 2}.{C_END}")
        return state.get("last_completed_page_index", -1), state.get("accumulated_html", "")
    except (json.JSONDecodeError, IOError): return -1, ""

def split_pdf(pdf_path: Path, output_dir: Path) -> List[Path]:
    print(f"[*] Splitting '{pdf_path.name}'...")
    page_paths = []
    try:
        reader = pypdf.PdfReader(pdf_path)
        for i, page in enumerate(reader.pages):
            writer = pypdf.PdfWriter(); writer.add_page(page)
            page_output_path = output_dir / f"{pdf_path.stem}_page_{i+1}.pdf"
            with open(page_output_path, "wb") as f: writer.write(f)
            page_paths.append(page_output_path)
        print(f"  -> {C_GREEN}Split into {len(page_paths)} pages.{C_END}")
        return page_paths
    except Exception as e: print(f"  -> {C_RED}Failed to split PDF: {e}{C_END}"); raise
# ===============================================================
# END: MODIFICATION
# ===============================================================

# ===============================================================
# START: MODIFICATION - "FAIL FAST" RETRY LOGIC
# ===============================================================
def send_message_with_retry(chat, model, parts, max_retries=3):
    """
    Sends a message, but fails fast on the first 429 error to allow for
    quick API key rotation. Retries are for transient errors, not hard limits.
    """
    token_count = model.count_tokens(parts).total_tokens
    print(f"  -> Request will use ~{token_count} tokens.")
    api_rate_limiter.wait_for_slot(token_count)
    
    for attempt in range(max_retries):
        try:
            return chat.send_message(parts)
        except google.api_core.exceptions.ResourceExhausted as e:
            # AGGRESSIVE SWITCH: If the very first attempt fails with a 429,
            # assume the key is hard-limited and propagate the error immediately
            # to trigger a key switch instead of waiting.
            if attempt == 0:
                print(f"{C_YELLOW}[API 429] Immediate rate limit hit. Propagating error to switch API key...{C_END}")
                raise e

            # If it's not the first attempt, it might be a transient issue, so we wait.
            print(f"{C_YELLOW}[API 429] Retrying... (Attempt {attempt + 1}/{max_retries}){C_END}")
            if attempt + 1 == max_retries:
                raise e # Fail after the last attempt
            
            match = re.search(r'seconds: (\d+)', str(e))
            sleep_duration = int(match.group(1)) + 1 if match else 2 ** (attempt + 1)
            time.sleep(sleep_duration)
        except Exception as e:
            print(f"{C_RED}[FATAL API ERROR] An unexpected error occurred: {e}{C_END}")
            raise e
# ===============================================================
# END: MODIFICATION
# ===============================================================

def process_single_worksheet(ws_path: Path, training_files: List[Any], model: genai.GenerativeModel):
    item_start_time = time.time()
    ws_name = ws_path.name
    output_path = ws_path.with_suffix('.key.html')
    progress_path = ws_path.with_suffix('.pdf.progress.json')

    temp_dir_manager = None
    try:
        try:
            reader = pypdf.PdfReader(ws_path)
            if len(reader.pages) == 0: return 'failed', ws_name, "PDF is empty."
        except Exception as e: return 'failed', ws_name, f"Could not read PDF: {e}"

        temp_dir_manager = tempfile.TemporaryDirectory()
        page_paths = split_pdf(ws_path, Path(temp_dir_manager.name))

        system_prompt = get_system_prompt()
        chat = model.start_chat(history=[
            {'role': 'user', 'parts': [system_prompt] + training_files},
            {'role': 'model', 'parts': ["Understood. I am ready."]}
        ])

        start_page_index, accumulated_html = load_progress(progress_path)
        
        for i, page_path in enumerate(page_paths):
            if i <= start_page_index: continue

            print(f"[*] Processing page {C_YELLOW}{i+1}/{len(page_paths)}{C_END} for {C_BOLD}{ws_name}{C_END}...")
            # Use a static method call here for consistency
            page_file = ApiKeyFileManager._upload_files_with_retry(None, [page_path])

            if i == 0:
                prompt = get_first_page_prompt(ws_name)
                response = send_message_with_retry(chat, model, [prompt] + page_file)
                initial_html = response.text.strip().removeprefix("```html").removesuffix("```").strip()
                review_response = send_message_with_retry(chat, model, [get_full_document_review_prompt(), initial_html])
                accumulated_html = review_response.text.strip().removeprefix("```html").removesuffix("```").strip().replace('{{WORKSHEET_NAME}}', ws_name)
            else:
                prompt = get_next_page_prompt()
                response = send_message_with_retry(chat, model, [prompt] + page_file)
                raw_snippet = response.text.strip().removeprefix("```html").removesuffix("```").strip()
                review_response = send_message_with_retry(chat, model, [get_snippet_review_prompt(), raw_snippet])
                refined_snippet = review_response.text.strip().removeprefix("```html").removesuffix("```").strip()
                
                insertion_point = next((p for p in ['</div>\n</body>', '</div></body>', '</body>'] if p in accumulated_html), None)
                if not insertion_point: return 'failed', ws_name, "Structural error: Could not merge snippet."
                accumulated_html = accumulated_html.replace(insertion_point, refined_snippet + '\n' + insertion_point, 1)
            
            save_progress(progress_path, i, accumulated_html)
            output_path.write_text(accumulated_html, encoding='utf-8')
            print(f"  -> {C_GREEN}Page {i+1} complete. Progress saved.{C_END}")

        if temp_dir_manager: temp_dir_manager.cleanup()
        if progress_path.exists(): progress_path.unlink()
        return 'success', ws_name, time.time() - item_start_time

    except google.api_core.exceptions.ResourceExhausted as e: raise e
    except Exception as e:
        if temp_dir_manager: temp_dir_manager.cleanup()
        return 'failed', ws_name, str(e)

def process_worksheet_with_key_rotation(ws_path, file_manager, api_keys, model_name, training_pdf_paths):
    if ws_path.with_suffix('.key.html').exists() and not ws_path.with_suffix('.pdf.progress.json').exists():
        return 'skipped', ws_path.name, "Output file already exists."

    # --- START: New Safety Settings Configuration ---
    safety_settings = {
        'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
        'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
        'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
        'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
    }
    # --- END: New Safety Settings Configuration ---

    for i, key in enumerate(api_keys):
        try:
            print(f"[*] Attempting {C_BOLD}{ws_path.name}{C_END} with Key #{i+1} ('...{key[-4:]}')")
            training_files = file_manager.get_training_files(key, training_pdf_paths)
            genai.configure(api_key=key)
            # --- MODIFIED: Added safety_settings to the model initialization ---
            model = genai.GenerativeModel(model_name, safety_settings=safety_settings)
            return process_single_worksheet(ws_path, training_files, model)
        except google.api_core.exceptions.ResourceExhausted:
            print(f"{C_RED}[FATAL RATE LIMIT] API Key #{i+1} seems exhausted.{C_END}")
            if i < len(api_keys) - 1: print(f"{C_YELLOW}Switching to next key...{C_END}")
            else: return 'failed', ws_path.name, "All API keys exhausted. Progress saved for next run."
        except Exception as e:
            return 'failed', ws_path.name, f"Unexpected error with key #{i+1}: {e}"
    
    return 'failed', ws_path.name, "Failed with all available API keys."


def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate HTML answer keys from PDFs with multi-key, resumable, parallel processing.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--training-folder", type=str, required=True, help="Folder with 'training' PDFs.")
    parser.add_argument("--worksheets-folder", type=str, required=True, help="Folder with 'worksheet' PDFs.")
    parser.add_argument("--max-workers", type=int, default=4, help="Max parallel worksheets.")
    args = parser.parse_args()

    api_keys = load_api_keys()
    if not api_keys:
        print(f"{C_RED}[ERROR] No Google API keys found in .env file.{C_END}")
        return
    print(f"{C_GREEN}[INFO] Found {len(api_keys)} API key(s).{C_END}")

    training_path = Path(args.training_folder)
    worksheets_path = Path(args.worksheets_folder)
    if not training_path.is_dir() or not worksheets_path.is_dir():
        print(f"{C_RED}[ERROR] Invalid folder paths provided.{C_END}")
        return

    training_pdf_paths = list(training_path.glob("*.pdf"))
    worksheet_pdf_paths = list(worksheets_path.glob("*.pdf"))
    if not worksheet_pdf_paths:
        print("No worksheet PDFs found to process."); return

    file_manager = ApiKeyFileManager()
    model_name = 'gemini-2.5-pro' # Changed to a valid model name
    results = {'success': 0, 'skipped': 0, 'failed': 0}
    processing_times = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_ws = {
            executor.submit(process_worksheet_with_key_rotation, ws_path, file_manager, api_keys, model_name, training_pdf_paths): ws_path 
            for ws_path in worksheet_pdf_paths
        }

        for i, future in enumerate(concurrent.futures.as_completed(future_to_ws)):
            ws_path = future_to_ws[future]
            print(f"[{i+1}/{len(worksheet_pdf_paths)}] Finalizing result for {C_BOLD}{ws_path.name}{C_END}...")
            try:
                status, ws_name, result = future.result()
                results[status] += 1
                if status == 'success':
                    processing_times.append(result)
                    print(f"  -> {C_GREEN}[SUCCESS] {ws_name} processed in {format_time(result)}.{C_END}")
                elif status == 'skipped':
                    print(f"  -> {C_BLUE}[SKIPPED] {ws_name}. Reason: {result}{C_END}")
                else: # failed
                    print(f"  -> {C_RED}[FAILED]  {ws_name}. Reason: {result}{C_END}")
            except Exception as exc:
                results['failed'] += 1
                print(f"  -> {C_RED}[ERROR]   {ws_path.name} generated an exception: {exc}{C_END}")

    total_duration = time.time() - script_start_time
    print(f"\n{C_BLUE}{C_BOLD}{'-'*25} All Tasks Completed {'-'*25}{C_END}")
    print(f"Summary: {C_GREEN}Success: {results['success']}{C_END}, {C_BLUE}Skipped: {results['skipped']}{C_END}, {C_RED}Failed: {results['failed']}{C_END}")
    if processing_times: print(f"Average time per processed worksheet: {format_time(sum(processing_times) / len(processing_times))}")
    print(f"{C_BOLD}Total execution time: {format_time(total_duration)}{C_END}")

if __name__ == "__main__":
    main()
