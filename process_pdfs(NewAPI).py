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

# Import types for configuration
from google.generativeai import GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold, SafetySettingDict
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
                
                if self.history: # Ensure history is not empty before accessing
                    oldest_timestamp, _ = self.history[0]
                    sleep_time = (oldest_timestamp + self.period_seconds) - now + 0.1
                else: # If history is empty, it means we just popped everything, so no need to sleep
                    sleep_time = 0.1 # Small sleep to prevent tight loop in case of edge issues
                
                reason = "request limit" if current_requests >= self.max_requests_per_period else "token limit"
                if sleep_time > 0: # Only print and sleep if necessary
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
***You must think extremly hard ****
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
    <title>Answer Key: {{{{WORKSHEET_NAME}}}}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&family=Source+Code+Pro:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {{{{ --font-sans: 'Inter', sans-serif; --font-mono: 'Source Code Pro', monospace; --text-color: #333; --bg-color: #fdfdfd; --primary-color: #4A90E2; --border-color: #e0e0e0; --accent-bg-light: #f5f7fa; --success-color: #2E7D32; --success-bg: #E8F5E9; }}}}
        body {{{{ font-family: var(--font-sans); line-height: 1.7; margin: 0; background-color: var(--bg-color); color: var(--text-color); }}}}
        .container {{{{ max-width: 900px; margin: 40px auto; background: #fff; padding: 20px 40px; border-radius: 12px; border: 1px solid var(--border-color); box-shadow: 0 8px 30px rgba(0,0,0,0.05); }}}}
        h2, h3 {{{{ font-weight: 700; color: #111; border-bottom: 1px solid var(--border-color); padding-bottom: 12px; margin-top: 50px; }}}}
        h2 {{{{ text-align: center; font-size: 2.25em; margin-top: 0; margin-bottom: 20px; color: #000; }}}}
        h3 {{{{ font-size: 1.75em; color: var(--primary-color); }}}}
        .problem-statement {{{{ background-color: var(--accent-bg-light); padding: 25px; border-left: 5px solid var(--primary-color); border-radius: 8px; margin: 30px 0; font-size: 1.1em; }}}}
        .solution {{{{ padding: 10px 5px; }}}}
        .diagram-container {{{{ text-align: center; margin: 40px auto; padding: 25px; border: 1px solid var(--border-color); border-radius: 8px; }}}}
        hr {{{{ border: 0; height: 1px; background-color: var(--border-color); margin: 70px 0; }}}}
        .final-answer {{{{ font-weight: 700; color: var(--success-color); background-color: var(--success-bg); padding: 4px 8px; border-radius: 6px; display: inline-block; border: 1px solid #a5d6a7; }}}}
    </style>
</head>
<body>
<div class="container">
    <h2>Answer Key for {{{{WORKSHEET_NAME}}}}</h2>
</div>
</body>
</html>
"""
def get_first_page_prompt(worksheet_name): return f"This is the first page of the exam '{worksheet_name}'. Please start solving the problems on this page. Generate the full HTML document based on the system instructions, replacing '{{{{WORKSHEET_NAME}}}}' with the actual worksheet name."
def get_next_page_prompt(): return "Excellent work. Here is the next page of the exam. Please solve the problems on this new page and generate **only the new HTML content** for these problems. Your output must be an HTML snippet that starts directly with the content for the new problems (e.g., `<hr><h3>...`)."
def get_snippet_review_prompt(): return "You are now acting as a meticulous editor. Review and refine the following HTML snippet. Your most important task is to rewrite any solution that shows self-correction or a non-linear path, presenting a single, direct, confident path to the answer. Your final output must be the complete, polished HTML snippet."
def get_full_document_review_prompt(): return "You are a final editor. Review the complete HTML document. Ensure it meets all initial requirements and refine any solutions to be direct and confident. Your final output MUST be the complete, corrected, and polished HTML document."

# ===============================================================
# START: MULTI-KEY FILE MANAGER (UPDATED FOR google-genai)
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

    # The 'client' parameter is no longer needed here as genai functions are global after configure()
    def get_training_files(self, api_key: str, training_pdf_paths: List[Path]) -> List[Any]:
        with self.lock:
            if api_key in self.file_cache:
                print(f"{C_BLUE}[CACHE] Using cached training files for API key ending in '...{api_key[-4:]}'.{C_END}")
                return self.file_cache[api_key]

            print(f"{C_BLUE}[SETUP] Initializing and verifying training files for API key '...{api_key[-4:]}'.{C_END}")
            
            # Configure the API key before any genai calls
            genai.configure(api_key=api_key)

            if not training_pdf_paths:
                self.file_cache[api_key] = []
                return []

            session_files = self._load_session()
            current_filenames = {p.name for p in training_pdf_paths}
            paths_to_upload = [p for p in training_pdf_paths if p.name not in session_files.keys()]
            
            # The _verify_session_files also needs to be updated to use genai.get_file()
            reused_files = self._verify_session_files(session_files, current_filenames, training_pdf_paths)
            
            reused_filenames = {f.display_name for f in reused_files}
            # Only add paths to upload if they are in session_files but failed verification
            paths_to_upload.extend([p for p in training_pdf_paths if p.name in session_files.keys() and p.name not in reused_filenames and p not in paths_to_upload])


            newly_uploaded_files = []
            if paths_to_upload:
                print(f"[*] Uploading {len(paths_to_upload)} new or unverified file(s) for this key...")
                try:
                    # Update to use genai.upload_file()
                    newly_uploaded_files = self._upload_files_with_retry(paths_to_upload)
                except Exception as e:
                    print(f"{C_RED}[FATAL] Could not upload training files for key '...{api_key[-4:]}'. Error: {e}{C_END}")
                    raise

            all_files = reused_files + newly_uploaded_files
            self.file_cache[api_key] = all_files
            
            path_map = {p.name: p for p in training_pdf_paths}
            # Ensure the order of final_local_paths matches all_files for consistent session saving
            final_local_paths = [path_map[f.display_name] for f in all_files]
            self._save_session(final_local_paths, all_files)

            print(f"{C_GREEN}Setup complete for key '...{api_key[-4:]}'.{C_END}")
            return all_files

    def _load_session(self) -> Dict[str, str]:
        if not Path(SESSION_FILE).exists(): return {}
        try:
            with open(SESSION_FILE, 'r') as f: return {item['local_name']: item['server_id'] for item in json.load(f)}
        except (json.JSONDecodeError, IOError): return {}

    # The 'client' parameter is no longer needed here
    def _verify_session_files(self, session_files, current_filenames, all_paths):
        reused_files = []
        verified_names = current_filenames.intersection(session_files.keys())
        if not verified_names: return []
        
        print(f"[*] Verifying {len(verified_names)} file(s) from session for this key...")
        for name in sorted(list(verified_names)):
            try:
                # Update to use genai.get_file()
                file_info = genai.get_file(name=session_files[name])
                reused_files.append(file_info)
            except Exception:
                print(f"  -> {C_YELLOW}Verification failed for {name}. It will be re-uploaded.{C_END}")
        if reused_files: print(f"  -> {C_GREEN}Successfully verified and reused {len(reused_files)} file(s).{C_END}")
        return reused_files

    # The 'client' parameter is no longer needed here
    def _upload_files_with_retry(self, file_paths, max_retries=3):
        uploaded = []
        for path in file_paths:
            for attempt in range(max_retries):
                try:
                    print(f" - Uploading: {C_YELLOW}{path.name}{C_END}...")
                    # Update to use genai.upload_file()
                    file_info = genai.upload_file(
                        path=str(path), # genai.upload_file expects a path string or file-like object
                        display_name=path.name,
                    )
                    uploaded.append(file_info)
                    break
                except Exception as e:
                    if attempt + 1 == max_retries: raise
                    time.sleep(2 ** attempt)
        return uploaded

    def _save_session(self, training_pdf_paths, uploaded_files):
        session_data = [{"local_name": path.name, "server_id": file.name} for path, file in zip(training_pdf_paths, uploaded_files)]
        
        # Load existing data to merge, if any
        existing_data = {}
        if Path(SESSION_FILE).exists():
            try:
                with open(SESSION_FILE, 'r') as f:
                    for item in json.load(f):
                        existing_data[item['server_id']] = item
            except json.JSONDecodeError:
                pass # If file is malformed, start fresh

        # Merge new session data
        for item in session_data:
            existing_data[item['server_id']] = item

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
# END: MULTI-KEY FILE MANAGER
# ===============================================================

# ===============================================================
# START: "FAIL FAST" RETRY LOGIC (FIXED count_tokens)
# ===============================================================
def send_message_with_retry(model, chat, parts, max_retries=3):
    """
    Sends a message, but fails fast on the first 429 error to allow for
    quick API key rotation. Retries are for transient errors, not hard limits.
    """
    # Token counting with google-genai - FIXED VERSION
    try:
        # Use the model's count_tokens method instead of genai.count_tokens
        token_count = model.count_tokens(parts).total_tokens
        print(f"  -> Request will use ~{token_count} tokens.")
        api_rate_limiter.wait_for_slot(token_count)
    except Exception as e:
        print(f"{C_YELLOW}[WARNING] Could not count tokens: {e}. Proceeding without rate limiting for this request.{C_END}")
    
    for attempt in range(max_retries):
        try:
            response = chat.send_message(parts)
            return response
        except Exception as e:
            error_str = str(e)
            # Check for rate limit errors (429)
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                if attempt == 0:
                    print(f"{C_YELLOW}[API 429] Immediate rate limit hit. Propagating error to switch API key...{C_END}")
                    raise e

                print(f"{C_YELLOW}[API 429] Retrying... (Attempt {attempt + 1}/{max_retries}){C_END}")
                if attempt + 1 == max_retries:
                    raise e
                
                match = re.search(r'seconds: (\d+)', error_str)
                sleep_duration = int(match.group(1)) + 1 if match else 2 ** (attempt + 1)
                time.sleep(sleep_duration)
            else:
                print(f"{C_RED}[FATAL API ERROR] An unexpected error occurred: {e}{C_END}")
                raise e
# ===============================================================
# END: "FAIL FAST" RETRY LOGIC
# ===============================================================

# 'client' parameter removed from this function as well.
def process_single_worksheet(ws_path: Path, training_files: List[Any], model_id: str, generation_config: dict, safety_settings: List[SafetySettingDict]):
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
        
        # Initialize the model
        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_prompt, # system instruction is now part of model init
            safety_settings=safety_settings,
            generation_config=GenerationConfig( # Use GenerationConfig type
                temperature=generation_config.get('temperature', 1.0),
                top_p=generation_config.get('top_p', 0.95),
            )
        )
        
        # Start a chat session
        # The first message in start_chat is usually the user's initial prompt,
        # but here we're conceptually providing training files before the real conversation starts.
        # The training files are handled as part of `genai.configure_model` or by being part of the initial `contents`
        # when a model is instantiated for multi-turn conversational use.
        # For multi-modal (text + file) conversational turns, it's better to pass them directly
        # in the send_message call.
        
        print(f"[*] Starting chat session with model {model_id}...")
        # Start chat with a blank initial history, training files will be sent in the first actual message.
        chat = model.start_chat(history=[])

        start_page_index, accumulated_html = load_progress(progress_path)
        
        for i, page_path in enumerate(page_paths):
            if i <= start_page_index: continue

            print(f"[*] Processing page {C_YELLOW}{i+1}/{len(page_paths)}{C_END} for {C_BOLD}{ws_name}{C_END}...")
            
            # Upload page file using genai.upload_file()
            # It's generally better to upload files once and get their File objects,
            # then pass the File objects in the `send_message` call.
            page_file = genai.upload_file(
                path=str(page_path), 
                display_name=page_path.name,
            )

            if i == 0:
                prompt = get_first_page_prompt(ws_name)
                # First message includes system prompt text, training files, and the first page
                # The prompt has the context needed.
                contents_for_first_page = [prompt] + training_files + [page_file]
                response = send_message_with_retry(model, chat, contents_for_first_page)
                
                initial_html = response.text.strip().removeprefix("```html").removesuffix("```").strip()
                review_response = send_message_with_retry(model, chat, [get_full_document_review_prompt(), initial_html])
                accumulated_html = review_response.text.strip().removeprefix("```html").removesuffix("```").strip().replace('{{WORKSHEET_NAME}}', ws_name)
            else:
                prompt = get_next_page_prompt()
                # Subsequent messages include the prompt and the new page file
                response = send_message_with_retry(model, chat, [prompt, page_file])
                raw_snippet = response.text.strip().removeprefix("```html").removesuffix("```").strip()
                review_response = send_message_with_retry(model, chat, [get_snippet_review_prompt(), raw_snippet])
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

    except Exception as e:
        if '429' in str(e) or 'RESOURCE_EXHAUSTED' in str(e) or 'quota' in str(e).lower():
            raise e
        if temp_dir_manager: temp_dir_manager.cleanup()
        return 'failed', ws_name, str(e)

def process_worksheet_with_key_rotation(ws_path, file_manager, api_keys, model_name, training_pdf_paths):
    if ws_path.with_suffix('.key.html').exists() and not ws_path.with_suffix('.pdf.progress.json').exists():
        return 'skipped', ws_path.name, "Output file already exists."

    generation_config = {
        "temperature": 1.0,
        "top_p": 0.95,
    }
    
    safety_settings = [
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_NONE
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_NONE
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_NONE
    ),
    SafetySettingDict(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE
    ),
]

    for i, key in enumerate(api_keys):
        try:
            print(f"[*] Attempting {C_BOLD}{ws_path.name}{C_END} with Key #{i+1} ('...{key[-4:]}')")
            
            # Configure the API key globally for genai functions
            genai.configure(api_key=key)
            
            # file_manager.get_training_files no longer needs the 'client' parameter
            training_files = file_manager.get_training_files(key, training_pdf_paths)
            
            # process_single_worksheet no longer needs the 'client' parameter
            return process_single_worksheet(ws_path, training_files, model_name, generation_config, safety_settings)
        except Exception as e:
            error_str = str(e)
            if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
                print(f"{C_RED}[FATAL RATE LIMIT] API Key #{i+1} seems exhausted.{C_END}")
                if i < len(api_keys) - 1: print(f"{C_YELLOW}Switching to next key...{C_END}")
                else: return 'failed', ws_path.name, "All API keys exhausted. Progress saved for next run."
            else:
                return 'failed', ws_path.name, f"Unexpected error with key #{i+1}: {e}"
    
    return 'failed', ws_path.name, "Failed with all available API keys."


def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser(description="Generate HTML answer keys from PDFs with multi-key, resumable, parallel processing.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--training-folder", type=str, required=True, help="Folder with 'training' PDFs.")
    parser.add_argument("--worksheets-folder", type=str, required=True, help="Folder with 'worksheet' PDFs.")
    parser.add_argument("--max-workers", type=int, default=1, help="Max parallel worksheets.")
    args = parser.parse_args()

    api_keys = load_api_keys()
    if not api_keys:
        print(f"{C_RED}[ERROR] No Google API keys found in .env file.{C_END}")
        return
    print(f"{C_GREEN}[INFO] Found {len(api_keys)} API key(s).{C_END}")

    training_path = Path(args.training_folder)
    worksheets_path = Path(args.worksheets_folder)

    print(f"DEBUG: Training path resolved to: {training_path}")
    print(f"DEBUG: Worksheets path resolved to: {worksheets_path}")

    # SINGLE folder validation check
    if not training_path.is_dir() or not worksheets_path.is_dir():
        print(f"{C_RED}[ERROR] Invalid folder paths provided.{C_END}")
        print(f"Training folder exists: {training_path.is_dir()}")
        print(f"Worksheets folder exists: {worksheets_path.is_dir()}")
        return

    # Get PDF files
    training_pdfs = sorted(training_path.glob("*.pdf"))
    worksheet_pdfs = sorted(worksheets_path.glob("*.pdf"))

    if not training_pdfs:
        print(f"{C_YELLOW}[WARNING] No training PDFs found in {training_path}{C_END}")
    else:
        print(f"{C_GREEN}[INFO] Found {len(training_pdfs)} training PDF(s).{C_END}")

    if not worksheet_pdfs:
        print(f"{C_RED}[ERROR] No worksheet PDFs found in {worksheets_path}{C_END}")
        return
    else:
        print(f"{C_GREEN}[INFO] Found {len(worksheet_pdfs)} worksheet PDF(s).{C_END}")

    # Initialize file manager
    file_manager = ApiKeyFileManager()
    model_name = "gemini-2.5-pro"  # or your preferred model

    # Process worksheets
    print(f"\n{C_BOLD}Starting parallel processing with {args.max_workers} worker(s)...{C_END}\n")
    
    results = {'success': [], 'failed': [], 'skipped': []}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                process_worksheet_with_key_rotation,
                ws_path,
                file_manager,
                api_keys,
                model_name,
                training_pdfs
            ): ws_path for ws_path in worksheet_pdfs
        }
        
        for future in concurrent.futures.as_completed(futures):
            status, ws_name, detail = future.result()
            results[status].append((ws_name, detail))
            
            if status == 'success':
                print(f"{C_GREEN}✓ SUCCESS: {ws_name} (took {format_time(detail)}){C_END}")
            elif status == 'skipped':
                print(f"{C_BLUE}⊘ SKIPPED: {ws_name} - {detail}{C_END}")
            else:
                print(f"{C_RED}✗ FAILED: {ws_name} - {detail}{C_END}")

    # Summary
    total_time = time.time() - script_start_time
    print(f"\n{C_BOLD}{'='*60}{C_END}")
    print(f"{C_BOLD}PROCESSING COMPLETE{C_END}")
    print(f"{C_BOLD}{'='*60}{C_END}")
    print(f"{C_GREEN}Success: {len(results['success'])}{C_END}")
    print(f"{C_BLUE}Skipped: {len(results['skipped'])}{C_END}")
    print(f"{C_RED}Failed: {len(results['failed'])}{C_END}")
    print(f"\nTotal execution time: {format_time(total_time)}")

if __name__ == "__main__":
    main()
