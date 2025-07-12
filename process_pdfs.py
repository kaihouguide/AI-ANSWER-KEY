import google.generativeai as genai
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time
import concurrent.futures
import json

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
# START: MODIFIED PROMPT FUNCTION (D3.js for 2D, Plotly.js for 3D)
# ===============================================================
def get_system_prompt(worksheet_name):
    """
    Generates an enhanced system prompt demanding textbook-quality output
    and a more professional visual style, using D3.js for 2D and Plotly.js for 3D diagrams.
    """
    prompt = f"""
You are a premier expert in mathematics and physics, tasked with creating a professional, textbook-quality HTML answer key from a provided worksheet. Your output must be flawless in its accuracy, pedagogy, and visual presentation.

You have been provided with a set of files to inform your answers. Your primary knowledge source for methods, theorems, and notational conventions MUST be these provided documents. You do not need to state which PDF you used for a specific method.

Your task is to process the following worksheet:
- `{worksheet_name}`

**Core Task: Recreate the Worksheet with Impeccable Solutions**
Your goal is to SOLVE THE ENTIRE EXAM, remaking it with the highest possible accuracy and clarity.
1.  **Preserve Structure:** Replicate the original structure, numbering, and layout as closely as possible within the professional HTML format.
2.  **Transcribe the Problem:** For every problem, transcribe the question exactly as it appears into a `<div class="problem-statement">`.
3.  **Provide the Solution:** Write your full, detailed, step-by-step solution in the `<div class="solution">` that immediately follows.

**Crucial Directive for 3D Geometry:**
For ANY problem involving the calculation of volume, surface area, or the visualization of surfaces, planes, or curves in 3D space, generating a Plotly.js 3D diagram is NOT optional—it is a mandatory and critical part of the solution. The purpose is to provide visual intuition alongside the analytical solution. For a problem like "Find the volume bounded by y = x^2, z = 0, and y + z = 4," a 3D plot showing these surfaces is required.

**Explanation Style: Crystal Clear, Pedagogical, and Professional**
Your explanations must be exceptionally clear, easy to follow, and structured like a perfect example from a university textbook.

*   **1. State the Strategy:** Begin with a concise summary of the method you will use.
    *   *Example:* "To solve this, we will apply Green's Theorem to convert the line integral into a more manageable double integral over the enclosed rectangular region."

*   **2. Justify Key Choices:** Briefly explain why the chosen method is appropriate.
    *   *Example:* "Green's Theorem is ideal here because the curve is closed and the vector field components are simple polynomials, making the partial derivatives easy to compute."

*   **3. Show All Calculation Steps:** Meticulously show every step of the calculation. Do not skip any algebraic simplification, differentiation, or integration steps.
    *   When applying a theorem, explicitly state the components (e.g., "Let \\(P = ...\\) and \\(Q = ...\\). The partial derivatives are \\(\\frac{{\\partial Q}}{{\\partial x}} = ...\\) and \\(\\frac{{\\partial P}}{{\\partial y}} = ...\\).").

*   **4. Guide the Reader:** Use short, declarative sentences to introduce each major part of the calculation.

*   **5. Define All Variables:** Any new variables or coordinate systems must be clearly defined before use.

**Visuals & Diagrams: Textbook Quality is Mandatory**
You must generate a complete HTML document using the exact, professional template below. All diagrams and graphs must be of textbook quality.

1.  **Structure:** For each problem, use `<hr>`, `<h3>`, `.problem-statement`, and `.solution`.
2.  **2D Diagrams with D3.js:** If a 2D diagram is helpful (e.g., function graphs, regions of integration, vector fields in 2D, FBDs), it MUST be of professional quality.
    *   Create a `<div class="diagram-container">` containing a `<div id="d3-chart-unique-id" class="d3-chart"></div>` and a `<script>` tag with the D3.js code.
    *   **ESSENTIAL D3.js Setup Steps:**
        *   Define SVG dimensions and margins.
        *   Append an `<svg>` element to the `div` with `id="d3-chart-unique-id"`.
        *   Define scales (e.g., `d3.scaleLinear()`) for x and y axes.
        *   Create and append axes (e.g., `d3.axisBottom`, `d3.axisLeft`).
        *   Add grid lines.
        *   Plot data using `d3.line()` or other shape generators.
        *   Add labels and a title.
    *   **Quality Mandates:** Use professional colors, clearly labeled axes, titles, and grid lines. Ensure proper scaling and aspect ratio.
3.  **3D Diagrams with Plotly.js (MANDATORY for 3D concepts):** For any problem involving 3D visualizations (e.g., surfaces, planes, or curves in space), you MUST use the Plotly.js library.
    *   Create a `<div class="diagram-container">` for the plot.
    *   Inside, add a `<div id="plotly-chart-unique-id" style="width: 100%; height: 400px; display: flex; justify-content: center; align-items: center;"></div>`.
    *   Immediately following, add a `<script>` tag. Inside this script, define your Plotly.js plot.

    *   **ESSENTIAL Plotly.js Setup Steps (MUST be followed for each 3D plot):**
        1.  **Container Sizing:** Get the container element and its dimensions for `Plotly.newPlot`.
            ```javascript
            const containerDiv = document.getElementById('plotly-chart-unique-id');
            const plotWidth = containerDiv.clientWidth;
            const plotHeight = containerDiv.clientHeight; // Or a fixed height like 400
            ```
        2.  **Define `data` array:** This array contains objects for each trace (e.g., a surface, a scatter plot).
            *   For a 3D surface plot (`z = f(x,y)`), use `type: 'surface'`. You'll need to generate 2D arrays for `z` values, and optionally 1D arrays for `x` and `y` if your domain isn't linearly spaced or needs specific labels.
            *   **Example for a single surface (e.g., `z = x^2 + y^2`):**
                ```javascript
                // Generate x, y, z data based on the problem's function and domain
                const numPoints = 30; // Number of points along each axis for the grid
                const x = [];
                const y = [];
                const z = [];

                for (let i = 0; i < numPoints; i++) {{
                    const xi = -5 + (10 * i / (numPoints - 1)); // Example x range from -5 to 5
                    x.push(xi);
                    const row = [];
                    for (let j = 0; j < numPoints; j++) {{
                        const yj = -5 + (10 * j / (numPoints - 1)); // Example y range from -5 to 5
                        if (i === 0) y.push(yj); // Populate y array only once
                        // Calculate z value based on the mathematical function of the problem
                        row.push(xi * xi + yj * yj); // Example function: z = x^2 + y^2
                    }}
                    z.push(row);
                }}

                const data = [{{
                    z: z, // REQUIRED: A 2D array of Z values
                    x: x, // OPTIONAL: 1D array of X values, if custom ranges/labels are needed
                    y: y, // OPTIONAL: 1D array of Y values, if custom ranges/labels are needed
                    type: 'surface',
                    colorscale: 'Viridis', // Or 'Plotly', 'Jet', 'Portland', 'Greys', etc. for professional look
                    opacity: 0.8, // Adjust as needed, values from 0 to 1 for transparency
                    showscale: false // Hide color scale bar if not needed
                }}];
                ```
            *   For multiple surfaces or objects, add more trace objects to the `data` array, each with its own `type` and data.
        3.  **Define `layout` object:** This object configures the overall plot appearance, including the 3D scene.
            *   Set `title`, `autosize`, and crucially the `scene` object for 3D axis labels and camera.
            *   **Example `layout` for a 3D scene:**
                ```javascript
                const layout = {{
                    title: 'Your 3D Plot Title Here',
                    autosize: true,
                    width: plotWidth,
                    height: plotHeight,
                    margin: {{ l: 0, r: 0, b: 0, t: 30 }},
                    scene: {{
                        xaxis: {{ title: 'X-axis', range: [-5, 5] }}, // Set appropriate ranges based on the problem
                        yaxis: {{ title: 'Y-axis', range: [-5, 5] }},
                        zaxis: {{ title: 'Z-axis', range: [-5, 5] }},
                        aspectmode: 'cube', // 'data', 'auto', 'cube', 'manual' for controlling axis scaling
                        camera: {{
                            eye: {{ x: 1.8, y: 1.8, z: 1.8 }} // Adjust camera position for good initial view
                        }}
                    }}
                }};
                ```
        4.  **Create the plot:** `Plotly.newPlot('plotly-chart-unique-id', data, layout);` **Ensure the `div` ID matches exactly.**
        5.  **Window Resize Handling (Recommended for responsiveness):**
            ```javascript
            window.addEventListener('resize', function() {{
                const updatedContainerDiv = document.getElementById('plotly-chart-unique-id');
                Plotly.relayout('plotly-chart-unique-id', {{
                    width: updatedContainerDiv.clientWidth,
                    height: updatedContainerDiv.clientHeight || 400, // Maintain a minimum height if needed
                    autosize: true
                }});
            }});
            ```
    *   **Quality Mandates (for both D3.js and Plotly.js):**
        *   **Aesthetics:** Use professional and aesthetically pleasing colors.
        *   **Clarity:** Ensure the visualization clearly represents the mathematical concept. Label all axes and include titles where appropriate.
        *   **Interactivity:** D3.js plots should be interactive where applicable (e.g., tooltips, zooming for graphs), and Plotly.js plots are interactive by default (rotate, zoom, pan for 3D).

**INDEPENDENT PROBLEM SOLVING:**
You MUST IGNORE any pre-existing answers in the provided files. Generate all solutions from scratch.
you don't have to mention what pdf you got the concept/solution from.

**HTML TEMPLATE (USE THIS EXACTLY):**
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Answer Key: {worksheet_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script> <!-- Plotly.js CDN -->
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
        body {{
            font-family: var(--font-sans);
            line-height: 1.7;
            margin: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
        }}
        .container {{
            max-width: 900px;
            margin: 40px auto;
            background: #fff;
            padding: 20px 40px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: 0 8px 30px rgba(0,0,0,0.05);
        }}
        h2, h3 {{
            font-family: var(--font-sans);
            font-weight: 700;
            color: #111;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 12px;
            margin-top: 50px;
        }}
        h2 {{
            text-align: center;
            font-size: 2.25em;
            margin-top: 0;
            margin-bottom: 20px;
            color: #000;
        }}
        h3 {{
            font-size: 1.75em;
            color: var(--primary-color);
        }}
        .problem-statement {{
            background-color: var(--accent-bg-light);
            padding: 25px;
            border-left: 5px solid var(--primary-color);
            border-radius: 8px;
            margin: 30px 0;
            font-size: 1.1em;
        }}
        .solution {{
            padding: 10px 5px;
        }}
        .diagram-container {{
            text-align: center;
            margin: 40px auto;
            padding: 25px;
            background-color: #fff;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            filter: drop-shadow(0 4px 6px rgba(0,0,0,0.04));
        }}
        .d3-chart {{
             margin: auto;
             width: 100%;
             max-width: 650px;
        }}
        .diagram-caption {{
            font-size: 0.95em;
            font-style: italic;
            color: #666;
            margin-top: 20px;
        }}
        .d3-chart text, .axis text {{
            font-family: var(--font-sans);
            font-size: 12px;
        }}
        .grid .tick {{
            stroke: lightgray;
            opacity: 0.7;
        }}
        .grid path {{
            stroke-width: 0;
        }}
        hr {{
            border: 0;
            height: 1px;
            background-color: var(--border-color);
            margin: 70px 0;
        }}
        code {{
            font-family: var(--font-mono);
            background-color: var(--accent-bg-dark);
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.95em;
        }}
        .final-answer {{
            font-weight: 700;
            color: var(--success-color);
            background-color: var(--success-bg);
            padding: 4px 8px;
            border-radius: 6px;
            display: inline-block;
            border: 1px solid #a5d6a7;
        }}
    </style>
</head>
<body>
<div class="container">
    <h2>Answer Key for {worksheet_name}</h2>

    <!-- AI GENERATED CONTENT GOES HERE. -->

</div>
</body>
</html>
"""
    return prompt
# ===============================================================
# END: MODIFIED PROMPT FUNCTION
# ===============================================================

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
        print(f"{C_GREEN}[i] Session data saved to {SESSION_FILE} for future use.{C_END}")
    except Exception as e:
        print(f"{C_YELLOW}[WARNING] Could not save session data: {e}{C_END}")

def load_session():
    """Loads and verifies a previous session, asking the user for confirmation."""
    if not os.path.exists(SESSION_FILE):
        return None
    try:
        with open(SESSION_FILE, "r") as f:
            session_data = json.load(f)
        if not session_data:
            return None

        print(f"\n{C_BLUE}Found a previous session with the following training files:{C_END}")
        for item in session_data:
            print(f" - {C_YELLOW}{item['local_name']}{C_END}")

        choice = input(f"{C_BOLD}Would you like to reuse these files and skip re-uploading? (y/n): {C_END}").lower()

        if choice == 'y':
            print("Reusing session files. Verifying with Google's servers...")
            reused_files = [genai.get_file(name=item['server_id']) for item in session_data]
            print(f"{C_GREEN}[+] Successfully restored and verified {len(reused_files)} file(s) from the previous session.{C_END}")
            return reused_files
        else:
            print("Okay, starting a new session. The old session file will be overwritten on completion.")
            return None
    except Exception as e:
        print(f"{C_RED}[WARNING] Could not load previous session: {e}. Starting a new one.{C_END}")
        return None

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

def process_single_worksheet(ws_path, training_files, model):
    """
    Processes a single worksheet PDF. This function is designed to be run in a separate thread.
    Returns a tuple of (status, source_path, message/duration).
    """
    item_start_time = time.time()
    ws_name = ws_path.name
    output_path = ws_path.with_suffix('.key.html')

    if output_path.exists():
        return ('skipped', ws_name, f"Output file already exists.")

    try:
        worksheet_file = upload_files_with_retry([ws_path])[0]

        prompt = get_system_prompt(ws_name)
        api_request_files = training_files + [worksheet_file]

        print(f"[*] Generating answer key for {C_BOLD}{ws_name}{C_END} with {C_BLUE}Gemini 2.5 Pro{C_END}...")
        
        try:
            response = model.generate_content([prompt] + api_request_files)
            if not response.parts:
                return('failed', ws_name, "Model returned an empty response. This may be due to a safety filter or content policy.")
            html_content = response.text
        except Exception as api_error:
            return ('failed', ws_name, f"API Error: {api_error}")

        html_content = html_content.strip()
        # Clean up markdown code block fences if present at the start/end of the response
        if html_content.startswith("```html"):
            html_content = html_content.removeprefix("```html").strip()
        if html_content.endswith("```"):
            html_content = html_content.removesuffix("```").strip()
        
        # Basic check to ensure it's valid HTML
        if "<!DOCTYPE html>" not in html_content or "</body>" not in html_content:
            return ('failed', ws_name, f"Model produced malformed HTML. Response snippet: {html_content[:500]}...") # Increased snippet length

        output_path.write_text(html_content, encoding='utf-8')
        
        item_duration = time.time() - item_start_time
        return ('success', ws_name, item_duration)

    except Exception as e:
        return ('failed', ws_name, str(e))

def main():
    script_start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Generate HTML answer keys for worksheets based on training data PDFs using parallel processing.",
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

    training_pdf_paths = list(training_path.glob("*.pdf"))
    training_files = []
    
    # --- Session Management Logic ---
    training_files = load_session()
    if training_files is None:
        # --- Phase 1: Uploading Training Material ---
        print(f"\n{C_BLUE}{C_BOLD}--- Phase 1: Uploading New Training Material ---{C_END}")
        if not training_pdf_paths:
            print(f"{C_YELLOW}[WARNING] No training PDFs found. The AI will use its general knowledge.{C_END}")
        else:
            try:
                training_files = upload_files_with_retry(training_pdf_paths)
                print(f"{C_GREEN}[+] Successfully uploaded {len(training_files)} new training file(s).{C_END}")
                save_session(training_pdf_paths, training_files)
            except Exception as e:
                print(f"{C_RED}[FATAL] Could not upload training files. Aborting. Error: {e}{C_END}")
                return

    print(f"\n{C_BLUE}{C_BOLD}--- Phase 2: Processing Worksheets in Parallel (max {args.max_workers} at a time) ---{C_END}")
    worksheet_pdf_paths = list(worksheets_path.glob("*.pdf"))
    total_worksheets = len(worksheet_pdf_paths)

    if not worksheet_pdf_paths:
        print("No worksheet PDFs found to process.")
        return

    # Use the gemini-2.5-pro model
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    processing_times = []
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_ws = {executor.submit(process_single_worksheet, ws_path, training_files, model): ws_path for ws_path in worksheet_pdf_paths}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_ws)):
            ws_path = future_to_ws[future]
            print(f"[{i+1}/{total_worksheets}] Processing result for {C_BOLD}{ws_path.name}{C_END}...")
            try:
                status, ws_name, result = future.result()
                if status == 'success':
                    duration = result
                    processing_times.append(duration)
                    success_count += 1
                    print(f"  -> {C_GREEN}[SUCCESS] {C_BOLD}{ws_name}{C_END} processed in {format_time(duration)}.")
                elif status == 'skipped':
                    skipped_count += 1
                    print(f"  -> {C_BLUE}[SKIPPED] {C_BOLD}{ws_name}{C_END}. Reason: {result}")
                elif status == 'failed':
                    failed_count += 1
                    print(f"  -> {C_RED}[FAILED]  {C_BOLD}{ws_name}{C_END}. Reason: {result}")
            except Exception as exc:
                failed_count += 1
                print(f"  -> {C_RED}[ERROR]   {C_BOLD}{ws_path.name}{C_END} generated an unexpected exception: {exc}")

    script_end_time = time.time()
    total_duration = script_end_time - script_start_time

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
