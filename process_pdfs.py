--- START OF FILE process_pdfs.py ---

import google.generativeai as genai
import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
import time
import concurrent.futures
import json
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
import shutil

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
    print(f"{C_BLUE}[DEBUG] Attempting to configure AI...{C_END}") ### DEBUG ###
    load_dotenv(override=True)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(f"{C_RED}Google API Key not found. Please set the GOOGLE_API_KEY in your .env file.{C_END}")
    genai.configure(api_key=api_key)
    print(f"{C_GREEN}[DEBUG] AI configured successfully.{C_END}") ### DEBUG ###

def upload_files_with_retry(file_paths, max_retries=3):
    uploaded_files = []
    # Use a dictionary to keep track of original paths for uploaded files
    path_map = {path.resolve(): path for path in file_paths}
    resolved_paths = list(path_map.keys())

    for path in resolved_paths:
        original_path = path_map[path]
        for attempt in range(max_retries):
            try:
                print(f" - Uploading: {C_YELLOW}{original_path.name}{C_END}...")
                uploaded_file = genai.upload_file(path=original_path, display_name=original_path.name)
                # Store the original path alongside the uploaded file object
                uploaded_files.append({'file': uploaded_file, 'path': original_path})
                break
            except Exception as e:
                print(f" {C_RED}[!] Attempt {attempt + 1} failed for {original_path.name}: {e}{C_END}")
                if attempt + 1 == max_retries:
                    print(f" {C_RED}[!] Failed to upload {original_path.name} after {max_retries} attempts.{C_END}")
                    raise
                time.sleep(2 ** attempt)
    return uploaded_files

def manage_training_files(training_pdf_paths):
    """
    Manages training files by loading known files from a session,
    uploading new ones, and saving the updated cumulative session.
    Returns a list of verified `genai.File` objects.
    """
    session_data = {}
    if Path(SESSION_FILE).exists():
        try:
            with open(SESSION_FILE, "r") as f:
                session_data = json.load(f)
            # --- START: ROBUSTNESS FIX ---
            # Verify that the loaded data is a dictionary. If not, treat as empty.
            if not isinstance(session_data, dict):
                print(f"{C_YELLOW}[WARNING] Session file '{SESSION_FILE}' was malformed (not a dictionary). "
                      f"A new session will be created and all training files will be re-uploaded.{C_END}")
                session_data = {}
            # --- END: ROBUSTNESS FIX ---
        except (json.JSONDecodeError, IOError) as e:
            print(f"{C_YELLOW}[WARNING] Could not read session file {SESSION_FILE}: {e}. Starting fresh.{C_END}")
            session_data = {}

    local_filenames_in_session = set(session_data.keys())
    all_current_local_paths = {p.name: p for p in training_pdf_paths}

    files_to_upload_paths = [p for name, p in all_current_local_paths.items() if name not in local_filenames_in_session]
    known_files_to_verify = {name: server_id for name, server_id in session_data.items() if name in all_current_local_paths}

    verified_files = []
    newly_uploaded_files = [] # <<< FIX: Initialize list here to ensure it always exists.

    # --- Verify existing files ---
    if known_files_to_verify:
        print(f"\n{C_BLUE}--- Verifying Previously Uploaded Training Files ---{C_END}")
        for local_name, server_id in known_files_to_verify.items():
            try:
                print(f" - Verifying: {C_YELLOW}{local_name}{C_END}...")
                verified_file = genai.get_file(name=server_id)
                verified_files.append(verified_file)
            except Exception as e:
                print(f" {C_YELLOW}[!] Could not verify '{local_name}' (ID: {server_id}). It will be re-uploaded. Reason: {e}{C_END}")
                files_to_upload_paths.append(all_current_local_paths[local_name])
                if local_name in session_data:
                    del session_data[local_name]
        
        if verified_files:
             print(f"{C_GREEN}[+] Verified and reused {len(verified_files)} file(s).{C_END}")

    # --- Upload new files ---
    if files_to_upload_paths:
        print(f"\n{C_BLUE}{C_BOLD}--- Uploading New/Updated Training Material ---{C_END}")
        try:
            newly_uploaded_file_info = upload_files_with_retry(files_to_upload_paths)
            
            for info in newly_uploaded_file_info:
                uploaded_file = info['file']
                original_path = info['path']
                newly_uploaded_files.append(uploaded_file)
                session_data[original_path.name] = uploaded_file.name

            if newly_uploaded_files:
                 print(f"{C_GREEN}[+] Successfully uploaded {len(newly_uploaded_files)} new/updated training file(s).{C_END}")

        except Exception as e:
            print(f"{C_RED}[FATAL] Could not upload training files. Aborting. Error: {e}{C_END}")
            raise

    # --- Save the updated session ---
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(session_data, f, indent=4)
        print(f"{C_GREEN}[i] Cumulative session data updated in {SESSION_FILE}.{C_END}")
    except Exception as e:
        print(f"{C_YELLOW}[WARNING] Could not save session data: {e}{C_END}")

    return verified_files + newly_uploaded_files


def split_pdf(pdf_path, output_dir):
    """Splits a PDF into single pages."""
    output_dir.mkdir(exist_ok=True)
    pdf_document = fitz.open(pdf_path)
    page_paths = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        output_path = output_dir / f"page_{page_num + 1}.pdf"
        new_pdf = fitz.open()
        new_pdf.insert_pdf(pdf_document, from_page=page_num, to_page=page_num)
        new_pdf.save(str(output_path))
        new_pdf.close()
        page_paths.append(output_path)
    pdf_document.close()
    return page_paths

def merge_html_files(html_files, output_path, worksheet_name):
    """Merges multiple HTML files into a single file."""
    if not html_files:
        return

    # Use the first HTML file as the base template
    with open(html_files[0], 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find the main container to append content to
    main_container = soup.find('div', class_='container')
    if not main_container:
        # If no container, use the body as a fallback
        main_container = soup.body

    # Clear existing generated content from the template before merging
    if main_container:
        for element in main_container.find_all(["h3", "div", "hr"]):
             if 'problem-statement' in element.get('class', []) or \
                'solution' in element.get('class', []) or \
                'diagram-container' in element.get('class', []) or \
                element.name == 'hr' or element.name == 'h3':
                 element.decompose()


    # Iterate through all HTML files (including the first) and append their content
    for html_file in html_files:
        with open(html_file, 'r', encoding='utf-8') as f:
            page_soup = BeautifulSoup(f, 'html.parser')
            # Find the content within the container of the page
            page_content_container = page_soup.find('div', class_='container')
            if page_content_container:
                 # Append all children of the container to the main soup
                 for child in page_content_container.children:
                     if child.name: # Ensure it's a tag, not just a string
                         main_container.append(child)

    # Update the title and main heading
    if soup.title:
        soup.title.string = f"Answer Key: {worksheet_name}"
    h2_heading = soup.find('h2')
    if h2_heading:
        h2_heading.string = f"Answer Key for {worksheet_name}"


    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(str(soup.prettify()))

def process_single_worksheet(ws_path, training_files, model, temp_dir):
    """
    Processes a single worksheet PDF by splitting it into pages,
    generating HTML for each page, and then merging the results.
    """
    item_start_time = time.time()
    ws_name = ws_path.name
    output_path = ws_path.with_suffix('.key.html')

    if output_path.exists():
        return ('skipped', ws_name, "Output file already exists.")

    try:
        # Create a temporary directory for this worksheet's pages
        worksheet_temp_dir = temp_dir / ws_path.stem
        worksheet_temp_dir.mkdir(exist_ok=True)

        # 1. Split the PDF into pages
        page_paths = split_pdf(ws_path, worksheet_temp_dir)
        
        generated_html_files = []
        uploaded_page_files = []


        # 2. Upload all pages first
        print(f"[*] Pre-uploading all {len(page_paths)} pages for {C_BOLD}{ws_name}{C_END}...")
        try:
            # upload_files_with_retry returns a list of dictionaries
            uploaded_info = upload_files_with_retry(page_paths)
            # We only need the genai.File object for the API call
            uploaded_page_files = [info['file'] for info in uploaded_info]
        except Exception as e:
            return ('failed', ws_name, f"Could not pre-upload pages. Error: {e}")


        # 3. Process each page sequentially
        for i, (page_path, page_file) in enumerate(zip(page_paths, uploaded_page_files)):
            page_name = page_path.name
            page_output_path = worksheet_temp_dir / f"page_{i+1}.html"

            try:
                prompt = get_system_prompt(f"{ws_name} - Page {i+1}")
                # Use the pre-uploaded file object
                api_request_files = training_files + [page_file]

                print(f"[*] Generating answer key for {C_BOLD}{ws_name} (Page {i+1}/{len(page_paths)}){C_END}...")
                
                try:
                    response = model.generate_content(api_request_files) # Removed stream=True
                    if not response.parts:
                         print(f"  -> {C_YELLOW}[WARNING] Model returned an empty response for page {i+1}.{C_END}")
                         continue

                    html_content = response.text
                except Exception as api_error:
                    print(f"  -> {C_RED}[FAILED] API Error for page {i+1}: {api_error}{C_END}")
                    continue

                html_content = html_content.strip()
                if html_content.startswith("```html"):
                    html_content = html_content.removeprefix("```html").strip()
                if html_content.endswith("```"):
                    html_content = html_content.removesuffix("```").strip()
                
                if "<!DOCTYPE html>" not in html_content or "</body>" not in html_content:
                    print(f"  -> {C_YELLOW}[WARNING] Model produced malformed HTML for page {i+1}. Snippet: {html_content[:200]}...{C_END}")
                    continue

                page_output_path.write_text(html_content, encoding='utf-8')
                generated_html_files.append(page_output_path)

            except Exception as e:
                print(f"  -> {C_RED}[FAILED] Could not process page {i+1}: {e}{C_END}")
                continue
        
        # 4. Merge the generated HTML files
        if generated_html_files:
            print(f"[*] Merging {len(generated_html_files)} HTML page(s) into final answer key for {C_BOLD}{ws_name}{C_END}...")
            merge_html_files(generated_html_files, output_path, ws_name)
            item_duration = time.time() - item_start_time
            return ('success', ws_name, item_duration)
        else:
            return ('failed', ws_name, "No HTML pages were successfully generated.")

    except Exception as e:
        return ('failed', ws_name, str(e))
    finally:
        # 5. Clean up temporary files
        if 'worksheet_temp_dir' in locals() and worksheet_temp_dir.exists():
            shutil.rmtree(worksheet_temp_dir)


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
    
    ### DEBUG ###
    print(f"{C_BLUE}{C_BOLD}[DEBUG] Script starting...{C_END}")
    print(f"{C_BLUE}[DEBUG] Arguments received:{C_END}")
    print(f"{C_BLUE}[DEBUG]   -> Training Folder: {args.training_folder}{C_END}")
    print(f"{C_BLUE}[DEBUG]   -> Worksheets Folder: {args.worksheets_folder}{C_END}")
    print(f"{C_BLUE}[DEBUG]   -> Max Workers: {args.max_workers}{C_END}")
    ### END DEBUG ###

    try:
        configure_ai()
    except ValueError as e:
        print(e)
        return

    training_path = Path(args.training_folder).resolve() ### DEBUG: Use resolve() for absolute path ###
    worksheets_path = Path(args.worksheets_folder).resolve() ### DEBUG: Use resolve() for absolute path ###
    temp_dir = Path("./temp_processing_files")

    ### DEBUG ###
    print(f"{C_BLUE}[DEBUG] Resolved Training Path: {training_path}{C_END}")
    print(f"{C_BLUE}[DEBUG] Resolved Worksheets Path: {worksheets_path}{C_END}")
    ### END DEBUG ###

    if not training_path.is_dir() or not worksheets_path.is_dir():
        print(f"{C_RED}[ERROR] One or both provided paths are not valid directories.{C_END}")
        if not training_path.is_dir():
             print(f"{C_RED}[ERROR]   -> Path not found or not a directory: {training_path}{C_END}")
        if not worksheets_path.is_dir():
             print(f"{C_RED}[ERROR]   -> Path not found or not a directory: {worksheets_path}{C_END}")
        return
        
    temp_dir.mkdir(exist_ok=True)

    training_pdf_paths = list(training_path.glob("*.pdf"))
    
    ### DEBUG ###
    print(f"{C_BLUE}[DEBUG] Found {len(training_pdf_paths)} PDF(s) in the training folder.{C_END}")
    for pdf in training_pdf_paths:
        print(f"{C_BLUE}[DEBUG]   -> Found: {pdf.name}{C_END}")
    ### END DEBUG ###

    # --- Phase 1: Cumulative Session and Training File Management ---
    try:
        training_files = manage_training_files(training_pdf_paths)
        if not training_pdf_paths:
            print(f"{C_YELLOW}[WARNING] No training PDFs found in {args.training_folder}. The AI will use its general knowledge.{C_END}")
        elif not training_files:
             print(f"{C_RED}[ERROR] No training files were successfully uploaded or verified. Aborting.{C_END}")
             shutil.rmtree(temp_dir)
             return
    except Exception as e:
        print(f"{C_RED}[FATAL ERROR] An exception occurred during training file management: {e}{C_END}") ### DEBUG ###
        shutil.rmtree(temp_dir)
        return

    print(f"\n{C_BLUE}{C_BOLD}--- Phase 2: Processing Worksheets in Parallel (max {args.max_workers} at a time) ---{C_END}")
    worksheet_pdf_paths = list(worksheets_path.glob("*.pdf"))
    total_worksheets = len(worksheet_pdf_paths)
    
    ### DEBUG ###
    print(f"{C_BLUE}[DEBUG] Found {len(worksheet_pdf_paths)} PDF(s) in the worksheets folder.{C_END}")
    for pdf in worksheet_pdf_paths:
        print(f"{C_BLUE}[DEBUG]   -> Found: {pdf.name}{C_END}")
    ### END DEBUG ###

    if not worksheet_pdf_paths:
        print("No worksheet PDFs found to process. Exiting.") ### DEBUG: More explicit message ###
        shutil.rmtree(temp_dir)
        return

    # Use the gemini-2.5-pro model
    model = genai.GenerativeModel('models/gemini-2.5-pro')
    
    processing_times = []
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_ws = {executor.submit(process_single_worksheet, ws_path, training_files, model, temp_dir): ws_path for ws_path in worksheet_pdf_paths}

        for i, future in enumerate(concurrent.futures.as_completed(future_to_ws)):
            ws_path = future_to_ws[future]
            print(f"\n[{i+1}/{total_worksheets}] Finalizing result for {C_BOLD}{ws_path.name}{C_END}...")
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

    # Clean up the main temporary directory
    if temp_dir.exists():
        print(f"{C_BLUE}[DEBUG] Cleaning up temporary directory: {temp_dir}{C_END}") ### DEBUG ###
        shutil.rmtree(temp_dir)

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
    print(f"{C_BLUE}{C_BOLD}[DEBUG] Script finished.{C_END}") ### DEBUG ###

if __name__ == "__main__":
    main()
