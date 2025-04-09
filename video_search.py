"""
video_search.py (renamed as main25.py)

A FastAPI application that:
  - Performs 12Labs video search
  - Allows scene selection in a cart that persists across filtering/pagination
    (only flushes if the query changes)
  - Provides a floating video preview on the Select Timepoints page
  - Generates aggregated line or box graphs with CSV data, saved as a downloadable file
  - Displays a Neuro-Insight logo (remote URL) on the home page
  - For the line aggregator CSV, time is in milliseconds in column A,
    and measures appear horizontally across row 1 to mimic the original CSV style.
"""

import os
import math
import csv
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import base64
from fastapi import FastAPI, HTTPException, Query, Form
from fastapi.responses import HTMLResponse
from twelvelabs import TwelveLabs
from html import escape  # Used for escaping HTML special characters

app = FastAPI()

# ------------------------------
# Helper function for safe filenames
# ------------------------------
def safe_filename(query: str) -> str:
    """
    Returns a file-name–safe version of the query by keeping alphanumerics and _ or -
    and replacing other characters with underscores.
    """
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in query)

# ------------------------------
# 12Labs and CSV CONFIGURATION
# ------------------------------
client = TwelveLabs(api_key="tlk_3QAYDJ91QAGJSP2Q575VE0HSFRY6")
INDEX_ID = "67a949247f2be71ef38c2149"
CSV_FOLDER = "csv_data"

# Here we define the exact measure order we want horizontally:
ORDERED_MEASURES = [
    "Approach / Withdraw",
    "Engagement",
    "Emotional Intensity",
    "Memory Encoding - Detail",
    "Memory Encoding - Global",
    "Memory Encoding - Composite",
    "General Attention - Detail",
    "General Attention - Global",
    "General Attention - Composite",
    "Visual Attention - Detail",
    "Visual Attention - Global",
    "Visual Attention - Composite"
]

MEASURE_COLUMNS = ORDERED_MEASURES  # same set, reused

# ------------------------------
# Azure Blob Storage SAS SETTINGS
# ------------------------------
AZURE_BLOB_BASE_URL = "https://neuroinsightmp4s.blob.core.windows.net/tvcmp4s"
AZURE_SAS_TOKEN = (
    "sp=rl&st=2025-03-25T02:52:43Z&se=2026-03-25T10:52:43Z&spr=https"
    "&sv=2024-11-04&sr=c&sig=%2BcEVFsLbeU5A%2FAFqNzg8QUgMT3MFNWMh0v0Vs%2F7vbsQ%3D"
)

def get_computed_confidence(score: float) -> str:
    if score >= 80:
        return "high"
    elif score >= 75:
        return "medium"
    else:
        return "low"

def flatten_clips(grouped):
    flat = []
    for item in grouped:
        if hasattr(item, "clips"):
            flat.extend(item.clips)
        else:
            flat.append(item)
    return flat

def gather_all_clips(query: str):
    try:
        from twelvelabs import TwelveLabsError  # Just in case a specific error type is needed
    except ImportError:
        TwelveLabsError = Exception

    try:
        results = client.search.query(
            index_id=INDEX_ID,
            query_text=query,
            options=["visual", "audio"],
            operator="or",
            threshold="low",
            group_by="video",
            sort_option="score",
            adjust_confidence_level=0.5,
            page_limit=50
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    all_clips = flatten_clips(results.data)
    try:
        total_hits = results.page_info.total_results
    except Exception:
        total_hits = len(all_clips)
    while getattr(results.page_info, "next_page_token", None):
        token = results.page_info.next_page_token
        try:
            results = client.search.by_page_token(page_token=token)
            all_clips.extend(flatten_clips(results.data))
        except Exception:
            break
    all_clips.sort(key=lambda clip: clip.score, reverse=True)
    return all_clips, total_hits

def paginate(items, page: int, per_page: int):
    total = len(items)
    start = (page - 1) * per_page
    end = start + per_page
    return items[start:end], total

def render_clip(clip) -> str:
    video_id = clip.video_id
    clip_start = clip.start
    clip_end = clip.end
    computed = get_computed_confidence(clip.score)
    thumbnail_url = getattr(clip, "thumbnail_url", "") or ""
    ad_name = "Unknown"

    csv_filename = os.path.join(CSV_FOLDER, f"{video_id}.csv")
    if os.path.exists(csv_filename):
        try:
            header_row = pd.read_csv(csv_filename, header=None, nrows=1)
            for col in range(1, 13):
                val = header_row.iat[0, col]
                if pd.notnull(val) and str(val).strip() != "":
                    ad_name = str(val).strip()
                    break
        except Exception:
            ad_name = "Unknown"

    checkbox_value = f"{video_id}|{clip_start}|{clip_end}|{thumbnail_url}|{ad_name}"

    html = (
        f"<li id='clip-{video_id}-{clip_start}-{clip_end}' "
        f"style='padding:5px; border:1px solid #ddd; margin-bottom:5px;'>"
    )
    html += (
        f"<input type='checkbox' style='transform: scale(1.5); margin-right:5px;' "
        f"name='selected_clips' value='{checkbox_value}' onchange='toggleHighlight(this)'> "
    )
    html += (
        f"Video ID: {video_id} | Score: {clip.score:.2f} | "
        f"Start: {clip_start} | End: {clip_end} | Confidence: {computed}<br>"
    )

    if thumbnail_url:
        html += (
            f'<img id="thumbnail-{video_id}-{clip_start}-{clip_end}" src="{thumbnail_url}" '
            f'alt="Thumbnail" style="width:200px; cursor:pointer;" '
            f'onclick="previewHighlight(\'{video_id}-{clip_start}-{clip_end}\', '
            f'\'{AZURE_BLOB_BASE_URL}/{video_id}.mp4?{AZURE_SAS_TOKEN}#t={clip_start},{clip_end}\')"><br>'
        )
    playback_url = f"{AZURE_BLOB_BASE_URL}/{video_id}.mp4?{AZURE_SAS_TOKEN}#t={clip_start},{clip_end}"
    html += f'<a href="{playback_url}" target="_blank">Play in New Tab</a><br>'
    html += f"<strong>Ad Name:</strong> {ad_name}<br>"

    if os.path.exists(csv_filename):
        try:
            df = pd.read_csv(csv_filename, header=1)
            df.rename(columns={df.columns[0]: "Time"}, inplace=True)
            csv_max_time = df["Time"].max()
            if abs(clip_end - csv_max_time) <= 0.05:
                clip_end = csv_max_time
            segment = df[(df["Time"] >= clip_start) & (df["Time"] <= clip_end)]
            if not segment.empty:
                stats = {}
                for col in ORDERED_MEASURES:
                    if col in df.columns:
                        stats[col] = {
                            "min": segment[col].min(),
                            "max": segment[col].max(),
                            "avg": segment[col].mean(),
                            "std": segment[col].std()
                        }
                if stats:
                    html += "<br><strong>CSV Metrics:</strong><br>"
                    for measure, values in stats.items():
                        html += (
                            f"{measure}: Min: {values['min']:.2f}, "
                            f"Max: {values['max']:.2f}, "
                            f"Avg: {values['avg']:.2f}, "
                            f"Std: {values['std']:.2f}<br>"
                        )
                else:
                    html += "No measure columns found in CSV.<br>"
            else:
                html += "No CSV data found in this segment.<br>"
        except Exception as csv_err:
            html += f"CSV Error: {csv_err}<br>"
    else:
        html += "CSV file not found.<br>"

    html += "</li>"
    return html

@app.get("/", response_class=HTMLResponse)
def home_page():
    return HTMLResponse(content=f"""
    <html>
      <head>
        <title>12Labs Video Search</title>
        <style>
          body {{
            font-family: Arial, sans-serif;
            text-align: center;
            margin-top: 50px;
          }}
          .logo-container {{
            margin-bottom: 20px;
          }}
        </style>
      </head>
      <body>
        <div class="logo-container">
          <img src="https://www.neuro-insight.com/wp-content/uploads/2023/01/footer-site-logo-300x214.png"
               alt="Neuro-Insight Logo" style="width:150px;">
        </div>
        <h1 style="margin-bottom: 30px;">12Labs Video Search</h1>
        <form action="/search" method="get">
          <label for="query">Enter your search query (e.g., "man"):</label><br>
          <input type="text" id="query" name="query" size="50" required><br><br>
          <input type="submit" value="Search">
        </form>
      </body>
    </html>
    """)

cache = {}

@app.get("/search", response_class=HTMLResponse)
def search_results(
    query: str = Query(...),
    conf_filter: str = Query("all"),
    page: int = Query(1, ge=1)
):
    if query not in cache:
        clips, total_hits = gather_all_clips(query)
        cache[query] = {"clips": clips, "total_hits": total_hits}
    else:
        clips = cache[query]["clips"]
        total_hits = cache[query]["total_hits"]

    all_count = len(clips)
    high_count = sum(1 for c in clips if get_computed_confidence(c.score) == "high")
    medium_count = sum(1 for c in clips if get_computed_confidence(c.score) == "medium")
    low_count = sum(1 for c in clips if get_computed_confidence(c.score) == "low")

    if conf_filter.lower() == "all":
        filtered = clips
    else:
        filtered = [c for c in clips if get_computed_confidence(c.score) == conf_filter.lower()]

    per_page = 50
    page_results, filtered_total = paginate(filtered, page, per_page)
    total_pages = math.ceil(filtered_total / per_page)

    filter_form = (
        f'<form action="/search" method="get">'
        f'<input type="hidden" name="query" value="{escape(query)}">'
        f'<label for="conf_filter">Filter by Confidence: </label>'
        f'<select name="conf_filter">'
        f'<option value="all" {"selected" if conf_filter.lower() == "all" else ""}>Show All ({all_count})</option>'
        f'<option value="high" {"selected" if conf_filter.lower() == "high" else ""}>High ({high_count})</option>'
        f'<option value="medium" {"selected" if conf_filter.lower() == "medium" else ""}>Medium ({medium_count})</option>'
        f'<option value="low" {"selected" if conf_filter.lower() == "low" else ""}>Low ({low_count})</option>'
        f'</select>'
        f'<input type="submit" value="Filter">'
        f'</form>'
    )

    nav_links = ""
    if page > 1:
        nav_links += f'<a href="/search?query={escape(query)}&conf_filter={conf_filter}&page=1">Skip to Start</a> | '
        nav_links += f'<a href="/search?query={escape(query)}&conf_filter={conf_filter}&page={page-1}">Previous</a> | '
    if page < total_pages:
        nav_links += f'<a href="/search?query={escape(query)}&conf_filter={conf_filter}&page={page+1}">Next</a> | '
        nav_links += f'<a href="/search?query={escape(query)}&conf_filter={conf_filter}&page={total_pages}">Skip to End</a>'

    query_check_script = f"""
    <script>
    function checkQueryChange(currentQuery) {{
        var oldQuery = sessionStorage.getItem("lastQuery");
        if (oldQuery !== currentQuery) {{
            sessionStorage.removeItem("selectedClips");
        }}
        sessionStorage.setItem("lastQuery", currentQuery);
    }}
    function loadStoredSelections() {{
        var stored = getStoredSelection();
        var checkboxes = document.querySelectorAll("input[name='selected_clips']");
        checkboxes.forEach(function(cb) {{
            if (stored.indexOf(cb.value) !== -1) {{
                cb.checked = true;
                cb.parentNode.style.backgroundColor = "#e0ffe0";
            }}
        }});
    }}
    window.onload = function() {{
        checkQueryChange("{escape(query)}");
        updateCartDisplay();
        loadStoredSelections();
    }};
    </script>
    """

    html = "<html><head><title>Search Results</title>" + query_check_script
    html += """
    <style>
    li { transition: background-color 0.3s; }
    /* Modified cart container: set a max-height and enable vertical scrolling */
    #cartContainer {
        position: fixed;
        top: 10px;
        right: 10px;
        width: 300px;
        max-height: 300px;
        overflow-y: auto;
        background-color: #f9f9f9;
        border: 1px solid #ccc;
        padding: 10px;
        z-index: 999;
    }
    #cartItems p { margin: 0 0 5px 0; }
    </style>
    <script>
    function getStoredSelection() {
        var saved = sessionStorage.getItem("selectedClips");
        return saved ? JSON.parse(saved) : [];
    }
    function setStoredSelection(arr) {
        sessionStorage.setItem("selectedClips", JSON.stringify(arr));
    }
    function toggleHighlight(checkbox) {
        var li = checkbox.parentNode;
        if (checkbox.checked) {
            li.style.backgroundColor = "#e0ffe0";
            var stored = getStoredSelection();
            if (stored.indexOf(checkbox.value) === -1) {
                stored.push(checkbox.value);
                setStoredSelection(stored);
            }
        } else {
            li.style.backgroundColor = "";
            var stored = getStoredSelection();
            var index = stored.indexOf(checkbox.value);
            if (index > -1) {
                stored.splice(index, 1);
                setStoredSelection(stored);
            }
        }
        updateCartDisplay();
    }
    function updateCartDisplay() {
        var stored = getStoredSelection();
        var cartItemsDiv = document.getElementById("cartItems");
        var cartCountSpan = document.getElementById("cartCount");
        cartItemsDiv.innerHTML = "";
        cartCountSpan.textContent = stored.length;
        stored.forEach(function(item) {
            var parts = item.split("|");
            var adName = parts[4] ? parts[4] : parts[0];
            var displayText = "Ad: " + adName + " (" + parts[1] + " - " + parts[2] + ")";
            var p = document.createElement("p");
            p.textContent = displayText + " ";
            var removeBtn = document.createElement("button");
            removeBtn.textContent = "Remove";
            removeBtn.style.marginLeft = "10px";
            removeBtn.onclick = function() {
                removeCartItem(item);
            };
            p.appendChild(removeBtn);
            cartItemsDiv.appendChild(p);
        });
    }
    function removeCartItem(value) {
        var stored = getStoredSelection();
        var index = stored.indexOf(value);
        if (index > -1) {
            stored.splice(index, 1);
            setStoredSelection(stored);
        }
        updateCartDisplay();
    }
    function emptyCart() {
        sessionStorage.removeItem("selectedClips");
        updateCartDisplay();
    }
    function selectAll() {
        var checkboxes = document.querySelectorAll("input[name='selected_clips']");
        var stored = getStoredSelection();
        checkboxes.forEach(function(cb) {
            cb.checked = true;
            cb.parentNode.style.backgroundColor = "#e0ffe0";
            if (stored.indexOf(cb.value) === -1) {
                stored.push(cb.value);
            }
        });
        setStoredSelection(stored);
        updateCartDisplay();
    }
    function deselectAll() {
        var checkboxes = document.querySelectorAll("input[name='selected_clips']");
        checkboxes.forEach(function(cb) {
            cb.checked = false;
            cb.parentNode.style.backgroundColor = "";
        });
        sessionStorage.removeItem("selectedClips");
        updateCartDisplay();
    }
    function previewHighlight(clipId, playbackUrl) {
        var containerId = 'video-container-' + clipId;
        var container = document.getElementById(containerId);
        if (container) {
            container.style.display = container.style.display === 'none' ? 'block' : 'none';
        } else {
            container = document.createElement('div');
            container.setAttribute('id', containerId);
            container.style.border = "1px solid #ccc";
            container.style.padding = "5px";
            container.style.marginTop = "5px";
            container.style.position = "relative";
            var closeBtn = document.createElement('div');
            closeBtn.innerHTML = "X";
            closeBtn.style.position = "absolute";
            closeBtn.style.top = "5px";
            closeBtn.style.right = "5px";
            closeBtn.style.cursor = "pointer";
            closeBtn.style.fontWeight = "bold";
            closeBtn.onclick = function() {
                container.parentNode.removeChild(container);
            };
            container.appendChild(closeBtn);
            var videoElem = document.createElement('video');
            videoElem.setAttribute('controls', 'controls');
            videoElem.setAttribute('width', '320');
            videoElem.setAttribute('height', '240');
            var sourceElem = document.createElement('source');
            sourceElem.setAttribute('src', playbackUrl);
            sourceElem.setAttribute('type', 'video/mp4');
            videoElem.appendChild(sourceElem);
            container.appendChild(videoElem);
            var thumb = document.getElementById('thumbnail-' + clipId);
            if (thumb) {
                thumb.parentNode.insertBefore(container, thumb.nextSibling);
            } else {
                document.body.appendChild(container);
            }
        }
    }
    </script>
    """
    html += "</head><body>"
    html += f"""
    <div id="cartContainer">
      <h2>Selected Scenes (<span id="cartCount">0</span>)</h2>
      <div id="cartItems"></div>
      <button onclick="emptyCart()">Empty Cart</button>
      <form method="post" action="/select_timepoints">
         <input type="hidden" name="cartSelections" id="cartInput" value="">
         <!-- Pass the query to the next endpoint -->
         <input type="hidden" name="query" value="{escape(query)}">
         <button type="button" onclick="document.getElementById('cartInput').value = JSON.stringify(getStoredSelection()); this.form.submit();">
             Select Timepoints for Averages
         </button>
      </form>
    </div>
    """
    html += f"<h1>Results for: {query}</h1>"
    html += filter_form
    html += f"<p><strong>Total Hits:</strong> {total_hits}</p>"
    html += f"<p>Displaying {len(page_results)} of {filtered_total} (Page {page} of {total_pages})</p>"
    html += f"<p>{nav_links}</p>"
    html += "<button type='button' onclick='selectAll()'>Select All</button> "
    html += "<button type='button' onclick='deselectAll()'>Deselect All</button><br><br>"
    html += "<ul style='list-style-type:none; padding:0;'>"
    for clip in page_results:
        html += render_clip(clip)
    html += "</ul>"
    html += '<br><a href="/">Back to Home</a>'
    html += "</body></html>"
    return HTMLResponse(content=html)

@app.get("/update_metrics", response_class=HTMLResponse)
def update_metrics(video_id: str = Query(...), start_time: float = Query(...), end_time: float = Query(...)):
    csv_filename = os.path.join(CSV_FOLDER, f"{video_id}.csv")
    if not os.path.exists(csv_filename):
        return HTMLResponse(content="CSV file not found.", status_code=404)
    try:
        df = pd.read_csv(csv_filename, header=1)
        df.rename(columns={df.columns[0]: "Time"}, inplace=True)
        csv_max_time = df["Time"].max()
        if abs(end_time - csv_max_time) <= 0.05:
            end_time = csv_max_time
        segment = df[(df["Time"] >= start_time) & (df["Time"] <= end_time)]
        if segment.empty:
            return HTMLResponse(content="No CSV data found for these timepoints.")
        html_metrics = "<ul>"
        for col in ORDERED_MEASURES:
            if col in df.columns:
                min_val = segment[col].min()
                max_val = segment[col].max()
                avg_val = segment[col].mean()
                std_val = segment[col].std()
                html_metrics += (
                    f"<li>{col}: Min: {min_val:.2f}, Max: {max_val:.2f}, "
                    f"Avg: {avg_val:.2f}, Std: {std_val:.2f}</li>"
                )
        html_metrics += "</ul>"
        return HTMLResponse(content=html_metrics)
    except Exception as e:
        return HTMLResponse(content=f"Error computing metrics: {e}", status_code=500)

@app.post("/select_timepoints", response_class=HTMLResponse)
def select_timepoints(cartSelections: str = Form(...), query: str = Form(...)):
    flush_cart_script = "<script>sessionStorage.removeItem('selectedClips');</script>"
    try:
        selections = json.loads(cartSelections)
    except Exception:
        selections = []
    if not selections:
        return HTMLResponse(content="No moments selected.", status_code=400)

    html = "<html><head><title>Select Timepoints</title>"
    html += """
    <style>
    .moment {
        border: 1px solid #ccc;
        margin-bottom: 10px;
        padding: 5px;
        width: 100%;
        overflow: auto;
        position: relative;
    }
    .moment img {
        width: 200px;
        float: left;
        margin-right: 10px;
    }
    .videoPreview {
        float: right;
        width: 340px;
        border: 1px solid #ccc;
        padding: 5px;
        position: relative;
        margin-bottom: 10px;
    }
    .videoPreview button.close {
        position: absolute;
        top: 5px;
        right: 5px;
        cursor: pointer;
        font-weight: bold;
    }
    </style>
    <script>
    var AZURE_BLOB_BASE_URL = '""" + AZURE_BLOB_BASE_URL + """';
    var AZURE_SAS_TOKEN = '""" + AZURE_SAS_TOKEN + """';
    var activePreviewContainer = null;

    function updateMetrics(index) {
        var videoId = document.getElementById("video_" + index).value;
        var startTime = document.getElementById("start_" + index).value;
        var endTime = document.getElementById("end_" + index).value;
        var url = "/update_metrics?video_id=" + encodeURIComponent(videoId) +
                  "&start_time=" + encodeURIComponent(startTime) +
                  "&end_time=" + encodeURIComponent(endTime);
        fetch(url)
            .then(response => response.text())
            .then(data => {
                document.getElementById("metrics_" + index).innerHTML = data;
            })
            .catch(error => {
                document.getElementById("metrics_" + index).innerHTML = "Error updating metrics.";
            });
        updatePreview(index);
    }

    function updatePreview(index) {
        if (activePreviewContainer) {
            var videoId = document.getElementById("video_" + index).value;
            var startTime = document.getElementById("start_" + index).value;
            var endTime = document.getElementById("end_" + index).value;
            var playbackUrl = AZURE_BLOB_BASE_URL + "/" + videoId + ".mp4?" + AZURE_SAS_TOKEN + "#t=" + startTime + "," + endTime;
            var videoElem = activePreviewContainer.querySelector("video");
            if (videoElem) {
                var sourceElem = videoElem.querySelector("source");
                if (sourceElem) {
                    sourceElem.setAttribute('src', playbackUrl);
                }
                videoElem.load();
            }
        }
    }

    function removeMoment(index) {
        var elem = document.getElementById("moment_" + index);
        if(elem) { elem.parentNode.removeChild(elem); }
    }

    function previewHighlight(clipId, playbackUrl, index) {
        if (activePreviewContainer) {
            if (activePreviewContainer.parentNode) {
                activePreviewContainer.parentNode.removeChild(activePreviewContainer);
            }
            activePreviewContainer = null;
        }
        var container = document.createElement('div');
        container.className = "videoPreview";
        container.setAttribute('id', 'video-container-' + clipId);

        var closeBtn = document.createElement('button');
        closeBtn.innerHTML = "X";
        closeBtn.className = "close";
        closeBtn.onclick = function() {
            container.parentNode.removeChild(container);
            activePreviewContainer = null;
        };
        container.appendChild(closeBtn);

        var videoElem = document.createElement('video');
        videoElem.setAttribute('controls', 'controls');
        videoElem.setAttribute('width', '320');
        videoElem.setAttribute('height', '240');

        var sourceElem = document.createElement('source');
        sourceElem.setAttribute('src', playbackUrl);
        sourceElem.setAttribute('type', 'video/mp4');
        videoElem.appendChild(sourceElem);
        container.appendChild(videoElem);

        var momentDiv = document.getElementById("moment_" + index);
        if (momentDiv) {
            momentDiv.insertBefore(container, momentDiv.firstChild);
        } else {
            document.body.appendChild(container);
        }
        activePreviewContainer = container;
    }
    </script>
    """ + flush_cart_script
    html += "</head><body>"
    html += "<h1>Select Timepoints for Each Moment</h1>"
    html += f'<h2>Query: {escape(query)}</h2>'
    html += '<a href="javascript:history.back()">Back to Search Results</a><br><br>'
    html += "<form method='post' action='/compute_averages'>"
    html += f"<input type='hidden' name='query' value='{escape(query)}'>"
    html += f"<input type='hidden' id='clip_count' value='{len(selections)}'>"

    for i, clip_str in enumerate(selections):
        parts = clip_str.split("|")
        if len(parts) != 5:
            continue
        video_id, default_start, default_end, thumbnail_url, ad_name = parts
        html += f"<div class='moment' id='moment_{i}'>"
        html += f"<img id='thumbnail-{video_id}-{default_start}-{default_end}' src='{thumbnail_url}' alt='Thumbnail' style='width:200px; float:left; margin-right:10px;'>"
        playback_url = f"{AZURE_BLOB_BASE_URL}/{video_id}.mp4?{AZURE_SAS_TOKEN}#t={default_start},{default_end}"
        html += f'<a href="javascript:void(0)" onclick="previewHighlight(\'{video_id}-{i}\', \'{playback_url}\', {i})">Preview Video</a><br>'
        html += f"<strong>Ad Name:</strong> {ad_name}<br>"
        html += f"<strong>Video ID:</strong> {video_id} - Default Start: {default_start}, Default End: {default_end}<br>"
        html += f"<input type='hidden' name='video_id' id='video_{i}' value='{video_id}'>"
        html += f"<input type='hidden' name='ad_name' value='{ad_name}'>"
        html += f"Start Time: <input type='text' name='start_time' id='start_{i}' value='{default_start}' onchange='updateMetrics({i})'> "
        html += f"End Time: <input type='text' name='end_time' id='end_{i}' value='{default_end}' onchange='updateMetrics({i})'> "
        html += f"<button type='button' onclick='removeMoment({i})'>Remove</button><br>"
        html += f"<div id='metrics_{i}' style='margin-top:5px; clear:both;'></div>"
        html += "</div>"

    html += "<input type='submit' value='Compute Averages'>"
    html += "</form>"
    html += '<br><a href="javascript:history.back()">Back to Search Results</a>'
    html += "</body></html>"
    return HTMLResponse(content=html)

@app.post("/compute_averages", response_class=HTMLResponse)
def compute_averages(
    video_id: list[str] = Form(...),
    start_time: list[float] = Form(...),
    end_time: list[float] = Form(...),
    ad_name: list[str] = Form(...),
    query: str = Form(...)
):
    combined_segments = []
    for i in range(len(video_id)):
        vid = video_id[i]
        try:
            st = float(start_time[i])
            et = float(end_time[i])
        except ValueError:
            continue
        csv_filename = os.path.join(CSV_FOLDER, f"{vid}.csv")
        if not os.path.exists(csv_filename):
            continue
        try:
            df = pd.read_csv(csv_filename, header=1)
            df.rename(columns={df.columns[0]: "Time"}, inplace=True)
            csv_max_time = df["Time"].max()
            if abs(et - csv_max_time) <= 0.05:
                et = csv_max_time
            segment = df[(df["Time"] >= st) & (df["Time"] <= et)]
            if not segment.empty:
                combined_segments.append(segment)
        except Exception:
            continue

    if not combined_segments:
        return HTMLResponse(content="No valid CSV data found for the selected moments.", status_code=400)

    combined_df = pd.concat(combined_segments)
    average_metrics = {}
    for col in ORDERED_MEASURES:
        if col in combined_df.columns:
            average_metrics[col] = combined_df[col].mean()

    html = "<html><head><title>Computed Averages</title></head><body>"
    html += "<h1>Computed Average Metrics</h1>"
    html += f"<h2>Query: {escape(query)}</h2>"
    html += "<ul>"
    for measure, avg_val in average_metrics.items():
        html += f"<li>{measure}: {avg_val:.2f}</li>"
    html += "</ul>"

    durations = []
    for i in range(len(video_id)):
        try:
            st = float(start_time[i])
            et = float(end_time[i])
            durations.append(et - st)
        except:
            pass
    pure_duration = min(durations) if durations else 0

    html += f"""
    <h2>Aggregated Results (Graph Selection)</h2>
    <p><strong>Pure Event Duration:</strong> {pure_duration:.2f} seconds</p>
    <form method="post" action="/aggregated_graphs">
        <label for="graph_type">Select Graph Type:</label><br>
        <input type="radio" id="box" name="graph_type" value="box" checked>
        <label for="box">Box and Whisker (Pure Data)</label><br>
        <input type="radio" id="line" name="graph_type" value="line">
        <label for="line">Line Graph (with Pre/Post Sliders)</label><br><br>
    """
    html += f'<input type="hidden" name="query" value="{escape(query)}">'
    for i in range(len(video_id)):
        html += f'<input type="hidden" name="video_id" value="{video_id[i]}">'
        html += f'<input type="hidden" name="start_time" value="{start_time[i]}">'
        html += f'<input type="hidden" name="end_time" value="{end_time[i]}">'
        html += f'<input type="hidden" name="ad_name" value="{ad_name[i]}">'
    html += """
        <div id="line_options" style="display:none;">
            <label for="pre_duration">Pre-highlight Duration (seconds, 0-10):</label>
            <input type="range" id="pre_duration" name="pre_duration" min="0" max="10" step="0.1" value="0"
                   oninput="this.nextElementSibling.value = this.value">
            <output>0.00</output><br>
            <label for="post_extra">Post-highlight Extra Duration (seconds, 0-10):</label>
            <input type="range" id="post_extra" name="post_extra" min="0" max="10" step="0.1" value="0"
                   oninput="this.nextElementSibling.value = this.value">
            <output>0.00</output><br>
        </div>
        <br><input type="submit" value="View Aggregated Graphs">
    </form>
    <script>
      const radios = document.getElementsByName('graph_type');
      radios.forEach(radio => {
        radio.addEventListener('change', function() {
          document.getElementById('line_options').style.display = (this.value === 'line') ? 'block' : 'none';
        });
      });
    </script>
    """
    html += '<br><button onclick="javascript:history.back()">Back to Select Timepoints</button>'
    html += "<br><a href='/'>Back to Home</a>"
    html += "</body></html>"
    return HTMLResponse(content=html)

@app.post("/aggregated_graphs", response_class=HTMLResponse)
def aggregated_graphs(
    graph_type: str = Form(...),
    video_id: list[str] = Form(...),
    start_time: list[float] = Form(...),
    end_time: list[float] = Form(...),
    ad_name: list[str] = Form(...),
    pre_duration: float = Form(0),
    post_extra: float = Form(0),
    query: str = Form(...)
):
    durations = []
    for i in range(len(video_id)):
        try:
            st = float(start_time[i])
            et = float(end_time[i])
            durations.append(et - st)
        except:
            pass
    pure_duration = min(durations) if durations else 0
    if graph_type == "line":
        return aggregated_line(video_id, start_time, end_time, ad_name, pre_duration, post_extra, pure_duration, query)
    else:
        return aggregated_box(video_id, start_time, end_time, ad_name, pure_duration, query)

def aggregated_line(video_id, start_time, end_time, ad_name, pre_duration, post_extra, pure_duration, query):
    """
    Renders line graphs from -pre_duration to (pure_duration+post_extra) for each measure,
    and creates a CSV (comma-delimited) with separate cells for each column.
    The CSV file name is based on the query.
    """
    effective_post = pure_duration + post_extra
    common_time = np.linspace(-pre_duration, effective_post, 100)
    tol = 0.05
    included_events = []
    excluded_names = []

    for i in range(len(video_id)):
        try:
            st = float(start_time[i])
            et = float(end_time[i])
        except:
            continue

        csv_filename = os.path.join(CSV_FOLDER, f"{video_id[i]}.csv")
        if not os.path.exists(csv_filename):
            continue
        try:
            df = pd.read_csv(csv_filename, header=1)
            df.rename(columns={df.columns[0]: "Time"}, inplace=True)
            available_start = df["Time"].min()
            available_end = df["Time"].max()

            if abs(et - available_end) <= tol:
                et = available_end

            # Exclude if insufficient data
            if (pre_duration > 0 or post_extra > 0):
                if (st - pre_duration) < (available_start - tol) or (st + effective_post) > (available_end + tol):
                    excluded_names.append(ad_name[i])
                    continue

            seg = df[(df["Time"] >= (st - pre_duration)) & (df["Time"] <= (st + effective_post))].copy()
            seg["RelativeTime"] = seg["Time"] - st
            included_events.append(seg)
        except:
            excluded_names.append(ad_name[i])

    if not included_events:
        return HTMLResponse(content="No valid CSV data for graphing after exclusions.", status_code=400)

    # Interpolate data
    interp_data = {}
    for measure in ORDERED_MEASURES:
        interp_data[measure] = []

    for seg in included_events:
        times = seg["RelativeTime"].values
        for measure in ORDERED_MEASURES:
            if measure in seg.columns:
                values = seg[measure].values
                interp_vals = np.interp(common_time, times, values)
                interp_data[measure].append(interp_vals)

    # Compute the average across segments for each measure
    averaged = {}
    for measure in ORDERED_MEASURES:
        if interp_data[measure]:
            stacked = np.vstack(interp_data[measure])
            mean_vals = np.mean(stacked, axis=0)
            averaged[measure] = mean_vals
        else:
            averaged[measure] = None

    # Build the line graphs with titles including the query
    fig1, ax1 = plt.subplots(figsize=(6,4))
    if averaged["Approach / Withdraw"] is not None:
        ax1.plot(common_time, averaged["Approach / Withdraw"], label="Approach / Withdraw", color="blue")
    ax1.set_xlim(-pre_duration, effective_post)
    ax1.set_ylim(-0.5, 0.5)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Approach / Withdraw")
    ax1.set_title("Approach / Withdraw - Query: " + query)
    ax1.legend()
    buf1 = BytesIO()
    fig1.savefig(buf1, format="png")
    buf1.seek(0)
    img1 = base64.b64encode(buf1.read()).decode("utf-8")
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(6,4))
    if averaged["Engagement"] is not None:
        ax2.plot(common_time, averaged["Engagement"], label="Engagement", color="green")
    ax2.set_xlim(-pre_duration, effective_post)
    ax2.set_ylim(0, 1.0)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Engagement")
    ax2.set_title("Engagement - Query: " + query)
    ax2.legend()
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    buf2.seek(0)
    img2 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(6,4))
    if averaged["Emotional Intensity"] is not None:
        ax3.plot(common_time, averaged["Emotional Intensity"], label="Emotional Intensity", color="red")
    ax3.set_xlim(-pre_duration, effective_post)
    ax3.set_ylim(0, 1.0)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Emotional Intensity")
    ax3.set_title("Emotional Intensity - Query: " + query)
    ax3.legend()
    buf3 = BytesIO()
    fig3.savefig(buf3, format="png")
    buf3.seek(0)
    img3 = base64.b64encode(buf3.read()).decode("utf-8")
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(6,4))
    for measure in ["Memory Encoding - Detail", "Memory Encoding - Global", "Memory Encoding - Composite"]:
        if averaged[measure] is not None:
            ax4.plot(common_time, averaged[measure], label=measure)
    ax4.set_xlim(-pre_duration, effective_post)
    ax4.set_ylim(0, 1.0)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Memory Encoding")
    ax4.set_title("Memory Encoding - Query: " + query)
    ax4.legend(fontsize="small")
    buf4 = BytesIO()
    fig4.savefig(buf4, format="png")
    buf4.seek(0)
    img4 = base64.b64encode(buf4.read()).decode("utf-8")
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(6,4))
    for measure in ["General Attention - Detail", "General Attention - Global", "General Attention - Composite"]:
        if averaged[measure] is not None:
            ax5.plot(common_time, averaged[measure], label=measure)
    ax5.set_xlim(-pre_duration, effective_post)
    ax5.set_ylim(0, 1.0)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("General Attention")
    ax5.set_title("General Attention - Query: " + query)
    ax5.legend(fontsize="small")
    buf5 = BytesIO()
    fig5.savefig(buf5, format="png")
    buf5.seek(0)
    img5 = base64.b64encode(buf5.read()).decode("utf-8")
    plt.close(fig5)

    fig6, ax6 = plt.subplots(figsize=(6,4))
    for measure in ["Visual Attention - Detail", "Visual Attention - Global", "Visual Attention - Composite"]:
        if averaged[measure] is not None:
            ax6.plot(common_time, averaged[measure], label=measure)
    ax6.set_xlim(-pre_duration, effective_post)
    ax6.set_ylim(0, 1.0)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Visual Attention")
    ax6.set_title("Visual Attention - Query: " + query)
    ax6.legend(fontsize="small")
    buf6 = BytesIO()
    fig6.savefig(buf6, format="png")
    buf6.seek(0)
    img6 = base64.b64encode(buf6.read()).decode("utf-8")
    plt.close(fig6)

    # Create a CSV file (comma-delimited) with separate cells for each column.
    csv_output = StringIO()
    csv_writer = csv.writer(csv_output, delimiter=',')
    header = ["Query", "Time(ms)"] + ORDERED_MEASURES
    csv_writer.writerow(header)

    for i in range(len(common_time)):
        t_ms = int(common_time[i] * 1000)
        row = [query, t_ms]
        for measure in ORDERED_MEASURES:
            if averaged[measure] is not None:
                row.append(averaged[measure][i])
            else:
                row.append("")
        csv_writer.writerow(row)

    csv_data = csv_output.getvalue()
    csv_base64 = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
    filename = f"{safe_filename(query)}_results.csv"
    download_link = f'<a href="data:text/csv;base64,{csv_base64}" download="{filename}">Download Results</a>'

    html = "<html><head><title>Line Graph Aggregated Results</title></head><body>"
    html += "<h1>Aggregated Averaged Metrics (Line Graphs) - Query: " + query + "</h1>"
    html += f"<p>Graphing from -{pre_duration:.2f} to {effective_post:.2f} seconds relative to each event's start.</p>"
    html += "<h2>Approach / Withdraw</h2>"
    html += f'<img src="data:image/png;base64,{img1}" alt="Approach / Withdraw"><br>'
    html += "<h2>Engagement</h2>"
    html += f'<img src="data:image/png;base64,{img2}" alt="Engagement"><br>'
    html += "<h2>Emotional Intensity</h2>"
    html += f'<img src="data:image/png;base64,{img3}" alt="Emotional Intensity"><br>'
    html += "<h2>Memory Encoding (Detail, Global, Composite)</h2>"
    html += f'<img src="data:image/png;base64,{img4}" alt="Memory Encoding"><br>'
    html += "<h2>General Attention (Detail, Global, Composite)</h2>"
    html += f'<img src="data:image/png;base64,{img5}" alt="General Attention"><br>'
    html += "<h2>Visual Attention (Detail, Global, Composite)</h2>"
    html += f'<img src="data:image/png;base64,{img6}" alt="Visual Attention"><br>'

    if excluded_names:
        html += "<p style='color:red;'><em>* The following ads were excluded due to insufficient pre or post data: "
        html += ", ".join(excluded_names) + "</em></p>"

    html += f"""
    <h3>Adjust Graph Duration</h3>
    <p><strong>Pure Event Duration:</strong> {pure_duration:.2f} seconds</p>
    <form method="post" action="/aggregated_results">
        <label for="pre_duration">Pre-highlight Duration (seconds):</label>
        <input type="range" id="pre_duration" name="pre_duration" min="0" max="10" step="0.1" value="{pre_duration}" oninput="this.nextElementSibling.value = this.value">
        <output>{pre_duration:.2f}</output><br>
        <label for="post_extra">Post-highlight Extra Duration (seconds):</label>
        <input type="range" id="post_extra" name="post_extra" min="0" max="10" step="0.1" value="{post_extra}" oninput="this.nextElementSibling.value = this.value">
        <output>{post_extra:.2f}</output><br>
    """
    for i in range(len(video_id)):
        html += f'<input type="hidden" name="video_id" value="{video_id[i]}">'
        html += f'<input type="hidden" name="start_time" value="{start_time[i]}">'
        html += f'<input type="hidden" name="end_time" value="{end_time[i]}">'
        html += f'<input type="hidden" name="ad_name" value="{ad_name[i]}">'
    html += f'<input type="hidden" name="query" value="{escape(query)}">'
    html += '<br><input type="submit" value="Update Graphs">'
    html += "</form>"
    html += "<br>" + download_link
    html += '<br><button onclick="javascript:history.back()">Back to Compute Averages</button>'
    html += "<br><a href='/'>Back to Home</a>"
    html += "</body></html>"
    return HTMLResponse(content=html)

def aggregated_box(video_id, start_time, end_time, ad_name, pure_duration, query):
    """
    Renders box plots and creates a CSV (comma-delimited) in which each measure's data is output in separate cells.
    The CSV file name is based on the query.
    """
    box_data = {m: [] for m in ORDERED_MEASURES}
    for i in range(len(video_id)):
        try:
            st = float(start_time[i])
            et = float(end_time[i])
        except:
            continue
        csv_filename = os.path.join(CSV_FOLDER, f"{video_id[i]}.csv")
        if not os.path.exists(csv_filename):
            continue
        try:
            df = pd.read_csv(csv_filename, header=1)
            df.rename(columns={df.columns[0]: "Time"}, inplace=True)
            csv_max_time = df["Time"].max()
            if abs(et - csv_max_time) <= 0.05:
                et = csv_max_time
            seg = df[(df["Time"] >= st) & (df["Time"] <= et)]
            if not seg.empty:
                for measure in ORDERED_MEASURES:
                    if measure in df.columns:
                        box_data[measure].extend(seg[measure].dropna().tolist())
        except:
            continue

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if box_data["Approach / Withdraw"]:
        axes[0,0].boxplot(box_data["Approach / Withdraw"])
        axes[0,0].set_title("Approach / Withdraw")
        axes[0,0].set_ylim(-0.5, 0.5)

    if box_data["Engagement"]:
        axes[0,1].boxplot(box_data["Engagement"])
        axes[0,1].set_title("Engagement")
        axes[0,1].set_ylim(0, 1.0)

    if box_data["Emotional Intensity"]:
        axes[0,2].boxplot(box_data["Emotional Intensity"])
        axes[0,2].set_title("Emotional Intensity")
        axes[0,2].set_ylim(0, 1.0)

    mem_data = []
    labels_mem = []
    for m in ["Memory Encoding - Detail", "Memory Encoding - Global", "Memory Encoding - Composite"]:
        if box_data[m]:
            mem_data.append(box_data[m])
            labels_mem.append(m)
    if mem_data:
        axes[1,0].boxplot(mem_data, labels=labels_mem)
        axes[1,0].set_title("Memory Encoding")
        axes[1,0].set_ylim(0, 1.0)

    gen_data = []
    labels_gen = []
    for m in ["General Attention - Detail", "General Attention - Global", "General Attention - Composite"]:
        if box_data[m]:
            gen_data.append(box_data[m])
            labels_gen.append(m)
    if gen_data:
        axes[1,1].boxplot(gen_data, labels=labels_gen)
        axes[1,1].set_title("General Attention")
        axes[1,1].set_ylim(0, 1.0)

    vis_data = []
    labels_vis = []
    for m in ["Visual Attention - Detail", "Visual Attention - Global", "Visual Attention - Composite"]:
        if box_data[m]:
            vis_data.append(box_data[m])
            labels_vis.append(m)
    if vis_data:
        axes[1,2].boxplot(vis_data, labels=labels_vis)
        axes[1,2].set_title("Visual Attention")
        axes[1,2].set_ylim(0, 1.0)

    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # Prepare CSV output with comma delimiter and separate cells for each data value.
    csv_output = StringIO()
    csv_writer = csv.writer(csv_output, delimiter=',')
    # Determine maximum number of values in any measure:
    max_len = max((len(v) for v in box_data.values() if v), default=0)
    header = ["Query", "Measure"] + [f"Value_{i+1}" for i in range(max_len)]
    csv_writer.writerow(header)
    for measure in ORDERED_MEASURES:
        if box_data[measure]:
            row = [query, measure] + box_data[measure]
            # Pad with empty strings if needed for uniform column count
            row += [""] * (max_len - len(box_data[measure]))
            csv_writer.writerow(row)
    csv_data = csv_output.getvalue()
    csv_base64 = base64.b64encode(csv_data.encode("utf-8")).decode("utf-8")
    filename = f"{safe_filename(query)}_results.csv"
    download_link = f'<a href="data:text/csv;base64,{csv_base64}" download="{filename}">Download Results</a>'

    html = "<html><head><title>Box and Whisker Aggregated Results</title></head><body>"
    html += "<h1>Aggregated Box and Whisker Graphs (Pure Data) - Query: " + query + "</h1>"
    html += f"<p>Pure Event Duration: {pure_duration:.2f} seconds</p>"
    html += f'<img src="data:image/png;base64,{img}" alt="Box and Whisker Graphs"><br>'
    html += '<br><button onclick="javascript:history.back()">Back to Compute Averages</button>'
    html += "<br>" + download_link
    html += "<br><a href='/'>Back to Home</a>"
    html += "</body></html>"
    return HTMLResponse(content=html)

@app.post("/aggregated_results", response_class=HTMLResponse)
def aggregated_results(
    pre_duration: float = Form(0),
    post_extra: float = Form(0),
    video_id: list[str] = Form(...),
    start_time: list[float] = Form(...),
    end_time: list[float] = Form(...),
    ad_name: list[str] = Form(...),
    query: str = Form(...)
):
    graph_type = "line"
    durations = []
    for i in range(len(video_id)):
        try:
            st = float(start_time[i])
            et = float(end_time[i])
            durations.append(et - st)
        except:
            pass
    pure_duration = min(durations) if durations else 0

    if graph_type == "line":
        return aggregated_line(video_id, start_time, end_time, ad_name, pre_duration, post_extra, pure_duration, query)
    else:
        return aggregated_box(video_id, start_time, end_time, ad_name, pure_duration, query)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
