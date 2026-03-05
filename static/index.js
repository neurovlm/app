import { Niivue, NVImage } from "./js/niivue.min.js";

const cache = {
    remap: null,
    links: null,
    titles: null,
    pmids: null,
    abstracts: null,
};

const abstractByDoi = new Map();
const abstractFetchByDoi = new Map();
const abstractByPmid = new Map();
const abstractFetchByPmid = new Map();

const NIFTI_DIMS = [46, 55, 46];
const NIFTI_PIX_DIMS = [1, 1, 1];
const NIFTI_AFFINE = [
    4.0, 0.0, 0.0, -90.0,
    0.0, 4.0, 0.0, -126.0,
    0.0, 0.0, 4.0, -72.0,
    0.0, 0.0, 0.0, 1.0,
];

const TEMPLATE_XFORM = [
    5.42400005033472, 0.0, 0.0, -19.306002583559717,
    0.0, 5.42400005033472, 0.0, -20.662002596143395,
    0.0, 0.0, 5.42400005033472, -0.3220024073881831,
    0.0, 0.0, 0.0, 1.0,
];

const DARK_BG_RGBA = [19 / 255, 19 / 255, 20 / 255, 1.0];
const MAX_CITATIONS_TO_RENDER = 100;

function waitForElement(selector, timeout = 10000) {
    return new Promise((resolve, reject) => {
        const existing = document.querySelector(selector);
        if (existing) {
            resolve(existing);
            return;
        }

        const observer = new MutationObserver(() => {
            const el = document.querySelector(selector);
            if (el) {
                clearTimeout(timeoutId);
                observer.disconnect();
                resolve(el);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
        });

        const timeoutId = setTimeout(() => {
            observer.disconnect();
            reject(new Error(`Timeout waiting for element: ${selector}`));
        }, timeout);
    });
}

async function fetchTextWithFallback(paths) {
    let lastError = null;
    for (const path of paths) {
        try {
            const response = await fetch(path);
            if (!response.ok) {
                lastError = new Error(`HTTP ${response.status} for ${path}`);
                continue;
            }
            return await response.text();
        } catch (error) {
            lastError = error;
        }
    }
    throw lastError ?? new Error("No fetch paths provided");
}

function splitLinesPreserveAlignment(text) {
    const normalized = text.replace(/\r/g, "");
    const lines = normalized.split("\n");
    if (lines.length > 0 && lines[lines.length - 1] === "") {
        lines.pop();
    }
    return lines;
}

function htmlToPlainText(value) {
    if (!value) {
        return "";
    }
    const div = document.createElement("div");
    div.innerHTML = String(value);
    return div.textContent || div.innerText || "";
}

function normalizeAbstractText(value) {
    if (value == null) {
        return "";
    }
    const text = htmlToPlainText(String(value))
        .replace(/\s+/g, " ")
        .trim();

    if (
        !text
        || text === "NA"
        || text === "N/A"
        || text === "__NA__"
        || text.toLowerCase() === "none"
    ) {
        return "";
    }

    return text;
}

async function preloadRemapData() {
    if (cache.remap) {
        return cache.remap;
    }
    try {
        const data = await fetchTextWithFallback(["/static/remap.txt"]);
        cache.remap = JSON.parse(data);
    } catch (error) {
        console.error("Error preloading remap data:", error);
        cache.remap = null;
    }
    return cache.remap;
}

async function preloadTitlesData() {
    if (cache.titles) {
        return cache.titles;
    }
    try {
        const data = await fetchTextWithFallback(["/static/models/titles.txt"]);
        cache.titles = splitLinesPreserveAlignment(data);
    } catch (error) {
        console.error("Error preloading titles data:", error);
        cache.titles = [];
    }
    return cache.titles;
}

async function preloadLinksData() {
    if (cache.links) {
        return cache.links;
    }
    try {
        const data = await fetchTextWithFallback(["/static/models/links.txt"]);
        cache.links = splitLinesPreserveAlignment(data);
    } catch (error) {
        console.error("Error preloading links data:", error);
        cache.links = [];
    }
    return cache.links;
}

async function preloadPmidsData() {
    if (cache.pmids) {
        return cache.pmids;
    }
    try {
        const data = await fetchTextWithFallback(["/static/models/pmids.txt"]);
        cache.pmids = splitLinesPreserveAlignment(data);
    } catch (error) {
        console.error("Error preloading PMID data:", error);
        cache.pmids = [];
    }
    return cache.pmids;
}

async function preloadAbstractsData() {
    if (cache.abstracts) {
        return cache.abstracts;
    }

    try {
        const data = await fetchTextWithFallback(["/static/models/abstracts.txt"]);
        cache.abstracts = splitLinesPreserveAlignment(data).map(normalizeAbstractText);
    } catch (error) {
        // Abstracts file is optional. Keep this quiet except in debug console.
        console.warn("No local abstracts file found. Falling back to DOI lookup on hover.");
        cache.abstracts = [];
    }
    return cache.abstracts;
}

function argmax(values) {
    let maxVal = Number.NEGATIVE_INFINITY;
    let maxIdx = 0;
    for (let i = 0; i < values.length; i++) {
        if (values[i] > maxVal) {
            maxVal = values[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

function autosizeTextarea(textarea) {
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
}

function hideTooltip(tooltip) {
    if (!tooltip) {
        return;
    }
    tooltip.classList.add("hidden");
    tooltip.setAttribute("aria-hidden", "true");
}

function setTooltipContent(tooltip, title, abstract, isLoading = false) {
    if (!tooltip) {
        return;
    }

    tooltip.textContent = "";

    const titleEl = document.createElement("div");
    titleEl.className = "abstract-title";
    titleEl.textContent = title;

    const abstractEl = document.createElement("div");
    abstractEl.className = "abstract-text";
    abstractEl.textContent = isLoading
        ? "Loading abstract..."
        : (abstract || "Abstract unavailable for this publication.");

    tooltip.appendChild(titleEl);
    tooltip.appendChild(abstractEl);
}

function positionTooltip(tooltip, event) {
    if (!tooltip || tooltip.classList.contains("hidden")) {
        return;
    }

    const margin = 12;
    const maxX = window.innerWidth - tooltip.offsetWidth - margin;
    const maxY = window.innerHeight - tooltip.offsetHeight - margin;

    let left = event.clientX + 14;
    let top = event.clientY + 14;

    if (left > maxX) {
        left = Math.max(margin, event.clientX - tooltip.offsetWidth - 14);
    }

    if (top > maxY) {
        top = Math.max(margin, event.clientY - tooltip.offsetHeight - 14);
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
}

function extractDoiFromLink(link) {
    if (!link) {
        return "";
    }

    const decoded = decodeURIComponent(String(link));
    const doiMatch = decoded.match(/10\.\d{4,9}\/[\-._;()/:A-Z0-9]+/i);
    if (!doiMatch) {
        return "";
    }

    return doiMatch[0].replace(/[)\],.;]+$/, "");
}

async function fetchJsonWithTimeout(url, timeoutMs = 7000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, {
            method: "GET",
            signal: controller.signal,
            headers: {
                Accept: "application/json",
            },
        });

        if (!response.ok) {
            return null;
        }

        return await response.json();
    } catch {
        return null;
    } finally {
        clearTimeout(timer);
    }
}

async function fetchAbstractFromCrossref(doi) {
    const url = `https://api.crossref.org/works/${encodeURIComponent(doi)}`;
    const json = await fetchJsonWithTimeout(url);
    const abstract = json?.message?.abstract;
    return normalizeAbstractText(abstract);
}

async function fetchAbstractFromEuropePmcByDoi(doi) {
    const query = encodeURIComponent(`DOI:"${doi}"`);
    const url = `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=${query}&format=json&pageSize=1&resultType=core`;
    const json = await fetchJsonWithTimeout(url);
    const abstract = json?.resultList?.result?.[0]?.abstractText;
    return normalizeAbstractText(abstract);
}

async function fetchAbstractFromEuropePmcByPmid(pmid) {
    const query = encodeURIComponent(`EXT_ID:${pmid} AND SRC:MED`);
    const url = `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=${query}&format=json&pageSize=1&resultType=core`;
    const json = await fetchJsonWithTimeout(url);
    const abstract = json?.resultList?.result?.[0]?.abstractText;
    return normalizeAbstractText(abstract);
}

async function resolveCitationAbstract(citation) {
    const localAbstract = normalizeAbstractText(citation.abstract);
    if (localAbstract) {
        return localAbstract;
    }

    const doi = extractDoiFromLink(citation.link);
    if (doi) {
        if (abstractByDoi.has(doi)) {
            return abstractByDoi.get(doi);
        }

        if (!abstractFetchByDoi.has(doi)) {
            const promise = (async () => {
                let text = await fetchAbstractFromCrossref(doi);
                if (!text) {
                    text = await fetchAbstractFromEuropePmcByDoi(doi);
                }
                abstractByDoi.set(doi, text || "");
                return text || "";
            })();
            abstractFetchByDoi.set(doi, promise);
        }

        const text = await abstractFetchByDoi.get(doi);
        if (text) {
            return text;
        }
    }

    const pmid = String(citation.pmid || "").trim();
    if (pmid) {
        if (abstractByPmid.has(pmid)) {
            return abstractByPmid.get(pmid);
        }

        if (!abstractFetchByPmid.has(pmid)) {
            const promise = (async () => {
                const text = await fetchAbstractFromEuropePmcByPmid(pmid);
                abstractByPmid.set(pmid, text || "");
                return text || "";
            })();
            abstractFetchByPmid.set(pmid, promise);
        }

        const text = await abstractFetchByPmid.get(pmid);
        if (text) {
            return text;
        }
    }

    return "";
}

function renderCitations(container, citations, tooltip) {
    container.textContent = "";
    const fragment = document.createDocumentFragment();
    const count = Math.min(MAX_CITATIONS_TO_RENDER, citations.length);

    for (let index = 0; index < count; index++) {
        const citation = citations[index];
        const row = document.createElement("div");
        row.className = "citation-item";

        const prefix = document.createElement("span");
        prefix.className = "citation-index";
        prefix.textContent = `[${index + 1}]`;
        row.appendChild(prefix);

        const normalizedTitle = String(citation.title || "")
            .replace(/\s+/g, " ")
            .trim();
        const title = normalizedTitle || "Untitled publication";
        if (citation.link) {
            const anchor = document.createElement("a");
            anchor.href = citation.link;
            anchor.target = "_blank";
            anchor.rel = "noopener noreferrer";
            anchor.textContent = title;
            row.appendChild(anchor);
        } else {
            const text = document.createElement("span");
            text.textContent = title;
            row.appendChild(text);
        }

        let hoverToken = 0;
        row.addEventListener("mouseenter", async (event) => {
            hoverToken += 1;
            const token = hoverToken;

            setTooltipContent(tooltip, title, "", true);
            tooltip.classList.remove("hidden");
            tooltip.setAttribute("aria-hidden", "false");
            positionTooltip(tooltip, event);

            const abstract = await resolveCitationAbstract(citation);
            if (token !== hoverToken) {
                return;
            }

            setTooltipContent(tooltip, title, abstract, false);
            positionTooltip(tooltip, event);
        });

        row.addEventListener("mousemove", (event) => {
            positionTooltip(tooltip, event);
        });

        row.addEventListener("mouseleave", () => {
            hoverToken += 1;
            hideTooltip(tooltip);
        });

        fragment.appendChild(row);
    }

    container.appendChild(fragment);
}

async function waitForDivToHide(divId) {
    const div = document.getElementById(divId);
    if (!div || div.style.display === "none") {
        return;
    }

    await new Promise((resolve) => {
        const observer = new MutationObserver(() => {
            if (div.style.display === "none") {
                observer.disconnect();
                resolve();
            }
        });

        observer.observe(div, {
            attributes: true,
            attributeFilter: ["style"],
        });
    });
}

async function run() {
    const remapPromise = preloadRemapData();
    const linksPromise = preloadLinksData();
    const titlesPromise = preloadTitlesData();
    const pmidsPromise = preloadPmidsData();
    const abstractsPromise = preloadAbstractsData();

    const textbox = document.getElementById("txid");
    const submitButton = document.getElementById("query_submit");
    const queryStatus = document.getElementById("query_status");
    const outputContainer = document.getElementById("output_container");
    const abstractTooltip = document.getElementById("abstract_tooltip");
    const inputContainer = document.querySelector("#query_box .input-container");

    autosizeTextarea(textbox);
    textbox.addEventListener("input", () => autosizeTextarea(textbox));
    outputContainer.addEventListener("scroll", () => hideTooltip(abstractTooltip));

    await waitForElement("#dataname");

    const parent = document.getElementsByName("w2figure0")[0];
    const surfaceCanvas = parent.children[0];
    parent.classList.add("plot-host");
    surfaceCanvas.classList.add("surface-view");

    function forcePlotContainment() {
        if (!parent) {
            return;
        }

        parent.style.width = "";
        parent.style.height = "";
        parent.style.maxWidth = "";
        parent.style.maxHeight = "";
        parent.style.left = "";
        parent.style.right = "";
        parent.style.top = "";
        parent.style.bottom = "";
        parent.style.overflow = "hidden";

        if (surfaceCanvas) {
            surfaceCanvas.style.width = "100%";
            surfaceCanvas.style.height = "100%";
            surfaceCanvas.style.maxWidth = "100%";
            surfaceCanvas.style.maxHeight = "100%";
            surfaceCanvas.style.left = "0";
            surfaceCanvas.style.top = "0";
            surfaceCanvas.style.right = "0";
            surfaceCanvas.style.bottom = "0";
            surfaceCanvas.style.position = "absolute";
            surfaceCanvas.style.overflow = "hidden";
        }

        const figureUi = document.getElementById("figure_ui");
        if (figureUi) {
            figureUi.remove();
        }
    }
    forcePlotContainment();

    const figureUiObserver = new MutationObserver(() => {
        const figureUi = document.getElementById("figure_ui");
        if (figureUi) {
            figureUi.remove();
        }
    });
    figureUiObserver.observe(document.body, { childList: true, subtree: true });

    const legacyDataOpts = document.getElementById("dataopts");
    if (legacyDataOpts) {
        legacyDataOpts.style.display = "none";
    }

    const canvas = document.createElement("canvas");
    canvas.id = "gl1";
    canvas.style.backgroundColor = "#131314";
    canvas.classList.add("hidden");
    parent.appendChild(canvas);

    const toggleContainer = document.createElement("div");
    toggleContainer.className = "toggle-container";

    const mniDiv = document.createElement("div");
    mniDiv.className = "toggle-option";
    mniDiv.id = "mni";
    mniDiv.textContent = "MNI";

    const fsaverageDiv = document.createElement("div");
    fsaverageDiv.className = "toggle-option active";
    fsaverageDiv.id = "fsaverage";
    fsaverageDiv.textContent = "fsaverage";

    toggleContainer.appendChild(mniDiv);
    toggleContainer.appendChild(fsaverageDiv);
    document.body.appendChild(toggleContainer);

    const splitter = document.createElement("div");
    splitter.id = "splitter";
    splitter.setAttribute("aria-hidden", "true");
    document.body.appendChild(splitter);

    const rhsPanel = document.getElementById("rhs");
    const rootStyle = document.documentElement;

    function updateViewportHeightVar() {
        const visualHeight = window.visualViewport ? window.visualViewport.height : 0;
        const fallbackHeight = window.innerHeight || 0;
        const nextHeight = Math.max(visualHeight || 0, fallbackHeight || 0);
        if (nextHeight > 0) {
            rootStyle.style.setProperty("--viewport-height", `${Math.round(nextHeight)}px`);
        }
    }

    const volumeList = [
        {
            url: "/static/mni/mni152.nii.gz",
            colormap: "gray",
            visible: true,
            opacity: 1,
        },
    ];

    const nv1 = new Niivue({
        backColor: DARK_BG_RGBA,
        backgroundColor: DARK_BG_RGBA,
        show3Dcrosshair: true,
        dragAndDropEnabled: true,
        onLocationChange: (data) => {
            document.getElementById("intensity").innerHTML = `&nbsp;&nbsp;${data.string}`;
        },
    });

    nv1.isAlphaClipDark = true;
    Object.assign(nv1.opts, {
        crosshairGap: 12,
        dragMode: nv1.dragModes.pan,
        yoke3Dto2DZoom: true,
        isResizeCanvas: true,
        isCornerOrientationText: false,
        isOrientCube: false,
        textHeight: 0,
    });

    let hasInitialMniCrosshair = false;
    function setInitialMniCrosshairOffset() {
        if (!nv1.volumes || nv1.volumes.length < 1) {
            return false;
        }

        const refVolume = nv1.volumes[0];
        const dims = refVolume.dimsRAS || refVolume.dims;
        if (!dims || dims.length < 3) {
            return false;
        }

        const nx = dims.length >= 4 ? Number(dims[1]) : Number(dims[0]);
        const ny = dims.length >= 4 ? Number(dims[2]) : Number(dims[1]);
        const nz = dims.length >= 4 ? Number(dims[3]) : Number(dims[2]);
        if (!Number.isFinite(nx) || !Number.isFinite(ny) || !Number.isFinite(nz) || nx <= 1 || ny <= 1 || nz <= 1) {
            return false;
        }

        const x = Math.max(0, Math.min(nx - 1, Math.floor(nx / 2) + 5));
        const y = Math.max(0, Math.min(ny - 1, Math.floor(ny / 2)));
        const z = Math.max(0, Math.min(nz - 1, Math.floor(nz / 2)));
        nv1.scene.crosshairPos = nv1.vox2frac([x, y, z]);
        return true;
    }

    nv1.attachTo("gl1");
    nv1.setSliceType(nv1.sliceTypeMultiplanar);
    nv1.loadVolumes(volumeList)
        .then(() => {
            forcePlotContainment();
            if (!hasInitialMniCrosshair) {
                hasInitialMniCrosshair = setInitialMniCrosshairOffset();
            }
            if (hasInitialMniCrosshair) {
                nv1.createOnLocationChange();
            }
            updateMniLayoutForViewport();
            scheduleResizePlots();
        })
        .catch((error) => {
            console.error("Error loading volumes:", error);
        });

    const drop = document.getElementById("sliceType");
    drop.onchange = () => {
        const st = parseInt(drop.value, 10);
        nv1.setSliceType(st);
    };

    const alphaSlider = document.getElementById("alphaSlider");
    alphaSlider.oninput = function onInput() {
        if (nv1.volumes.length > 1) {
            nv1.volumes[1].cal_min = this.value / 255;
            nv1.updateGLVolume();
        }
    };

    const MOBILE_LAYOUT_BREAKPOINT_PX = 980;
    const MIN_BRAIN_WIDTH_PX = 340;
    const MIN_RHS_WIDTH_PX = 360;

    const defaultMultiplanarLayout = Number.isFinite(nv1.opts.multiplanarLayout)
        ? nv1.opts.multiplanarLayout
        : 0;
    const defaultMultiplanarShowRender = Number.isFinite(nv1.opts.multiplanarShowRender)
        ? nv1.opts.multiplanarShowRender
        : 2;
    const defaultMultiplanarEqualSize = Boolean(nv1.opts.multiplanarEqualSize);
    const defaultTileMargin = Number.isFinite(nv1.opts.tileMargin) ? nv1.opts.tileMargin : 0;

    let currentView = "mni";
    let resizeRaf = 0;

    function resizeFsaveragePlot() {
        if (typeof figure !== "undefined" && figure && typeof figure.resize === "function") {
            figure.resize();
        }
        if (typeof viewer !== "undefined" && viewer && typeof viewer.resize === "function") {
            viewer.resize();
        }
    }

    function resizeMniPlot() {
        const rect = parent.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0 || !nv1.canvas || !nv1.gl) {
            return;
        }

        nv1.canvas.width = Math.max(1, Math.floor(rect.width * nv1.uiData.dpr));
        nv1.canvas.height = Math.max(1, Math.floor(rect.height * nv1.uiData.dpr));
        nv1.gl.viewport(0, 0, nv1.gl.canvas.width, nv1.gl.canvas.height);
        if (typeof nv1.resizeListener === "function") {
            nv1.resizeListener();
        } else if (typeof nv1.resizeScene === "function") {
            nv1.resizeScene();
        }

        if (currentView === "mni") {
            nv1.drawScene();
        }
    }

    function resizePlots() {
        forcePlotContainment();
        resizeFsaveragePlot();
        resizeMniPlot();
    }

    function scheduleResizePlots() {
        if (resizeRaf) {
            cancelAnimationFrame(resizeRaf);
        }
        resizeRaf = requestAnimationFrame(() => {
            resizeRaf = 0;
            resizePlots();
        });
    }

    function updateMniLayoutForViewport() {
        const rect = parent.getBoundingClientRect();
        const shouldUseRowLayout = window.innerWidth <= MOBILE_LAYOUT_BREAKPOINT_PX
            || (rect.width > 0 && rect.height > 0 && rect.width / rect.height < 2.5);

        if (shouldUseRowLayout) {
            nv1.opts.multiplanarLayout = 3;
            nv1.opts.multiplanarShowRender = 0;
            nv1.opts.multiplanarEqualSize = true;
            nv1.opts.tileMargin = 0;
            nv1.opts.sliceMosaicString = "";
            nv1.opts.centerMosaic = true;
        } else {
            nv1.opts.multiplanarLayout = defaultMultiplanarLayout;
            nv1.opts.multiplanarShowRender = defaultMultiplanarShowRender;
            nv1.opts.multiplanarEqualSize = defaultMultiplanarEqualSize;
            nv1.opts.tileMargin = defaultTileMargin;
            nv1.opts.sliceMosaicString = "";
            nv1.opts.centerMosaic = false;
        }
        nv1.setSliceType(nv1.sliceTypeMultiplanar);
    }

    function getSiteInsetPx() {
        const value = getComputedStyle(rootStyle).getPropertyValue("--site-inset");
        const parsed = Number.parseFloat(value);
        return Number.isFinite(parsed) ? parsed : 0;
    }

    function setRhsWidthPx(nextWidthPx) {
        if (!rhsPanel) {
            return;
        }

        const viewportWidth = window.innerWidth;
        const siteInset = getSiteInsetPx();
        const availableWidth = Math.max(0, viewportWidth - (siteInset * 2));
        const maxWidth = Math.max(280, availableWidth - MIN_BRAIN_WIDTH_PX);
        const minWidth = Math.min(MIN_RHS_WIDTH_PX, maxWidth);
        const clampedWidth = Math.max(minWidth, Math.min(nextWidthPx, maxWidth));
        rootStyle.style.setProperty("--rhs-width", `${Math.round(clampedWidth)}px`);
    }

    function updateSplitterVisibility() {
        if (!splitter) {
            return;
        }
        const isDesktop = window.innerWidth > MOBILE_LAYOUT_BREAKPOINT_PX;
        splitter.style.display = isDesktop ? "block" : "none";
        splitter.style.pointerEvents = isDesktop ? "auto" : "none";
    }

    function updateTextboxPlaceholder() {
        const measuredWidth = inputContainer ? inputContainer.getBoundingClientRect().width : 0;
        const effectiveWidth = measuredWidth > 0 ? measuredWidth : window.innerWidth;
        const isWide = effectiveWidth >= 430;
        textbox.placeholder = isWide
            ? "Query papers..."
            : "Query neuroimaging papers...";
    }

    let splitterDrag = null;
    splitter.addEventListener("pointerdown", (event) => {
        if (window.innerWidth <= MOBILE_LAYOUT_BREAKPOINT_PX || !rhsPanel) {
            return;
        }
        splitterDrag = {
            pointerId: event.pointerId,
            startX: event.clientX,
            startWidth: rhsPanel.getBoundingClientRect().width,
        };
        splitter.setPointerCapture(event.pointerId);
        document.body.classList.add("is-resizing");
        event.preventDefault();
    });

    window.addEventListener("pointermove", (event) => {
        if (!splitterDrag) {
            return;
        }
        const deltaX = event.clientX - splitterDrag.startX;
        const nextWidth = splitterDrag.startWidth - deltaX;
        setRhsWidthPx(nextWidth);
        scheduleResizePlots();
    });

    function endSplitterDrag(event) {
        if (!splitterDrag) {
            return;
        }
        if (event && splitter.hasPointerCapture(splitterDrag.pointerId)) {
            splitter.releasePointerCapture(splitterDrag.pointerId);
        }
        splitterDrag = null;
        document.body.classList.remove("is-resizing");
        scheduleResizePlots();
    }

    window.addEventListener("pointerup", endSplitterDrag);
    window.addEventListener("pointercancel", endSplitterDrag);
    window.addEventListener("blur", endSplitterDrag);

    updateViewportHeightVar();
    updateMniLayoutForViewport();
    updateSplitterVisibility();
    updateTextboxPlaceholder();

    if (typeof ResizeObserver !== "undefined") {
        const resizeObserver = new ResizeObserver(() => {
            scheduleResizePlots();
            updateTextboxPlaceholder();
        });
        resizeObserver.observe(parent);
        if (inputContainer) {
            resizeObserver.observe(inputContainer);
        }
    }
    window.addEventListener("resize", () => {
        updateViewportHeightVar();
        updateMniLayoutForViewport();
        updateSplitterVisibility();
        updateTextboxPlaceholder();
        scheduleResizePlots();
    });
    window.addEventListener("orientationchange", () => {
        updateViewportHeightVar();
        updateMniLayoutForViewport();
        updateSplitterVisibility();
        updateTextboxPlaceholder();
        scheduleResizePlots();
    });
    if (window.visualViewport) {
        window.visualViewport.addEventListener("resize", () => {
            updateViewportHeightVar();
            updateMniLayoutForViewport();
            updateTextboxPlaceholder();
            scheduleResizePlots();
        });
    }

    function switchToView(viewId) {
        currentView = viewId;
        const isMni = viewId === "mni";
        forcePlotContainment();

        document.body.classList.toggle("view-mni", isMni);
        document.body.classList.toggle("view-fsaverage", !isMni);

        if (isMni) {
            surfaceCanvas.classList.remove("visible");
            surfaceCanvas.classList.add("hidden");
            canvas.classList.remove("hidden");
            canvas.classList.add("visible");
            updateMniLayoutForViewport();
            if (!hasInitialMniCrosshair) {
                hasInitialMniCrosshair = setInitialMniCrosshairOffset();
                if (hasInitialMniCrosshair) {
                    nv1.createOnLocationChange();
                }
            }
            alphaSlider.style.display = "block";
            scheduleResizePlots();
            return;
        }

        canvas.classList.remove("visible");
        canvas.classList.add("hidden");
        alphaSlider.style.display = "none";
        surfaceCanvas.classList.remove("hidden");
        surfaceCanvas.classList.add("visible");
        scheduleResizePlots();
        setTimeout(scheduleResizePlots, 0);
    }

    const toggleOptions = document.querySelectorAll(".toggle-option");
    toggleOptions.forEach((option) => {
        option.addEventListener("click", () => {
            toggleOptions.forEach((opt) => opt.classList.remove("active"));
            option.classList.add("active");
            switchToView(option.id);
        });
    });

    let isQueryRunning = false;

    async function runQuery(query) {
        if (isQueryRunning) {
            return;
        }

        isQueryRunning = true;
        submitButton.disabled = true;
        queryStatus.textContent = "Running query...";
        hideTooltip(abstractTooltip);

        try {
            const response = await fetch("/api/query", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ input: query }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }

            const out = await response.json();
            const surface = out.surface || [];
            const volume = new Float32Array(out.volume || []);
            const pubOrder = Array.isArray(out.puborder) ? out.puborder : [];

            const [titles, links, pmids, abstracts] = await Promise.all([
                titlesPromise,
                linksPromise,
                pmidsPromise,
                abstractsPromise,
            ]);

            if (titles.length !== links.length) {
                console.warn(
                    `titles/links length mismatch: titles=${titles.length}, links=${links.length}`,
                );
            }

            const pmidToRow = new Map();
            for (let i = 0; i < pmids.length; i++) {
                const pmid = String(pmids[i] || "").trim();
                if (pmid && !pmidToRow.has(pmid)) {
                    pmidToRow.set(pmid, i);
                }
            }

            const resolveRow = (value) => {
                const numeric = Number(value);
                if (Number.isFinite(numeric)) {
                    const asIndex = Math.floor(numeric);
                    if (asIndex >= 0 && asIndex < titles.length) {
                        return asIndex;
                    }
                    const asPmid = String(asIndex);
                    if (pmidToRow.has(asPmid)) {
                        return pmidToRow.get(asPmid);
                    }
                }
                const fallbackPmid = String(value || "").trim();
                if (pmidToRow.has(fallbackPmid)) {
                    return pmidToRow.get(fallbackPmid);
                }
                return null;
            };

            const citations = [];
            for (const value of pubOrder) {
                if (citations.length >= MAX_CITATIONS_TO_RENDER) {
                    break;
                }
                const i = resolveRow(value);
                if (i == null) {
                    continue;
                }
                citations.push({
                    globalIndex: i,
                    title: titles[i] || `Untitled publication ${i}`,
                    link: links[i] || "",
                    pmid: pmids[i] || "",
                    abstract: abstracts[i] || "",
                });
            }

            renderCitations(outputContainer, citations, abstractTooltip);

            const reinds = await remapPromise;
            if (reinds && surface.length > 0) {
                const surfaceMap = new Float32Array(327684);
                for (let i = 0; i < 163842; i++) {
                    surfaceMap[i] = surface[reinds[i]];
                    surfaceMap[i + 163842] = surface[reinds[i + 163842]];
                }

                let surfaceMin = surfaceMap[0];
                let surfaceMax = surfaceMap[0];
                for (let i = 1; i < surfaceMap.length; i++) {
                    const value = surfaceMap[i];
                    if (value < surfaceMin) {
                        surfaceMin = value;
                    }
                    if (value > surfaceMax) {
                        surfaceMax = value;
                    }
                }

                const dataviews = dataset.fromJSON({
                    views: [{
                        data: ["__e506186d71676e98"],
                        state: null,
                        attrs: { priority: 1 },
                        desc: "",
                        cmap: ["magma"],
                        vmin: [surfaceMin],
                        vmax: [surfaceMax],
                        name: "neurovlm",
                    }],
                    data: {
                        __e506186d71676e98: {
                            split: 163842,
                            frames: 1,
                            name: "__e506186d71676e98",
                            subject: "fsaverage",
                            min: surfaceMin,
                            max: surfaceMax,
                            raw: false,
                        },
                    },
                    images: {
                        __e506186d71676e98: [surfaceMap],
                    },
                });
                viewer.addData(dataviews);
            }

            if (nv1.volumes.length > 1) {
                nv1.removeVolume(nv1.volumes[1]);
            }

            const bytes = await nv1.createNiftiArray(
                NIFTI_DIMS,
                NIFTI_PIX_DIMS,
                NIFTI_AFFINE,
                16,
                volume,
            );

            const nii = await NVImage.loadFromUrl({
                url: bytes,
                colormap: "magma",
                visible: true,
                opacity: 0.8,
                cal_min: 0.1,
                cal_max: 1.0,
            });

            const maxIdx = argmax(volume);
            const i4 = maxIdx % 46;
            const j4 = Math.floor((maxIdx % 2530) / 46);
            const k4 = Math.floor(maxIdx / 2530);

            const i = Math.round(TEMPLATE_XFORM[0] * i4 + TEMPLATE_XFORM[1] * j4 + TEMPLATE_XFORM[2] * k4 + TEMPLATE_XFORM[3]);
            const j = Math.round(TEMPLATE_XFORM[4] * i4 + TEMPLATE_XFORM[5] * j4 + TEMPLATE_XFORM[6] * k4 + TEMPLATE_XFORM[7]);
            const k = Math.round(TEMPLATE_XFORM[8] * i4 + TEMPLATE_XFORM[9] * j4 + TEMPLATE_XFORM[10] * k4 + TEMPLATE_XFORM[11]);

            nv1.addVolume(nii);
            nv1.scene.crosshairPos = nv1.vox2frac([i, j, k]);
            updateMniLayoutForViewport();
            nv1.drawScene();
            nv1.createOnLocationChange();
            alphaSlider.value = String(0.1 * 255);

            const duration = out.duration_ms ?? "?";
            queryStatus.textContent = `Done in ${duration} ms`;
        } catch (error) {
            console.error("Query processing error:", error);
            queryStatus.textContent = "Query failed. Check server logs.";
        } finally {
            isQueryRunning = false;
            submitButton.disabled = false;
            textbox.focus();
        }
    }

    async function submitCurrentQuery() {
        const query = textbox.value.trim();
        if (!query) {
            return;
        }
        textbox.value = "";
        autosizeTextarea(textbox);
        await runQuery(query);
    }

    textbox.addEventListener("keydown", async (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            await submitCurrentQuery();
        }
    });

    submitButton.addEventListener("click", async () => {
        await submitCurrentQuery();
    });

    await waitForDivToHide("ctmload");
    setTimeout(() => {
        const loader = document.getElementById("loader");
        if (loader) {
            loader.style.display = "none";
        }

        toggleOptions.forEach((opt) => opt.classList.remove("active"));
        mniDiv.classList.add("active");
        switchToView("mni");
        scheduleResizePlots();
    }, 450);
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", run);
} else {
    run();
}
