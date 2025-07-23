import {Niivue, NVImage} from "./js/niivue.min.js";

const worker = new Worker('/static/js/lzma-worker.js');

// Cache for frequently used data
const cache = {
    remap: null,
    surfaceMap: null
};

function waitForElement(selector, timeout = 10000) {
    /*/
    Waits for an element to be defined
    Needed because mriviewer is async and lags behind
    Before wasm took long enough to load it wasn't an issue
    /*/
    return new Promise((resolve, reject) => {
        const el = document.querySelector(selector);
        if (el) return resolve(el);

        const observer = new MutationObserver(() => {
            const el = document.querySelector(selector);
            if (el) {
                observer.disconnect();
                resolve(el);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
        });

        // Optional: timeout safety
        const timeoutId = setTimeout(() => {
            observer.disconnect();
            reject(new Error(`Timeout: Element ${selector} not found.`));
        }, timeout);

        // Clear timeout if element is found
        observer.originalDisconnect = observer.disconnect;
        observer.disconnect = () => {
            clearTimeout(timeoutId);
            observer.originalDisconnect();
        };
    });
}

// Pre-load remap data
async function preloadRemapData() {
    if (!cache.remap) {
        try {
            const response = await fetch('/static/remap.txt');
            const data = await response.text();
            cache.remap = JSON.parse(data);
        } catch (error) {
            console.error('Error preloading remap data:', error);
        }
    }
    return cache.remap;
}

// Optimized surface map processing
function processSurfaceMap(_surface_map, reinds) {
    if (!cache.surfaceMap) {
        cache.surfaceMap = new Float32Array(327684);
    }

    const surface_map = cache.surfaceMap;

    // Use batch processing for better performance
    for (let i = 0; i < 163842; i++) {
        surface_map[i] = _surface_map[reinds[i]];
        surface_map[i + 163842] = _surface_map[reinds[i + 163842]];
    }

    return surface_map;
}

// Debounced function to prevent multiple rapid queries
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

async function run() {
    // Pre-load remap data in parallel
    const remapPromise = preloadRemapData();

    // Get text box input
    const textbox = document.getElementById("txid");

    // Scroll through publications after query
    let scrollTop = 0;
    const outputContainer = document.getElementById("output_container");

    // Create a debounced query handler
    const debouncedQuery = debounce(async function(query) {
        try {
            // Store scroll position
            scrollTop = outputContainer?.scrollTop || 0;

            // Autoencoder query
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: query })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const out = await response.json();

            const { surface: _surface_map, volume, puborder: pub_order } = out;

            // Process citations in batches to avoid blocking
            const ordered_citations = Array.from(pub_order).map(i => citations[Math.floor(i)]);
            const ordered_links = Array.from(pub_order).map(i => citation_links[Math.floor(i)]);

            // Use DocumentFragment for better DOM performance
            const fragment = document.createDocumentFragment();
            const citationsToShow = ordered_citations.slice(0, 100);

            citationsToShow.forEach((item, index) => {
                const div = document.createElement('div');
                div.className = 'spaced';
                const link = document.createElement('a');
                link.href = ordered_links[index];
                link.target = '_blank';
                link.textContent = `[${index + 1}] ${item}`;
                div.appendChild(link);
                fragment.appendChild(div);
            });

            const citationsContainer = document.getElementById("output_container");
            citationsContainer.innerHTML = '';
            citationsContainer.appendChild(fragment);
            citationsContainer.scrollTop = scrollTop;

            // Wait for remap data if not already loaded
            const reinds = await remapPromise;

            if (reinds) {
                // Process surface map
                const surface_map = processSurfaceMap(_surface_map, reinds);

                // Calculate min/max more efficiently
                let surface_min = surface_map[0];
                let surface_max = surface_map[0];

                for (let i = 1; i < surface_map.length; i++) {
                    const val = surface_map[i];
                    if (val < surface_min) surface_min = val;
                    if (val > surface_max) surface_max = val;
                }

                // Re-create dataview
                const dataviews = dataset.fromJSON({
                    "views": [{
                        "data": ["__e506186d71676e98"],
                        "state": null,
                        "attrs": {"priority": 1},
                        "desc": "",
                        "cmap": ["RdBu_r"],
                        "vmin": [surface_min],
                        "vmax": [surface_max],
                        "name": "neurovlm"
                    }],
                    "data": {
                        "__e506186d71676e98": {
                            "split": 163842,
                            "frames": 1,
                            "name": "__e506186d71676e98",
                            "subject": "fsaverage",
                            "min": surface_min,
                            "max": surface_max,
                            "raw": false
                        }
                    },
                    "images": {
                        "__e506186d71676e98": [surface_map]
                    }
                });
                viewer.addData(dataviews);
            }

            // Volume processing
            const dims = [46, 55, 46];
            const pixDims = [1, 1, 1];
            const affine = [
                4., 0., 0., -90.,
                0., 4., 0., -126.,
                0., 0., 4., -72.,
                0., 0., 0., 1.
            ];

            const datatypeCode = 16; // code for float32

            // Remove existing volume before adding new one
            if (nv1.volumes.length > 1) {
                nv1.removeVolume(nv1.volumes[1]);
            }

            try {
                const bytes = await nv1.createNiftiArray(dims, pixDims, affine, datatypeCode, volume);
                const nii = await NVImage.loadFromUrl({
                    url: bytes,
                    colormap: "magma",
                    visible: true,
                    opacity: 0.5,
                    cal_min: 0.5,
                    cal_max: 1.0
                });
                nv1.addVolume(nii);
            } catch (error) {
                console.error('Error processing volume:', error);
            }

        } catch (error) {
            console.error('Query processing error:', error);
        }
    }, 300); // 300ms debounce

    // Wait for user query with debouncing
    textbox.addEventListener('keypress', async function (event) {
        if (event.key === 'Enter' && this.value.trim() !== "") {
            event.preventDefault();
            const query = this.value.trim();
            this.value = ""; // Clear immediately for better UX
            await debouncedQuery(query);
        }
    });

    // Wait for elements to be ready in parallel
    const [dataSelection] = await Promise.all([
        waitForElement("#dataname")
    ]);

    // MNI Space setup - create elements more efficiently
    const canvas = document.createElement('canvas');
    canvas.id = 'gl1';
    canvas.style.backgroundColor = "#131314";
    canvas.classList.add("hidden");

    const toggleContainer = document.createElement("div");
    toggleContainer.className = "toggle-container";

    const fsaverageDiv = document.createElement("div");
    fsaverageDiv.className = "toggle-option active";
    fsaverageDiv.id = "fsaverage";
    fsaverageDiv.textContent = "fsaverage";

    const mniDiv = document.createElement("div");
    mniDiv.className = "toggle-option";
    mniDiv.id = "mni";
    mniDiv.textContent = "MNI";

    toggleContainer.appendChild(fsaverageDiv);
    toggleContainer.appendChild(mniDiv);
    document.body.appendChild(toggleContainer);

    const parent = document.getElementsByName("w2figure0")[0];
    parent.appendChild(canvas);

    // Initialize Niivue
    const volumeList1 = [{
        url: "/static/mni/mni152.nii.gz",
        colormap: "gray",
        visible: true,
        opacity: 1
    }];

    const nv1 = new Niivue({
        show3Dcrosshair: true,
        dragAndDropEnabled: true,
        onLocationChange: (data) => {
            document.getElementById("intensity").innerHTML = "&nbsp;&nbsp;" + data.string;
        }
    });

    // Configure Niivue options
    Object.assign(nv1.opts, {
        crosshairGap: 12,
        dragMode: nv1.dragModes.pan,
        yoke3Dto2DZoom: true,
        isResizeCanvas: true
    });

    nv1.attachTo("gl1");
    nv1.setSliceType(nv1.sliceTypeMultiplanar);

    // Load volumes asynchronously
    nv1.loadVolumes(volumeList1).catch(error => {
        console.error('Error loading volumes:', error);
    });

    // Event handlers
    const drop = document.getElementById("sliceType");
    drop.onchange = () => {
        const st = parseInt(drop.value);
        nv1.setSliceType(st);
    };

    const alphaSlider = document.getElementById("alphaSlider");
    alphaSlider.oninput = function() {
        if (nv1.volumes.length > 1) {
            nv1.volumes[1].cal_min = this.value / 255;
            nv1.updateGLVolume();
        }
    };

    // Create data options div
    const dataOptsVol = document.createElement("div");
    Object.assign(dataOptsVol.style, {
        display: "none",
        color: "white",
        textShadow: "0px 2px 8px black, 0px 1px 8px black",
        fontSize: "24pt",
        fontWeight: "bold"
    });
    dataOptsVol.textContent = "neurovlm";
    dataOptsVol.id = "dataopts-vol";
    document.body.appendChild(dataOptsVol);

    // Toggle functionality with requestAnimationFrame for smoother transitions
    const toggleOptions = document.querySelectorAll('.toggle-option');
    toggleOptions.forEach(option => {
        option.addEventListener('click', () => {
            toggleOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');

            requestAnimationFrame(() => {
                if (option.id === "mni") {
                    const div1 = document.getElementsByName("w2figure0")[0].children[0];
                    div1.classList.remove("visible");
                    div1.classList.add("hidden");

                    canvas.classList.remove("hidden");
                    canvas.classList.add("visible");

                    dataOptsVol.style.display = "flex";
                    alphaSlider.style.display = "flex";

                    // Force reflow for Niivue
                    requestAnimationFrame(() => {
                        nv1.canvas.width = nv1.canvas.offsetWidth * nv1.uiData.dpr;
                        nv1.canvas.height = nv1.canvas.offsetHeight * nv1.uiData.dpr;
                        nv1.gl.viewport(0, 0, nv1.gl.canvas.width, nv1.gl.canvas.height);
                        nv1.drawScene();
                    });
                } else {
                    canvas.classList.remove("visible");
                    canvas.classList.add("hidden");
                    dataOptsVol.style.display = "none";
                    alphaSlider.style.display = "none";

                    const div2 = document.getElementsByName("w2figure0")[0].children[0];
                    div2.classList.remove("hidden");
                    div2.classList.add("visible");
                }
            });
        });
    });

    // Optimized loading indicator handler
    async function waitForDivToHide(divId) {
        const div = document.getElementById(divId);
        if (!div) return Promise.resolve();

        return new Promise(resolve => {
            if (div.style.display === "none") {
                resolve();
                return;
            }

            const observer = new MutationObserver(() => {
                if (div.style.display === "none") {
                    observer.disconnect();
                    resolve();
                }
            });

            observer.observe(div, {
                attributes: true,
                attributeFilter: ["style"]
            });
        });
    }

    // Handle loading completion
    waitForDivToHide("ctmload").then(() => {
        setTimeout(() => {
            const loader = document.getElementById("loader");
            if (loader) {
                loader.style.display = "none";
            }
        }, 500); // Reduced from 1000ms
    });
}

// Use DOMContentLoaded for faster initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
} else {
    run();
}