import {Niivue, NVImage} from "./js/niivue.min.js";

const worker = new Worker('/static/js/lzma-worker.js');

// Cache for frequently used data
const cache = {
    remap: null,
    links: null,
    titles: null
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

async function preloadTitlesData() {
    if (!cache.remap) {
        try {
            const response = await fetch('/static/models/titles.txt');
            const data = await response.text();
            cache.titles = data.split('\n').filter(line => line.trim() !== '');
        } catch (error) {
            console.error('Error preloading titles data:', error);
            cache.titles = [];
        }
    }
    return cache.titles;
}

async function preloadLinksData() {
    if (!cache.remap) {
        try {
            const response = await fetch('/static/models/links.txt');
            const data = await response.text();
            cache.titles = data.split('\n').filter(line => line.trim() !== '');
        } catch (error) {
            console.error('Error preloading links data:', error);
            cache.titles = [];
        }
    }
    return cache.titles;
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

// slm
const responseBox = document.getElementById("response_box");
const llmDiv = document.createElement("div");
llmDiv.id = "llm";
llmDiv.style.height = '40%';
llmDiv.style.flex = "1";
llmDiv.style.width = "96%";
llmDiv.style.display = "flex";
llmDiv.style.left = "5%";
llmDiv.style.position = "relative";
llmDiv.style.fontSize = "16px";
llmDiv.style.color = "white";
llmDiv.style.top = "5%";
llmDiv.style.textAlign = "justify";
llmDiv.style.overflowY = "auto";
responseBox.appendChild(llmDiv)

let scrollTop = 0;
const citationsContainer = document.getElementById("output_container");
scrollTop = citationsContainer?.scrollTop || 0;
const div = document.createElement('div');
const fragment = document.createElement("div");
fragment.appendChild(div);
citationsContainer.innerHTML = '';
fragment.style.height = '50%';
fragment.style.flex = "1";
citationsContainer.appendChild(fragment);
citationsContainer.scrollTop = scrollTop;
citationsContainer.style.height = '50%';


// slm
async function sendStreamRequest(query) {

    console.log("sending to lm.");
    const outputDiv = document.getElementById('llm');
    outputDiv.textContent = ''; // Clear previous output

    const response = await fetch('/llm/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: query })
    });

    if (!response.ok) {
        outputDiv.textContent = 'Error: ' + response.statusText;
        return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        outputDiv.textContent += chunk;
    }
}




async function run() {
    // Pre-load remap data in parallel
    const remapPromise = preloadRemapData();
    const linksPromise = preloadLinksData();
    const titlesPromise = preloadTitlesData();

    // Get text box input
    const textbox = document.getElementById("txid");

    // Scroll through publications after query
    let scrollTop = 0;
    const outputContainer = document.getElementById("output_container");
    outputContainer.style.height = '100%';

    // Create a debounced query handler
    const debouncedQuery = debounce(async function(query) {
        try {
            // Store scroll position
            scrollTop = outputContainer?.scrollTop || 0;
            console.log(query);

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
            const surface_ = out.surface;
            const volume = new Float32Array(out.volume);
            const pub_order = out.puborder;

            // Process citations in batches to avoid blocking
            const titles = await titlesPromise;
            const links = await linksPromise;

            const ordered_citations = Array.from(pub_order).map(i => titles[Math.floor(i)]);
            const ordered_links = Array.from(pub_order).map(i => links[Math.floor(i)]);

            // Use DocumentFragment for better DOM performance
            const fragment = document.createElement("div");
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
            fragment.style.height = '50%';
            fragment.style.flex = "1";
            citationsContainer.appendChild(fragment);
            citationsContainer.scrollTop = scrollTop;
            citationsContainer.style.height = '50%';

            // Wait for remap data if not already loaded
            const reinds = await remapPromise;

            if (reinds) {
                // Process surface map
                const surface_map = new Float32Array(327684);
                for (let i = 0; i < 163842; i++) {
                    surface_map[i] = surface_[reinds[i]];
                    surface_map[i + 163842] = surface_[reinds[i + 163842]];
                }
                let surface_min = surface_map[0];
                let surface_max = surface_map[0];
                for (let i = 1; i < surface_map.length; i++) {
                    const val = surface_map[i];
                    if (val < surface_min) surface_min = val;
                    if (val > surface_max) surface_max = val;
                }
                // let surface_min = 0.0;
                // let surface_max = 1.0;

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
                    opacity: 0.8,
                    cal_min: 0.1,
                    cal_max: 1.0
                });

                // Precomputed transformation matrix
                const t = [
                    5.42400005033472, 0.0, 0.0, -19.306002583559717,
                    0.0, 5.42400005033472, 0.0, -20.662002596143395,
                    0.0, 0.0, 5.42400005033472, -0.3220024073881831,
                    0.0, 0.0, 0.0, 1.0
                ]

                // Get argmax and transform to template coordinates
                const maxIdx = volume.indexOf(Math.max(...volume));
                const [i_4mm, j_4mm, k_4mm] = [maxIdx%46, Math.floor((maxIdx%2530)/46), Math.floor(maxIdx/2530)];
                const [i, j, k] = [
                    Math.round(t[0]*i_4mm + t[1]*j_4mm + t[2]*k_4mm + t[3]),
                    Math.round(t[4]*i_4mm + t[5]*j_4mm + t[6]*k_4mm + t[7]),
                    Math.round(t[8]*i_4mm + t[9]*j_4mm + t[10]*k_4mm + t[11])
                ];
                // nv1.scene.crosshairPos = [i, j, k];
                // nv1.onLocationChange = function(e) { console.log(e) }
                nv1.addVolume(nii);
                nv1.scene.crosshairPos = nv1.vox2frac([i, j, k]);
                nv1.drawScene();
                nv1.createOnLocationChange();
                document.getElementById("alphaSlider").value = 0.1 * 255;
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
            await sendStreamRequest(query);
        }
    });

    // Wait for elements to be ready in parallel
    const [dataSelection] = await Promise.all([
        waitForElement("#dataname")
    ]);

    // MNI Space setup
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

    toggleContainer.appendChild(mniDiv);
    toggleContainer.appendChild(fsaverageDiv);

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
        backColor: [0.075, 0.075, 0.078, 1],
        backgroundColor: [0.075, 0.075, 0.078, 1],
        show3Dcrosshair: true,
        dragAndDropEnabled: true,
        onLocationChange: (data) => {
            document.getElementById("intensity").innerHTML = "&nbsp;&nbsp;" + data.string;
        }
    });
    nv1.isAlphaClipDark = true;

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

    function switchToView(viewId) {
        if (viewId === "mni") {
            const div1 = document.getElementsByName("w2figure0")[0].children[0];
            div1.classList.remove("visible");
            div1.classList.add("hidden");
            canvas.classList.remove("hidden");
            canvas.classList.add("visible");
            dataOptsVol.style.display = "flex";
            alphaSlider.style.display = "flex";

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
    }

    const toggleOptions = document.querySelectorAll('.toggle-option');
    toggleOptions.forEach(option => {
        option.addEventListener('click', () => {
            toggleOptions.forEach(opt => opt.classList.remove('active'));
            option.classList.add('active');
            requestAnimationFrame(() => {
                switchToView(option.id);
            });
        });
    });

    // Loading indicator handler
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
            // Set mni as activen
            const mniOption = document.getElementById('mni');
            toggleOptions.forEach(opt => opt.classList.remove('active'));
            mniOption.classList.add('active');
            switchToView("mni");
        }, 500); // Reduced from 1000ms
    });
}

// Use DOMContentLoaded for faster initialization
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', run);
} else {
    run();
}