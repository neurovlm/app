var validator = new function() {
var gl;
this.webgl = false;
try {
if (window.WebGLRenderingContext) {
gl = document.createElement("canvas").getContext('experimental-webgl');
this.webgl = !! gl;
}
} catch (e) {}
this.vtex = function(vtmin, atmin) {
var vtex = gl.getParameter(gl.MAX_VERTEX_TEXTURE_IMAGE_UNITS);
var atex = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
if (vtex >= vtmin && atex >= atmin)
return true;
return false;
}
this.vattr = function(vmin) {
return gl.getParameter(gl.MAX_VERTEX_ATTRIBS) >= vmin;
}
};

var viewer, subjects, datasets, figure, sock, viewopts;

var colormaps = {};
$(document).ready(function() {
if (!validator.webgl) {
$(".loadmsg").hide();
$("#ctmload").html("<p>Sorry, it seems you have no <a href='http://get.webgl.org'>WebGL support.</a> :(</p><p>If you are using Chrome, try <a href='http://peter.sh/experiments/chromium-command-line-switches/#ignore-gpu-blacklist'>ignoring the GPU blacklist.</a></p>").show();
} else if (!validator.vtex(3, 5)) {
$(".loadmsg").hide();
$("#ctmload").html("<p>Sorry, your computer does not have the minimum number of texture units :(</p><p>Try upgrading your drivers and/or your browser</p>").show();
} else {

viewopts = {"voxlines": "false", "voxline_color": "#FFFFFF", "voxline_width": "0.01", "specularity": "1.0", "overlayscale": "1", "anim_speed": "2", "bumpy_flatmap": "false", "overlays_visible": ["rois", "sulci"], "labels_visible": ["rois"], "brightness": "0.5", "contrast": "0.25", "smoothness": "0.0", "dependency_paths": {"inkscape": "inkscape", "blender": "blender", "slim": "None", "meshlab": "None"}, "paths_default": {"stroke": "white", "fill": "none", "display": "inline", "filter": "url(#dropshadow)", "stroke-width": "3", "stroke-opacity": "1", "fill-opacity": "0", "stroke-dashoffset": "None", "stroke-dasharray": "None"}, "labels_default": {"font-family": "Helvetica, sans-serif", "font-size": "18pt", "font-weight": "bold", "font-style": "italic", "fill": "white", "fill-opacity": "1", "text-anchor": "middle", "filter": "url(#dropshadow)"}, "overlay_paths": {"stroke": "white", "fill": "none", "display": "inline", "filter": "url(#dropshadow)", "stroke-width": "2", "stroke-opacity": "1", "fill-opacity": "0", "stroke-dashoffset": "None", "stroke-dasharray": "None"}, "rois_paths": {"stroke": "white", "fill": "none", "display": "inline", "filter": "url(#dropshadow)", "stroke-width": "2", "stroke-opacity": "1", "fill-opacity": "0", "stroke-dashoffset": "None", "stroke-dasharray": "None"}, "rois_labels": {"font-family": "Helvetica, sans-serif", "font-size": "14pt", "font-weight": "bold", "font-style": "italic", "fill": "white", "fill-opacity": "1", "text-anchor": "middle", "filter": "url(#dropshadow)"}, "sulci_paths": {"stroke": "white", "fill": "none", "display": "inline", "filter": "url(#dropshadow)", "stroke-width": "6", "stroke-opacity": "0.6", "fill-opacity": "0", "stroke-dashoffset": "None", "stroke-dasharray": "None", "stroke-linecap": "round"}, "sulci_labels": {"font-family": "Helvetica, sans-serif", "font-size": "16pt", "font-weight": "", "font-style": "", "fill": "white", "fill-opacity": "1", "text-anchor": "middle", "filter": "url(#dropshadow)"}};
subjects = {"fsaverage": "fsaverage_[inflated]_mg2_9_v3.json"};
for (var name in subjects) {
subjects[name] = new mriview.Surface(subjects[name]);
}
figure = new jsplot.W2Figure();
viewer = figure.add(mriview.Viewer, "main", true);
dataviews = dataset.fromJSON({"views": [{"data": ["__e506186d71676e98"], "state": null, "attrs": {"priority": 1}, "desc": "", "cmap": ["RdBu_r"], "vmin": [0.009942443989932506], "vmax": [0.9896497896283171], "name": "data"}], "data": {"__e506186d71676e98": {"split": 163842, "frames": 1, "name": "__e506186d71676e98", "subject": "fsaverage", "min": 1.0484371415486748e-07, "max": 0.9999994752468082, "raw": false}}, "images": {"__e506186d71676e98": ["data/__e506186d71676e98_0.png"]}});
viewer.addData(dataviews);

}
});
