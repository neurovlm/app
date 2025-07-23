//monkeypatch a remove folder command
dat.GUI.prototype.removeFolder = function(name) {
    this.__folders[name].close();
    this.__folders[name].domElement.parentNode.parentNode.removeChild(this.__folders[name].domElement.parentNode);
    this.__folders[name] = undefined;
    this.onResize();
}

var jsplot = (function (module) {

    module.Color = function(data) {
        this.color = Color.colors[data];
    }
    module.Color.prototype.toRGBA = function() {
        return "rgba("+(this.color[0]*255)+", "+(this.color[1]*255)+", "+(this.color[2]*255)+", "+(this.color[3]*255)+")";
    }
    module.Color.colors = {
        'k': [0,0,0,1],
        'r': [1,0,0,1],
        'g': [0,1,0,1],
        'b': [0,0,1,1],
    }

    module.construct = function(cls, args) {
        function F() {
            return cls.apply(this, args);
        }
        F.prototype = cls.prototype;
        return new F();
    }

    module.Figure = function(parent) {

        this._notifying = false;
        this.axes = [];
        this.ax = null;
        this._registrations = {};

        this.object = document.createElement("div");
        this.object.className = 'jsplot_figure';
        this.parent = parent;
        if (parent === undefined) {
            $(document.body).append(this.object);
            this.init();
            window.addEventListener('resize', this.resize.bind(this));
        } else {
            parent.addEventListener('resize', this.resize.bind(this));
        }

        this.gui = new dat.GUI({autoPlace:false});
        this.gui.close();
        this.ui_element = document.createElement("div");
        this.ui_element.id = "figure_ui";
        this.object.appendChild(this.ui_element);
        this.ui_element.appendChild(this.gui.domElement);

    }
    THREE.EventDispatcher.prototype.apply(module.Figure.prototype);

    module.Figure.prototype.init = function() {}
    module.Figure.prototype.register = function(eventType, self, func) {
        if (this.parent && this.parent instanceof module.Figure) {
            this.parent.register(eventType, self, func);
        } else {
            if (!(this._registrations[eventType] instanceof Array))
                this._registrations[eventType] = [];

            var register = function(evt) {
                if (evt.self != self)
                    func.apply(self, evt.args);
            }.bind(this);
            this._registrations[eventType].push([self, register]);

            this.addEventListener(eventType, register);
        }
    }
    module.Figure.prototype.unregister = function(eventType, self) {
        var objects = this._registrations[eventType];
        for (var i = 0; i < objects.length; i++) {
            if (objects[i][0] === self)
                this.removeEventListener(eventType,objects[i][1]);
        }
    }
    module.Figure.prototype.notify = function(eventType, self, arguments) {
        if (this.parent && this.parent instanceof module.Figure) {
            this.parent.notify(eventType, self, arguments);
        } else {
            if (!this._notifying) {
                this._notifying = true;
                this.dispatchEvent({type:eventType, self:self, args:arguments});
                this._notifying = false;
            }
        }
    }
    module.Figure.prototype.resize = function(width, height) {
        if (width !== undefined)
            $(this.object).width(width);
        if (height !== undefined)
            $(this.object).height(height);
        var w = $(this.object).width();
        var h = $(this.object).height();
        this.dispatchEvent({type:'resize', width:w, height:h});
    }
    module.Figure.prototype.close = function() {
        window.close();
    }
    module.Figure.prototype.add = function(axcls) {
        var args = Array.prototype.slice.call(arguments).slice(1);
        args.unshift(this);
        this.ax = module.construct(axcls, args);
        this.axes.push(this.ax);
        $(this.parent.object).append(this.ax.object);
        if (this.ax.ui !== undefined)
            this.ax.ui.init(this.gui);
    }

    var w2fig_layer = 0;
    module.W2Figure = function(parent) {
        this._resizing = true;
        module.Figure.call(this, parent);
        this.axes = {};
    }
    module.W2Figure.prototype = Object.create(module.Figure.prototype);
    module.W2Figure.prototype.constructor = module.W2Figure;
    module.W2Figure.prototype.init = function() {
        //var style = "border:1px solid #dfdfdf; padding:5px;";
        this.w2obj = $(this.object).w2layout({
            name: 'w2figure'+(w2fig_layer++),
            panels: [
                { type: 'top', resizable: true, hidden: true },
                { type: 'bottom', resizable: true, hidden: true },
                { type: 'left', resizable: true, hidden: true },
                { type: 'right', resizable: true, hidden: true },
                { type: 'main' },
            ],
        });
        this.w2obj.onResize = this.resize.bind(this);
        this._resizing = false;
    }
    module.W2Figure.prototype.resize = function() {
        if (!this._resizing) {
            this._resizing = true;
            this.w2obj.resize();
            module.Figure.prototype.resize.call(this);
            this._resizing = false;
        }
    }
    module.W2Figure.prototype.add = function(axcls, where, instant) {
        var args = Array.prototype.slice.call(arguments).slice(3);
        args.unshift(this);

        var axes = module.construct(axcls, args);
        this.w2obj.show(where, instant);
        this.w2obj.content(where, axes.object);
        if (axes instanceof module.Figure) {
            axes.init();
        }
        this.axes[where] = axes;
        this.ax = axes;

        if (this.ax.ui !== undefined)
            this.ax.ui.init(this.gui);

        return axes;
    }
    module.W2Figure.prototype.show = function(where, instant) {
        this.w2obj.show(where, instant);
    }
    module.W2Figure.prototype.hide = function(where, instant) {
        this.w2obj.hide(where, instant);
    }
    module.W2Figure.prototype.toggle = function(where, instant) {
        this.w2obj.toggle(where, instant);
    }
    module.W2Figure.prototype.setSize = function(where, size) {
        if (typeof(size) == "string" && size[size.length-1] == '%') {
            var prop = parseFloat(size.slice(0, size.length-1)) / 100;
            size = $(this.object).width() * prop;
        }
        this.w2obj.set(where, {size:size});
        this.w2obj.resize();
    }
    module.W2Figure.prototype.getSize = function(where) {
        return this.w2obj.get(where).size;
    }

    module.GridFigure = function(parent, nrows, ncols) {
        module.Figure.call(this, parent);

        this.nrows = nrows;
        this.ncols = ncols;

        this.axes = [];
        this.cells = [];
        var table = document.createElement("table");
        this.object.appendChild(table);
        for (var i = 0; i < nrows; i++) {
            var tr = document.createElement("tr");
            tr.style.height = (100 / nrows)+"%";
            table.appendChild(tr);
            for (var j = 0; j < ncols; j++) {
                var td = document.createElement('td');
                td.style.width = (100 / ncols)+'%';
                //td.style.height = "100%";
                tr.appendChild(td);
                this.cells.push(td);
                this.axes.push(null);
            }
        }
    }
    module.GridFigure.prototype = Object.create(module.Figure.prototype);
    module.GridFigure.prototype.constructor = module.GridFigure;
    module.GridFigure.prototype.add = function(axcls, where) {
        var args = Array.prototype.slice.call(arguments).slice(2);
        args.unshift(this);
        this.ax = module.construct(axcls, args);
        this.axes[where] = this.ax;
        this.cells[where].appendChild(this.ax.object);
        return this.ax;
    }


    module.Axes = function(figure) {
        this.figure = figure;
        if (this.object === undefined) {
            this.object = document.createElement("div");
            this.object.className = "jsplot_axes";

            this.figure.addEventListener("resize", this.resize.bind(this));
        }

        // color legend
        function formatState (state) {
            if (!state.id) { return state.text; }
            var $state = $('<span class="colorlegend-option"><img class="colorlegend-option-image" src="' + colormaps[state.text].image.currentSrc + '" class="img-flag" />' + state.text + '</span>');
            return $state;
        };
        $(document).ready(function() {
            var selector = $(".colorlegend-select").select2({
                templateResult: formatState
            });
            $("#colorlegend-colorbar").on('click', function() {
                selector.show();
                selector.select2('open');
            });
            $('#brain').on('click', function () { selector.select2("close"); })
        });

    }
    THREE.EventDispatcher.prototype.apply(module.Axes.prototype);
    module.Axes.prototype.resize = function() {}

    module.MovieAxes = function(figure, url) {
        module.Axes.call(this, figure);
        $(this.object).html($("#movieaxes_html").html());
        this._target = null;
        var types = {
            ogv: 'video/ogg; codecs="theora, vorbis"',
            webm: 'video/webm; codecs="vp8, vorbis"',
            mp4: 'video/mp4; codecs="h264, aac"'
        }
        var src = $(this.object).find("source");
        var ext = url.match(/^(.*)\.(\w{3,4})$/);
        src.attr("type", types[ext]);
        src.attr("src", url);

        this.loadmsg = $(this.object).find("div.movie_load");
        this.movie = $(this.object).find("video")[0];

        this._update_func = function() {
            this.figure.notify("playsync", this, [this.movie.currentTime]);
        }.bind(this);
        this._progress_func = function() {
            if (this._target != null &&
                this.movie.seekable.length > 0 &&
                this.movie.seekable.end(0) >= this._target &&
                this.movie.parentNode != null) {
                var func = function() {
                    try {
                        this.movie.currentTime = this._target;
                        this._target = null;
                        this.loadmsg.hide()
                    } catch (e) {
                        console.log(e);
                        setTimeout(func, 5);
                    }
                }.bind(this);
                func();
            }
        }.bind(this);
        this.movie.addEventListener("timeupdate", this._update_func);
        this.movie.addEventListener("progress", this._progress_func);
        this.figure.register("playtoggle", this, this.playtoggle.bind(this));
        this.figure.register("setFrame", this, this.setFrame.bind(this));
    }
    module.MovieAxes.prototype = Object.create(module.Axes.prototype);
    module.MovieAxes.prototype.constructor = module.MovieAxes;
    module.MovieAxes.prototype.destroy = function() {
        this.figure.unregister("playtoggle", this);
        this.figure.unregister("setFrame", this);
        this.movie.removeEventListener("timeupdate", this._update_func);
    }
    module.MovieAxes.prototype.setFrame = function(time) {
        if (this.movie.seekable.length > 0 &&
            this.movie.seekable.end(0) >= time) {
            this.movie.currentTime = time;
            this.loadmsg.hide()
        } else {
            this._target = time;
            this.loadmsg.show()
        }
    }
    module.MovieAxes.prototype.playtoggle = function(state) {
        if (!this.movie.paused && state == "pause")
            this.movie.pause();
        else
            this.movie.play();
        this.figure.notify("playtoggle", this, [this.movie.paused?"pause":"play"]);
    }

    module.ImageAxes = function(figure) {
        module.Axes.call(this, figure);
    }
    module.ImageAxes.prototype = Object.create(module.Axes.prototype);
    module.ImageAxes.prototype.constructor = module.ImageAxes;
    module.ImageAxes.prototype.set = function(url) {
        $(this.object).fadeTo(0);
        var img = new Image();
        img.onload = function() {
            $(this.object).html(img);
            $(this.object).fadeTo(1);
        }.bind(this);
        img.src = url;
    }

    return module;
}(jsplot || {}));

var jsplot = (function (module) {

var retina_scale = window.devicePixelRatio || 1;

module.Axes3D = function(figure) {
    if (this.canvas === undefined) {
        module.Axes.call(this, figure);
        this.canvas = $(document.createElement("canvas"));
        this.object.appendChild(this.canvas[0]);
    }

    // scene and camera
    this.camera = new THREE.PerspectiveCamera( 45, this.canvas.width()/this.canvas.height(), 1., 1000. );
    this.camera.up.set(0,0,1);
    this.camera.position.set(0, -500, 0);
    this.camera.lookAt(new THREE.Vector3(0,0,0));

    this.controls = new jsplot.LandscapeControls();
    this.controls.bind(this.canvas[0]);
    this.controls.addEventListener("pick", this.pick.bind(this));
    this.controls.addEventListener("change", this.schedule.bind(this));

    //These lights approximately match what's done by vtk
    this.lights = [
        new THREE.DirectionalLight(0xffffff, .47),
        new THREE.DirectionalLight(0xffffff, .29),
        new THREE.DirectionalLight(0xffffff, .24)
    ];
    this.lights[0].position.set( 1, -1, -1 ).normalize();
    this.lights[1].position.set( -1, -.25, .75 ).normalize();
    this.lights[2].position.set( 1, -.25, .75 ).normalize();
    this.camera.add( this.lights[0] );
    this.camera.add( this.lights[1] );
    this.camera.add( this.lights[2] );

    this.views = [];

    // renderer
    this.renderer = new THREE.WebGLRenderer({
        alpha:true,
        antialias: retina_scale == 1,
        preserveDrawingBuffer:true,
        canvas:this.canvas[0],
    });
    this.renderer.setClearColor(new THREE.Color("#131314"), 1);
    this.renderer.sortObjects = true;

    this.state = "pause";
    this._startplay = null;
    this._animation = null;

    this._schedule = function() {
        this.draw();
        if (this.state == "play" || this._animation != null) {
            this.schedule();
        }
        //requestAnimationFrame(this._schedule);
    }.bind(this);

    //Figure registrations
    this.figure.register("playsync", this, function(time) {
        if (this._startplay != null)
            this._startplay = (new Date()) - (time * 1000);
    });
    this.figure.register("playtoggle", this, this.playpause.bind(this));
    this.figure.register("setFrame", this, this.setFrame.bind(this));

    this.root = new THREE.Group();
    this.root.name = 'root';
}
module.Axes3D.prototype = Object.create(module.Axes.prototype);
module.Axes3D.prototype.constructor = module.Axes3D;
module.Axes3D.prototype.resize = function(width, height) {
    if (width !== undefined) {
        $(this.object).find("#brain, #roilabels").css("width", width);
        width = $(this.object).width();
    }
    var w = width === undefined ? $(this.object).width()  : width;
    var h = height === undefined ? $(this.object).height()  : height;
    var aspect = w / h;

    this.width = w;
    this.height = h;

    this.renderer.setSize( w , h );
    this.renderer.domElement.style.width = w + 'px';
    this.renderer.domElement.style.height = h + 'px';

    this.camera.aspect = aspect;
    this.camera.updateProjectionMatrix();

    this.dispatchEvent({ type:"resize", width:w, height:h});
    this.schedule();
};
module.Axes3D.prototype.schedule = function() {
    if (!this._scheduled) {
        this._scheduled = true;
        requestAnimationFrame( this._schedule );
    }
};
module.Axes3D.prototype.draw = function () {
    var now = new Date();
    if (this.state == "play")
        this.setFrame((now - this._startplay) / 1000);

    if (this._animation) {
        var atime = (now - this._animation.start) / 1000;
        if (!(this._animate(atime)))
            delete this._animation;
    }

    this.controls.update(this.camera);

    var view, left, bottom, width, height;
    if (this.views.length > 1) {
        for (var i = 0; i < this.views.length; i++) {
            view = this.views[i];
            left   = Math.floor( this.width  * view.left );
            bottom = Math.floor( this.height * view.bottom );
            width  = Math.floor( this.width  * view.width );
            height = Math.floor( this.height * view.height );

            this.renderer.setViewport( left, bottom, width, height );
            this.renderer.setScissor( left, bottom, width, height );
            this.renderer.enableScissorTest ( true );

            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.drawView(this.views[i].scene, i);
        }
    } else if (this.views.length > 0) {
        this.renderer.enableScissorTest(false);
        this.drawView(this.views[0].scene, 0);
    }
    this._scheduled = false;
    this.dispatchEvent({type:"draw"});

    //requestAnimationFrame( this._schedule );
};
module.Axes3D.prototype.drawView = function(scene) {
    this.renderer.render(scene, this.camera);
};
module.Axes3D.prototype.animate = function(animation) {
    var state = {};
    var anim = [];
    animation.sort(function(a, b) { return a.idx - b.idx});
    for (var i = 0, il = animation.length; i < il; i++) {
        var f = animation[i];
        if (f.idx == 0) {
            this.setState(f.state, f.value);
            state[f.state] = {idx:0, val:f.value};
        } else {
            if (state[f.state] === undefined)
                state[f.state] = {idx:0, val:this.getState(f.state)};
            var start = {idx:state[f.state].idx, state:f.state, value:state[f.state].val}
            var end = {idx:f.idx, state:f.state, value:f.value};
            state[f.state].idx = f.idx;
            state[f.state].val = f.value;
            if (start.value instanceof Array) {
                var test = true;
                for (var j = 0; test && j < start.value.length; j++)
                    test = test && (start.value[j] == end.value[j]);
                if (!test)
                    anim.push({start:start, end:end, ended:false});
            } else if (start.value != end.value)
                anim.push({start:start, end:end, ended:false});
        }
    }
    if (this.active.fastshader) {
        this.meshes.left.material = this.active.fastshader;
        this.meshes.right.material = this.active.fastshader;
    }
    this._animation = {anim:anim, start:new Date()};
    this.schedule();
};
module.Axes3D.prototype.pick = function(evt) {
    var x = (evt.x / this.width)*2 - 1;
    var y = (evt.y / this.height)*2 - 1;
    var vector = new THREE.Vector3(x, y, 1).unproject(this.camera);
    this.raycaster.set(this.camera.position, vector.sub(this.camera.position).normalize());
    for (var i = 0; i < this.surfs.length; i++) {
        var intersects = this.raycaster.intersectObject(this.surfs[i].object, true);
        if (intersects.length > 0 && this.surfs[i].pick) {
            this.surfs[i].pick(intersects);
        }
    }
}
module.Axes3D.prototype._animate = function(sec) {
    var state = false;
    var idx, val, f, i, j;
    for (i = 0, il = this._animation.anim.length; i < il; i++) {
        f = this._animation.anim[i];
        if (!f.ended) {
            if (f.start.idx <= sec && sec < f.end.idx) {
                idx = (sec - f.start.idx) / (f.end.idx - f.start.idx);
                if (f.start.value instanceof Array) {
                    val = [];
                    for (j = 0; j < f.start.value.length; j++) {
                        //val.push(f.start.value[j]*(1-idx) + f.end.value[j]*idx);
                        val.push(this._animInterp(f.start.state, f.start.value[j], f.end.value[j], idx));
                    }
                } else {
                    //val = f.start.value * (1-idx) + f.end.value * idx;
                    val = this._animInterp(f.start.state, f.start.value, f.end.value, idx);
                }
                this.setState(f.start.state, val);
                state = true;
            } else if (sec >= f.end.idx) {
                this.setState(f.end.state, f.end.value);
                f.ended = true;
            } else if (sec < f.start.idx) {
                state = true;
            }
        }
    }
    return state;
};
module.Axes3D.prototype._animInterp = function(state, startval, endval, idx) {
    switch (state) {
        case 'azimuth':
            // Azimuth is an angle, so we need to choose which direction to interpolate
            if (Math.abs(endval - startval) >= 180) { // wrap
                if (startval > endval) {
                    return (startval * (1-idx) + (endval+360) * idx + 360) % 360;
                }
                else {
                    return (startval * (1-idx) + (endval-360) * idx + 360) % 360;
                }
            }
            else {
                return (startval * (1-idx) + endval * idx);
            }
        default:
            // Everything else can be linearly interpolated
            return startval * (1-idx) + endval * idx;
    }
};
module.Axes3D.prototype.playpause = function(state) {
    if (this.state == "play") {
        this.state = "pause";
    } else {
        //Start playing
        this._startplay = (new Date()) - (this.frame * 1000);
        this.state = "play";
        this.schedule();
    }
    this.figure.notify("playtoggle", this, [this.state]);
};
module.Axes3D.prototype.setGrid = function(m, n, idx) {
    //Establish of grid of views M rows by N columns
    if (this.views.length != m*n) {
        this.views = [];
        var scene, left, bottom, width = 1 / m, height = 1 / n;
        for (var i = 0; i < m; i++) {
            for (var j = 0; j < n; j++) {
                left = i / m;
                bottom = (n-i) / n;
                scene = new THREE.Scene();
                scene.add(this.camera);
                scene.add(this.root);
                // scene.fsquad = new THREE.Mesh(new THREE.PlaneGeometry(100, 100), null);
                // scene.fsquad.position.z = -1.0001;
                this.views.push({left:left, bottom:bottom, width:width, height:height, scene:scene});
            }
        }
    }
    return this.views[idx].scene;
}

module.Axes3D.prototype.getImage = function(width, height, post) {
    if (width === undefined)
        width = this.canvas.width();

    if (height === undefined)
        height = width * this.canvas.height() / this.canvas.width();

    console.log(width, height);
    var renderbuf = new THREE.WebGLRenderTarget(width, height, {
        minFilter: THREE.LinearFilter,
        magFilter: THREE.LinearFilter,
        format:THREE.RGBAFormat,
        stencilBuffer:false,
    });

    var clearAlpha = this.renderer.getClearAlpha();
    var clearColor = this.renderer.getClearColor();
    var oldw = this.canvas.width(), oldh = this.canvas.height();
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.renderer.setClearColor(new THREE.Color(0,0,0), 0);
    this.renderer.render(this.views[0].scene, this.camera, renderbuf);
    this.renderer.setSize(oldw, oldh);
    this.renderer.setClearColor(new THREE.Color(0,0,0), 1);
    this.camera.aspect = oldw / oldh;
    this.camera.updateProjectionMatrix();

    var img = mriview.getTexture(this.renderer.context, renderbuf)
    if (post !== undefined)
        $.post(post, {png:img.toDataURL()});
    // Draw again -- for some reason the scene disappears after getting the texture
    this.draw()
    return img;
};

return module;
}(jsplot || {}));
