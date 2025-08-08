var jsplot = (function (module) {
	var STATE = { NONE : -1, ROTATE : 0, PAN : 1, ZOOM : 2 };
	module.LandscapeControls = function() {
		this.target = new THREE.Vector3();
		this._foldedtarget = new THREE.Vector3();
		this._flattarget = new THREE.Vector3();
		this._flattarget.y = -60;

		this.azimuth = 45;
		this._foldedazimuth = 45;
		this._flatazimuth = 180;

		this.altitude = 75;
		this._foldedaltitude = 75;
		this._flataltitude = 0.1;

		this.radius = 400;

		this.mix = 0;

		this.rotateSpeed = 0.4;
		this.panSpeed = 0.3;
		this.zoomSpeed = 0.002;
		this.clickTimeout = 200; // milliseconds

		this.friction = .9;

		this._start = new THREE.Vector2();
		this._end = new THREE.Vector2();

		this._momentum = {change:[0,0]};
		this._state = STATE.NONE;

		this.twodbutton = $(document.createElement('button'));
		this.twodbutton.attr('id', 'twodbutton');
		this.twodbutton.html('2d');
		// this.twodbutton.click(this.set2d.bind(this));
	}
	THREE.EventDispatcher.prototype.apply(module.LandscapeControls.prototype);

	module.LandscapeControls.prototype._position = function() {
		var altrad = this.altitude*Math.PI / 180;
		var azirad = (this.azimuth+90)*Math.PI / 180;

		return new THREE.Vector3(
			this.radius*Math.sin(altrad)*Math.cos(azirad),
			this.radius*Math.sin(altrad)*Math.sin(azirad),
			this.radius*Math.cos(altrad)
		);
	}

	module.LandscapeControls.prototype.set2d = function() {
		this.setAzimuth(180);
		this.setAltitude(0.1);
	}

	module.LandscapeControls.prototype.update = function(camera) {
		var func;
		if (this._state != STATE.NONE) {
			if (this._state == STATE.ROTATE)
				func = this.rotate
			else if (this._state == STATE.PAN && this.mix == 1)
				func = this.rotate
			else if (this._state == STATE.PAN)
				func = this.pan
			else if (this._state == STATE.ZOOM)
				func = this.zoom

			var mousechange = this._end.clone().sub(this._start);
			func.call(this, mousechange.x, mousechange.y);
		}

		if (Math.abs(this._momentum.change[0]) > .05) {
			this._momentum.change[0] *= this.friction;
			this._momentum.change[1] *= this.friction;
		//	console.log(this._momentum.change);
			this._momentum.func.apply(this, this._momentum.change);
			setTimeout(function() {
				this.dispatchEvent( { type: "change" } );
			}.bind(this), 0);
		}

		camera.position.addVectors( this.target, this._position() );
		camera.lookAt( this.target );
		this._start = this._end;
	}

	module.LandscapeControls.prototype.rotate = function(x, y) {
		// in FLAT mode (mix = 1), PAN and ROTATE are reversed
		var mix;
		if (this._state != STATE.PAN) {
			mix = Math.pow(this.mix, 2);
			this.pan(x * mix, y * mix);
		} else {
			mix = 0;
		}

		var rx = x  * (1 - mix), ry = y * (1 - mix);
		this.setAzimuth(this.azimuth - this.rotateSpeed * rx);
		this.setAltitude(this.altitude - this.rotateSpeed * ry);

		this._momentum.change = [x, y];
		this._momentum.func = this.rotate;
	}

	var _upvec = new THREE.Vector3(0,0,1);
	module.LandscapeControls.prototype.pan = function(x, y) {
		var eye = this._position();

		var right = eye.clone().cross( _upvec );
		var up = right.clone().cross(eye);
		var pan = right.setLength( this.panSpeed * x ).add(
			up.setLength( this.panSpeed * y ));
		this.setTarget((new THREE.Vector3).copy(this.target).add(pan).toArray());
		// this.target.add( pan );
	}

	module.LandscapeControls.prototype.zoom = function(x, y) {
		this.setRadius(this.radius * (1 + this.zoomSpeed * y));
	}

	module.LandscapeControls.prototype.setMix = function(mix) {
		this.mix = mix;
		if (mix > 0 && mix < 1 || true) { // hacky, I'm leaving this for now..
			this.azimuth = (1 - this.mix) * this._foldedazimuth + this.mix * this._flatazimuth;
			this.altitude = (1 - this.mix) * this._foldedaltitude + this.mix * this._flataltitude;
			this.target.set(this._foldedtarget.x * (1-mix) + mix*this._flattarget.x,
			                this._foldedtarget.y * (1-mix) + mix*this._flattarget.y,
			                this._foldedtarget.z * (1-mix) + mix*this._flattarget.z);
		} else {
			this.setAzimuth(this.azimuth);
			this.setAltitude(this.altitude);
			this.setTarget(this.target.toArray());
		}

		this.update2Dbutton();
	}

	module.LandscapeControls.prototype.setAzimuth = function(az) {
		if (az === undefined)
			return this.azimuth;

		az = az < 0 ? az + 360 : az % 360;

		if ( this.mix == 1.0 )
			this._flatazimuth = az;
		else
			this._foldedazimuth = az;
		this.azimuth = az;
		this.update2Dbutton();
	}
	module.LandscapeControls.prototype.setAltitude = function(alt) {
		if (alt === undefined)
			return this.altitude;

		if ( this.mix == 1.0 ) {
			this._flataltitude = Math.min(Math.max(alt, 0.1), 75);
			this.altitude = this._flataltitude
		}
		else {
			this._foldedaltitude = Math.min(Math.max(alt, 0.1), 179.9);
			this.altitude = this._foldedaltitude;
		}
		this.update2Dbutton();
	}

	module.LandscapeControls.prototype.setRadius = function(rad) {
		if (rad === undefined)
			return this.radius;

		this.radius = Math.max(Math.min(rad, 600), 10);
	}

	module.LandscapeControls.prototype.setTarget = function(xyz) {
		if (!(xyz instanceof Array))
			return [this.target.x, this.target.y, this.target.z];

		if (this.mix < 1) {
			this._foldedtarget.set(xyz[0], xyz[1], xyz[2]);
			this.target.set(xyz[0], xyz[1], xyz[2]);
		} else{
			this._flattarget.set(xyz[0], xyz[1], 0);
			this.target.set(xyz[0], xyz[1], 0);
		}
	}

	module.LandscapeControls.prototype.update2Dbutton = function() {
		if ( this.mix == 1.0 ){
			if ( this.altitude > 0.1  ||  this.azimuth != 180 )
				this.enable2Dbutton();
			else
				this.disable2Dbutton();
		}
		else
			this.disable2Dbutton();

	}
	module.LandscapeControls.prototype.enable2Dbutton = function() {
		// this.twodbutton.prop('disabled', false);
		this.twodbutton.show();
	}
	module.LandscapeControls.prototype.disable2Dbutton = function() {
		// this.twodbutton.prop('disabled', true);
		this.twodbutton.hide();
	}

	module.LandscapeControls.prototype.bind = function(object) {
		var _mousedowntime = 0;
		var _clicktime = 0; // Time of last click (mouseup event)
		var _indblpick = false; // In double-click and hold?
		var _picktimer = false; // timer that runs pick event
		//this._momentumtimer = false; // time that glide has been going on post mouse-release
		var _nomove_timer;

		var keystate = null;
		var changeEvent = { type: 'change' };

		function getMouse ( event ) {
			var off = $(event.target).offset();
			return new THREE.Vector2( event.clientX - off.left, event.clientY - off.top);
		};

		// listeners
		function keydown( event ) {
			if (event.keyCode == 17) { // ctrl
				keystate = STATE.ZOOM;
			} else if (event.keyCode == 16) { // shift
				keystate = STATE.PAN;
			} else {
				keystate = null;
			}

		};

		function keyup( event ) {
			keystate = null;
		};

		function blur ( event ) {
			keystate = null
		}

		function mousedown( event ) {
			event.preventDefault();
			event.stopPropagation();

			if ( this._state === STATE.NONE ) {
				this._state = keystate !== null ? keystate : event.button;
				this._start = this._end = getMouse(event);
				if (event.button == 0) {
					_mousedowntime = new Date().getTime();
				}

				// Run double-click event if time since last click is short enough
				if ( _mousedowntime - _clicktime < this.clickTimeout && event.button == 0 ) {
					if (_picktimer) clearTimeout(_picktimer);
					var mouse2D = getMouse(event).clone();
					this.dispatchEvent({ type:"dblpick", x:mouse2D.x, y:mouse2D.y, keep:keystate == STATE.ZOOM });
					_indblpick = true;
				} else {
					this.dispatchEvent({ type:"mousedown" });
				}
			}
		};

		function mouseup( event ) {
			this._momentumtimer = new Date().getTime();

			event.preventDefault();
			event.stopPropagation();

			this._state = STATE.NONE;
			if (event.button == 0) {
				_clicktime = new Date().getTime();
			}

			// Run picker if time since mousedown is short enough
			if ( _clicktime - _mousedowntime < this.clickTimeout && event.button == 0) {
				var mouse2D = getMouse(event).clone();
				this.dispatchEvent({ type: "mouseup" });
				this.dispatchEvent({ type:"pick", x:mouse2D.x, y:mouse2D.y, keep:keystate == STATE.ZOOM});
			} else if ( event.button == 0 && _indblpick == true ) {
				this.dispatchEvent({ type:"undblpick" });
				_indblpick = false;
			} else {
				this.dispatchEvent({ type: "mouseup" });
			}
			this.dispatchEvent(changeEvent);
		};

		function mousemove( event ) {
			if ( this._state !== STATE.NONE ) {
				this._end = getMouse(event);
				this.dispatchEvent( changeEvent );
			}
		};


		function mousewheel( event ) {
			if (!event.altKey) {
				delta = event.deltaY

				// normalize across browsers
				if(navigator.userAgent.toLowerCase().indexOf('firefox') > -1){
					delta = delta * 18
				}

			    this.setRadius(this.radius + this.zoomSpeed * delta * 110.0);
			    this.dispatchEvent( changeEvent );
			}
		};

		//code from http://vetruvet.blogspot.com/2010/12/converting-single-touch-events-to-mouse.html
		var touchToMouse=function(b){if(!(b.touches.length>1)){var a=b.changedTouches[0],c="";switch(b.type){case "touchstart":c="mousedown";break;case "touchmove":c="mousemove";break;case "touchend":c="mouseup";break;default:return}var d=document.createEvent("MouseEvent");d.initMouseEvent(c,true,true,window,1,a.screenX,a.screenY,a.clientX,a.clientY,false,false,false,false,0,null);a.target.dispatchEvent(d);b.preventDefault()}};
		object.addEventListener( 'touchstart', touchToMouse );
		object.addEventListener( 'touchmove', touchToMouse );
		object.addEventListener( 'touchend', touchToMouse );

		object.addEventListener( 'contextmenu', function ( event ) { event.preventDefault(); }, false );

		object.addEventListener( 'mousemove', mousemove.bind(this), false );
		object.addEventListener( 'mousedown', mousedown.bind(this), false );
		object.addEventListener( 'mouseup', mouseup.bind(this), false );
		object.addEventListener( 'wheel', mousewheel.bind(this), false);
		object.addEventListener( 'mouseout', mouseup.bind(this), false );

		window.addEventListener( 'keydown', keydown.bind(this), false );
		window.addEventListener( 'keyup', keyup.bind(this), false );
		window.addEventListener( 'blur', blur.bind(this), false );
	}

	return module;
}(jsplot || {}));

var jsplot = (function (module) {
    module.Menu = function(gui) {
        this._gui = gui;
        this._desc = {};
        this._folders = {};
        this._controls = {};
        this._hidden = false; //the initial state of this folder
        this._bubble_evt = function() {
            this.dispatchEvent({type:"update"});
        }.bind(this);
        this._wheelListeners = {};
    }
    THREE.EventDispatcher.prototype.apply(module.Menu.prototype);
    module.Menu.prototype.init = function(gui) {
        this._gui = gui;
        for (var name in this._desc) {
            if (this._folders[name] === undefined && this._desc[name] instanceof module.Menu) {
                var hidden = this._desc[name]._hidden;
                this._folders[name] = this._addFolder(gui, name, hidden);
                this[name].init(this._folders[name]);
            } else if (this._controls[name] === undefined) {
                this._controls[name] = this._add(gui, name, this._desc[name]);
            }
        }
    }
    module.Menu.prototype.set = function(name, value) {
        var namespace = name.split('.');
        var n = namespace.shift();
        if (namespace.length > 0) {
            return this[n].set(namespace.join('.'), value);
        } else if (this[n] === undefined) {
            var action = this._desc[n].action;
            if (action[0][action[1]] instanceof Function)
                action[0][action[1]].call(action[0], value)
            else
                action[0][action[1]] = value;
        } else {
            this[n] = value;
            var action = this._desc[n].action;
            action[0][action[1]](value);
        }
        if (this._controls[n])
            this._controls[n].updateDisplay();
        this.dispatchEvent({type:"update"});
    }
    module.Menu.prototype.get = function(name) {
        var namespace = name.split('.');
        var n = namespace.shift();
        if (namespace.length > 0) {
            return this[n].get(namespace.join('.'));
        } else if (this[n] === undefined) { //object reference
            var action = this._desc[n].action;
            if (action[0][action[1]] instanceof Function)
                return action[0][action[1]].call(action[0]);
            return action[0][action[1]];
        } else { // object function
            var action = this._desc[n].action;
            if (action[0][action[1]] instanceof Function)
                return action[0][action[1]]();
            return this[n];
        }
    }
    module.Menu.prototype.addFolder = function(name, hidden, menu) {
        if (menu === undefined)
            menu = new module.Menu();
        menu._hidden = hidden;
        this._desc[name] = menu;
        //cause update events to bubble
        menu.addEventListener("update", this._bubble_evt);

        if (this._gui !== undefined) {
            this._folders[name] = this._addFolder(this._gui, name, hidden);
            menu.init(this._folders[name]);
        }

        return menu;
    }
    module.Menu.prototype.add = function(desc) {
        for (var name in desc) {
            var args = desc[name];
            this._desc[name] = args;

            if (this._gui !== undefined) {
                this._controls[name] = this._add(this._gui, name, args);
            }
        }
        return this;
    }
    module.Menu.prototype.remove = function(name) {
        if (name === undefined) {
            for (var n in this._desc) {
                this._remove(n);
            }
        } else if (name instanceof module.Menu){
            for (var n in this._desc) {
                if (this._desc[n] === name)
                    this._remove(n);
            }
        } else {
            this._remove(name);
        }
    }

    module.Menu.prototype._addFolder = function(gui, name, hidden) {
        var folder = gui.addFolder(name);
        if (!hidden)
            folder.open();
        this[name] = this._desc[name];
        return folder;
    }
    module.Menu.prototype._add = function(gui, name, desc) {
        if (desc.action instanceof Function) {
            //A button that runs a function (IE Reset)
            this[name] = desc.action;
            if (!desc.hidden)
                gui.add(desc, "action").name(name);
        } else if ( desc.action instanceof Array) {
            var obj = desc.action[0][desc.action[1]];
            if (obj instanceof Function) {
                var parent = desc.action[0];
                var method = desc.action[1];
                this[name] = parent[method]();

                if (!desc.hidden) {
                    //copy the args to avoid changing the desc
                    var newargs = [this, name];
                    for (var i = 2; i < desc.action.length; i++)
                        newargs.push(desc.action[i]);

                    var ctrl = gui.add.apply(gui, newargs);
                    ctrl.onChange(function(name) {
                        parent[method](this[name]);
                        this.dispatchEvent({type:"update"});
                    }.bind(this, name));

                    //replace the function so that calling it will update the control
                    ctrl._oldfunc = parent[method]; //store original so we can replace
                    var func = parent[method].bind(parent);
                    parent[method] = function(name, val) {
                        if (val === undefined)
                            return func();

                        func(val);
                        this[name] = val;
                        ctrl.updateDisplay();
                    }.bind(this, name)
                };
            } else if (!desc.hidden) {
                var ctrl = gui.add.apply(gui, desc.action).name(name);
                ctrl.onChange(function() {
                    this.dispatchEvent({type:"update"});
                }.bind(this));
            }
        }
        //setup keyboard shortcuts for commands
        // if (desc.key) {

        //     var key = desc.key;
        //     var action = desc.action;
        //     window.addEventListener("keypress", function(event) {

        //         if (window.colorlegendOpen) {
        //             return;
        //         }

        //         if (event.target.nodeName === 'INPUT' && (event.target.id !== "" || (event.target.hasOwnProperty('classList') && event.target.classList.index("select2-search__field") ==! -1))) {
        //             // note: normally you would want to block on all INPUT target tags. however, if you tab-key away from an input element, INPUT remains the target even if the element has been manually deblurred, but the id *will* be cleared. since it would be nice to be able to use shortcuts after tab-aways, this statement only blocks events from inputs with ids
        //             return;
        //         }
        //         if (event.defaultPrevented)
        //             return;

        //         if (desc.hasOwnProperty('modKeys')) {
        //             for (let modKey of desc.modKeys) {
        //                 if (!event[modKey]) {
        //                     return
        //                 }
        //             }
        //         }

        //         if (event.key == key) {
        //             action();
        //             event.preventDefault();
        //             event.stopPropagation();
        //         }
        //     }.bind(this), true);
        // }

        // setup mousewheel shortcuts
        if (desc.wheel && !(desc.help in this._wheelListeners)) {
            this._wheelListeners[desc.help] = window.addEventListener("wheel", function(event) {
                if (desc.hasOwnProperty('modKeys')) {
                    for (let modKey of desc.modKeys) {
                        if (!event[modKey]) {
                            return
                        }
                    }
                }

                event.preventDefault();
                event.stopPropagation();

                desc.action(event.deltaY);

                for (var i in gui.__controllers) {
                    gui.__controllers[i].updateDisplay();
                }
            }.bind(this), true);
        }

        return ctrl;
    }
    module.Menu.prototype._remove = function(name) {
        if (this._desc[name] instanceof module.Menu) {
            this[name].remove();
            this[name].removeEventListener(this._bubble_evt);
            delete this._folders[name];
            if (this._gui !== undefined)
                this._gui.removeFolder(name);
        } else if (this._controls[name] !== undefined) {
            if (this._controls[name]._oldfunc !== undefined) {
                var args = this._desc[name].action;
                var parent = args[0], func = args[1];
                parent[func] = this._controls[name]._oldfunc;
            }
            if (this._gui !== undefined)
                this._gui.remove(this._controls[name]);
            delete this._controls[name];
        }
    }
    return module;
}(jsplot || {}));

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
