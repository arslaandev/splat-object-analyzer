import * as THREE from 'three';
import { SOURCE_ROTATION_X } from './SceneCoordinates.js';
import { NavigationGrid } from './NavigationGrid.js';

export class NavigationLayer {
  constructor(manager, container) {
    this.manager = manager;
    this.root = container;
    this.generation = 0;
    this.group = new THREE.Group();
    this.group.rotation.x = SOURCE_ROTATION_X;
    manager.add(this.group);
    container.innerHTML = `
      <h2>Camera space</h2>
      <p data-navigation-status role="status" aria-live="polite">No navigation map loaded.</p>
      <div data-navigation-controls hidden>
        <label class="field navigation-check"><input type="checkbox" data-overlay> Show grid slice</label>
        <div class="field"><label for="navigation-state">Display</label><select id="navigation-state" data-state>
          <option value="all">All states</option><option value="free">Candidate space</option><option value="occupied">Obstacles</option><option value="unknown">Unknown</option>
        </select></div>
        <div class="field"><label for="navigation-axis">Slice axis (source coordinates)</label><select id="navigation-axis" data-axis>
          <option value="0">X</option><option value="1" selected>Y</option><option value="2">Z</option>
        </select></div>
        <div class="field"><label for="navigation-slice" data-slice-label>Slice</label><input id="navigation-slice" data-slice type="range" min="0" step="1"></div>
        <canvas data-map-preview role="img" aria-label="Navigation grid slice in source coordinates"></canvas>
        <div class="navigation-note" data-map-axes></div>
        <div class="navigation-legend"><span class="nav-free">Candidate</span> · <span class="nav-occupied">Obstacle</span> · <span class="nav-unknown">Unknown</span> · <span class="nav-clearance">Clearance margin</span></div>
        <div class="field"><label for="navigation-region">Connected region</label><select id="navigation-region" data-region></select></div>
        <button type="button" data-enter>Enter candidate space</button>
        <label class="field navigation-check"><input type="checkbox" data-lock> Constrain camera to candidate space</label>
        <p class="navigation-note">Geometry estimate only. Missing boundaries remain unknown. Unknown and clearance margins block constrained movement.</p>
      </div>`;
    this.status = container.querySelector('[data-navigation-status]');
    this.controls = container.querySelector('[data-navigation-controls]');
    this.overlay = container.querySelector('[data-overlay]');
    this.state = container.querySelector('[data-state]');
    this.axis = container.querySelector('[data-axis]');
    this.slice = container.querySelector('[data-slice]');
    this.region = container.querySelector('[data-region]');
    this.preview = container.querySelector('[data-map-preview]');
    this.enter = container.querySelector('[data-enter]');
    this.lock = container.querySelector('[data-lock]');
    this.overlay.addEventListener('change', () => this.draw());
    this.state.addEventListener('change', () => this.draw());
    this.slice.addEventListener('input', () => this.draw());
    this.axis.addEventListener('change', () => { this.setSlice(); this.draw(); });
    this.region.addEventListener('change', () => { this.setSlice(); this.draw(); });
    this.enter.addEventListener('click', () => this.enterRegion());
    this.lock.addEventListener('change', () => {
      if (this.lock.checked && !this.enterRegion()) this.lock.checked = false;
      manager.setNavigationConstraint(this.lock.checked ? this.map : null);
    });
  }

  clear() {
    this.generation++;
    this.abort?.abort();
    this.map = null;
    this.lock.checked = false;
    this.overlay.checked = false;
    this.manager.setNavigationConstraint(null);
    this.controls.hidden = true;
    this.status.textContent = 'No navigation map for this analysis. Reanalyze it or generate a map with navigation.py.';
    this.disposeSlice();
  }

  async load(url, sourceHash) {
    this.clear();
    if (!url) return;
    const generation = this.generation;
    const abort = new AbortController();
    this.abort = abort;
    this.status.textContent = 'Loading camera-space estimate…';
    try {
      const response = await fetch(url, {signal:abort.signal});
      if (!response.ok) throw new Error('Could not load the navigation map.');
      const data = await response.json();
      if (generation !== this.generation) return;
      if (!sourceHash || data.source_sha256 !== sourceHash) throw new Error('Navigation belongs to a different scene.');
      this.map = new NavigationGrid(data);
      const regions = data.regions;
      this.region.replaceChildren(...regions.map((region, i) => new Option(`Region ${i+1} · ${region.cell_count.toLocaleString()} cells`, String(region.id))));
      this.region.disabled = this.enter.disabled = this.lock.disabled = regions.length === 0;
      this.status.textContent = regions.length ? `${regions.length} candidate region${regions.length === 1 ? '' : 's'} · clearance radius ${data.camera_radius.toPrecision(3)} source units` : 'Interior uncertain: no enclosed region passed the geometry and clearance checks.';
      this.controls.hidden = false;
      this.setSlice();
      this.draw();
    } catch (error) {
      if (generation !== this.generation || error.name === 'AbortError') return;
      this.status.textContent = error.message;
    }
  }

  setSlice() {
    if (!this.map) return;
    const axis = Number(this.axis.value);
    this.slice.max = this.map.shape[axis]-1;
    const seed = this.map.metadata.regions.find(r => String(r.id) === this.region.value)?.seed;
    // The viewer uses Y-up after its source transform. Start the horizontal
    // slice below the camera, avoiding a plane coincident with the near clip.
    this.slice.value = seed ? Math.min(this.map.shape[axis]-1, this.map.cell(seed)[axis]+(axis === 1 ? 3 : 0)) : Math.floor(this.map.shape[axis]/2);
  }

  enterRegion() {
    const region = this.map?.metadata.regions.find(r => String(r.id) === this.region.value);
    if (!region || !this.map.isPositionValid(region.seed)) return false;
    const position = new THREE.Vector3(region.seed[0], -region.seed[1], -region.seed[2]);
    const direction = this.manager.camera.getWorldDirection(new THREE.Vector3());
    this.manager.camera.position.copy(position);
    this.manager.controls.target.copy(position).add(direction);
    this.manager.camera.near = Math.min(this.map.metadata.camera_radius/4, .01);
    this.manager.camera.updateProjectionMatrix();
    this.manager.controls.update();
    this.manager.navigationTouched = true;
    this.manager.renderer.domElement.focus({preventScroll:true});
    this.manager.navigationCheckpoint();
    return true;
  }

  disposeSlice() {
    if (!this.mesh) return;
    this.group.remove(this.mesh);
    this.mesh.geometry.dispose();
    this.mesh.material.dispose();
    this.mesh.dispose();
    this.mesh = null;
  }

  draw() {
    this.disposeSlice();
    if (!this.map) return;
    const axis = Number(this.axis.value), slice = Number(this.slice.value);
    const coordinate = this.map.origin[axis]+(slice+.5)*this.map.voxel;
    this.root.querySelector('[data-slice-label]').textContent = `Slice ${slice+1}/${this.map.shape[axis]} · ${coordinate.toFixed(3)} source units`;
    const axes = [0,1,2].filter(a => a !== axis);
    this.drawPreview(axes, axis, slice);
    if (!this.overlay.checked) return;
    const selected = [];
    for (let i = 0; i < this.map.shape[axes[0]]; i++) for (let j = 0; j < this.map.shape[axes[1]]; j++) {
      const cell = [0,0,0]; cell[axis] = slice; cell[axes[0]] = i; cell[axes[1]] = j;
      const state = this.map.state(cell);
      if (this.state.value === 'free' && state !== 2 || this.state.value === 'occupied' && state !== 1 || this.state.value === 'unknown' && state !== 0) continue;
      selected.push([cell,state]);
    }
    if (!selected.length) return;
    const size = [1,1,1].map((_,i) => this.map.voxel*(i === axis ? .04 : .94));
    const geometry = new THREE.BoxGeometry(...size);
    const material = new THREE.MeshBasicMaterial({transparent:true, opacity:.45, depthWrite:false, depthTest:false});
    const mesh = new THREE.InstancedMesh(geometry, material, selected.length);
    mesh.renderOrder = 20;
    const transform = new THREE.Matrix4();
    const colors = [0x78899b, 0xe98282, 0x52d6a5, 0xe2be72].map(c => new THREE.Color(c));
    selected.forEach(([cell,state],i) => {
      transform.makeTranslation(...cell.map((n,a) => this.map.origin[a]+(n+.5)*this.map.voxel));
      mesh.setMatrixAt(i, transform);
      mesh.setColorAt(i, colors[state]);
    });
    mesh.computeBoundingSphere();
    this.mesh = mesh;
    this.group.add(mesh);
  }

  drawPreview(axes, axis, slice) {
    const width = this.map.shape[axes[0]], height = this.map.shape[axes[1]];
    this.preview.width = width*2;
    this.preview.height = height*2;
    const ctx = this.preview.getContext('2d');
    const colors = ['#78899b', '#e98282', '#52d6a5', '#e2be72'];
    ctx.fillStyle = getComputedStyle(this.root).getPropertyValue('--surface');
    ctx.fillRect(0,0,width*2,height*2);
    for (let x=0;x<width;x++) for (let y=0;y<height;y++) {
      const index = [0,0,0]; index[axis]=slice; index[axes[0]]=x; index[axes[1]]=y;
      const state = this.map.state(index);
      if (this.state.value === 'free' && state !== 2 || this.state.value === 'occupied' && state !== 1 || this.state.value === 'unknown' && state !== 0) continue;
      ctx.globalAlpha = state === 0 ? .2 : .75;
      ctx.fillStyle = colors[state];
      ctx.fillRect(x*2,y*2,2,2);
    }
    ctx.globalAlpha = 1;
    const seed = this.map.metadata.regions.find(r => String(r.id) === this.region.value)?.seed;
    const cell = seed && this.map.cell(seed);
    if (cell) {
      ctx.strokeStyle = getComputedStyle(this.root).getPropertyValue('--text');
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc((cell[axes[0]]+.5)*2, (cell[axes[1]]+.5)*2, 5, 0, Math.PI*2);
      ctx.stroke();
    }
    const names = ['X','Y','Z'];
    this.root.querySelector('[data-map-axes]').textContent = `Source ${names[axes[0]]} → · ${names[axes[1]]} ↓ · Ring: projected starting point`;
  }
}
