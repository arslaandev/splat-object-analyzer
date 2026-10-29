// The same source-space grid and conservative supercover policy as navigation.py.
// Kept independent of THREE so coordinate and collision checks can run in Node.
export class NavigationGrid {
  constructor(data) {
    this.metadata = data;
    this.shape = data.shape;
    this.origin = data.origin;
    this.voxel = data.voxel_size;
    if (data.space !== 'source' || data.encoding !== 'rle_xyz_c' ||
        !Array.isArray(this.shape) || this.shape.length !== 3 || !this.shape.every(n => Number.isInteger(n) && n > 0) ||
        !Array.isArray(this.origin) || this.origin.length !== 3 || !this.origin.every(Number.isFinite) ||
        !Number.isFinite(this.voxel) || this.voxel <= 0) throw new Error('Invalid navigation grid metadata.');
    const size = this.shape.reduce((a,b) => a*b, 1);
    if (size > 8_000_000 || !Array.isArray(data.runs) || data.runs.length % 2) throw new Error('Invalid navigation grid size.');
    this.cells = new Uint8Array(size);
    let offset = 0;
    for (let i = 0; i < data.runs.length; i += 2) {
      const value = data.runs[i], count = data.runs[i+1];
      if (!Number.isInteger(value) || value < 0 || value > 3 || !Number.isInteger(count) || count <= 0 || offset+count > size) throw new Error('Invalid navigation grid encoding.');
      this.cells.fill(value, offset, offset+count);
      offset += count;
    }
    if (offset !== size) throw new Error('Incomplete navigation grid.');
  }

  state(index) {
    if (index.some((n,i) => n < 0 || n >= this.shape[i])) return 0;
    return this.cells[(index[0]*this.shape[1]+index[1])*this.shape[2]+index[2]];
  }

  cell(point) {
    if (!Array.isArray(point) || point.length !== 3 || !point.every(Number.isFinite)) return null;
    const index = point.map((n,i) => Math.floor((n-this.origin[i])/this.voxel));
    return index.some((n,i) => n < 0 || n >= this.shape[i]) ? null : index;
  }

  isPositionValid(point) {
    const index = this.cell(point);
    return index !== null && this.state(index) === 2;
  }

  isSegmentValid(start, end) {
    if (!this.isPositionValid(start) || !this.isPositionValid(end)) return false;
    const a = start.map((n,i) => (n-this.origin[i])/this.voxel);
    const b = end.map((n,i) => (n-this.origin[i])/this.voxel);
    const cell = a.map(Math.floor), delta = b.map((n,i) => n-a[i]);
    const step = delta.map(Math.sign);
    const increment = delta.map(n => Math.abs(n) > 1e-14 ? 1/Math.abs(n) : Infinity);
    const crossing = delta.map((n,i) => Math.abs(n) > 1e-14 ? (cell[i]+Number(step[i] > 0)-a[i])/n : Infinity);
    const parallel = [0,1,2].filter(i => Math.abs(delta[i]) <= 1e-14 && Math.abs(a[i]-Math.round(a[i])) < 1e-10);
    const variations = (index, axes, shifts, callback) => {
      for (let mask = 0; mask < (1 << axes.length); mask++) {
        const touched = [...index];
        axes.forEach((axis,j) => { if (mask & (1 << j)) touched[axis] += shifts[axis]; });
        if (!callback(touched)) return false;
      }
      return true;
    };
    const allowed = index => variations(index, parallel, [-1,-1,-1], touched => this.state(touched) === 2);
    const initial = [0,1,2].filter(i => Math.abs(a[i]-Math.round(a[i])) < 1e-10);
    if (!variations(cell, initial, [-1,-1,-1], allowed)) return false;
    for (let n = 0; n < this.shape.reduce((x,y) => x+y, 0)+4; n++) {
      const t = Math.min(...crossing);
      if (t > 1+1e-10) return true;
      const axes = [0,1,2].filter(i => Math.abs(crossing[i]-t) < 1e-10);
      if (!variations(cell, axes, step, allowed)) return false;
      axes.forEach(axis => { cell[axis] += step[axis]; crossing[axis] += increment[axis]; });
    }
    return false;
  }
}
