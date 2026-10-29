import assert from 'node:assert/strict';
import { NavigationGrid } from './viewer/src/NavigationGrid.js';
import { viewerPosition } from './viewer/src/SceneCoordinates.js';

function encoded(shape, cells, origin = [0,0,0], voxel = 1) {
  const runs = [];
  for (const cell of cells) {
    if (runs.length && runs.at(-2) === cell) runs[runs.length-1]++;
    else runs.push(cell,1);
  }
  return {space:'source', encoding:'rle_xyz_c', shape, origin, voxel_size:voxel, runs};
}
const shape = [40,42,30], cells = new Uint8Array(shape.reduce((a,b)=>a*b));
const at = (x,y,z) => (x*shape[1]+y)*shape[2]+z;
for (let x=5; x<33; x++) for(let y=5; y<37; y++) for(let z=5; z<25; z++) {
  if (y<15 || x>=23) cells[at(x,y,z)] = 2;
}
const map = new NavigationGrid(encoded(shape,cells));
const start = [9.5,9.5,12.5], goal = [27.5,32.5,12.5];
assert(map.isPositionValid(start) && map.isPositionValid(goal));
assert(!map.isSegmentValid(start,goal));
assert(map.isSegmentValid(start,[27.5,9.5,12.5]));
assert(map.isSegmentValid([27.5,9.5,12.5],goal));
assert(map.isSegmentValid(start,start));
assert(!map.isPositionValid([NaN,0,0]));
assert(!map.isPositionValid([-1,0,0]));
assert(!map.isSegmentValid(start,[100,9.5,12.5]));
const transformed = new NavigationGrid(encoded(shape,cells,[-17,31,7],.3));
const world = start.map((n,i)=>n*.3+[-17,31,7][i]);
assert(transformed.isPositionValid(world));
const viewer = viewerPosition({x:world[0],y:world[1],z:world[2]});
assert(transformed.isPositionValid([viewer.x,-viewer.y,-viewer.z]));
const toyCells = new Uint8Array(125).fill(2); toyCells[(2*5+1)*5+2] = 1;
const toy = new NavigationGrid(encoded([5,5,5],toyCells));
assert(!toy.isSegmentValid([1.5,1.5,2.5],[2.5,2.5,2.5]));
assert(!toy.isSegmentValid([1.5,2,2.5],[3.5,2,2.5]));
assert.throws(()=>new NavigationGrid({...encoded([5,5,5],toyCells),runs:[2,124]}));
assert.throws(()=>new NavigationGrid({...encoded([5,5,5],toyCells),runs:[5,125]}));
assert.throws(()=>new NavigationGrid({...encoded([5,5,5],toyCells),runs:[2,126]}));
console.log('Viewer navigation: RLE decoding, L-shaped segments, source/viewer coordinates, corner and plane collisions, and invalid inputs passed.');
