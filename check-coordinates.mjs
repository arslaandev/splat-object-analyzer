import assert from 'node:assert/strict';
import {viewerPosition, viewerRotation} from './viewer/src/SceneCoordinates.js';

function rotate(q, p) {
  const [x,y,z] = p, {x:a,y:b,z:c,w:d} = q;
  const tx=2*(b*z-c*y), ty=2*(c*x-a*z), tz=2*(a*y-b*x);
  return [x+d*tx+b*tz-c*ty, y+d*ty+c*tx-a*tz, z+d*tz+a*ty-b*tx];
}
// Every corner must land at the same position as the source scene after its
// global X rotation, including a nonzero object orientation and offset centre.
const q={x:0,y:Math.sin(.43),z:0,w:Math.cos(.43)};
const center={x:4,y:-2,z:7};
for (const x of [-2,2]) for (const y of [-.5,.5]) for (const z of [-1,1]) {
  const source=rotate(q,[x,y,z]).map((v,i)=>v+Object.values(center)[i]);
  const expected=viewerPosition({x:source[0],y:source[1],z:source[2]});
  const vc=viewerPosition(center);
  const actual=rotate(viewerRotation(q),[x,y,z]).map((v,i)=>v+Object.values(vc)[i]);
  actual.forEach((v,i)=>assert.ok(Math.abs(v-Object.values(expected)[i])<1e-9));
}
console.log('Source/viewer box corner transforms passed.');
