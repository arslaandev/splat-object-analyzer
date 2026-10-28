import * as THREE from "three";
import { Annotation } from "./Annotation.js";
import { viewerPosition, viewerRotation } from './SceneCoordinates.js';

const PALETTE = [
  "#2f81f7", "#3fb950", "#db6d28", "#a371f7",
  "#f778ba", "#e3b341", "#56d4dd", "#f85149",
];

// The splat is rotated π about X in SplatView; annotations must follow the same
// mapping comes from SceneCoordinates, shared with SplatView.
function toViewerSpace(obj) {
  const box = obj.oriented_box || obj;
  const p = viewerPosition(box.position);
  const position = new THREE.Vector3(p.x, p.y, p.z);
  const s = box.scale || box.size || { x: 0.5, y: 0.5, z: 0.5 };
  const size = new THREE.Vector3(s.x, s.y, s.z);
  const r = viewerRotation(box.rotation);
  const quaternion = new THREE.Quaternion(r.x, r.y, r.z, r.w).normalize();
  return { position, size, quaternion };
}

export class AnnotationParser {
  static async parse(source) {
    let text;
    if (typeof source === "string") {
      const res = await fetch(source);
      if (!res.ok) {
        throw new Error(`Failed to fetch annotations: ${res.status} ${res.statusText}`);
      }
      text = await res.text();
    } else {
      text = await source.text();
    }
    return AnnotationParser.fromText(text);
  }

  static fromText(text) {
    const data = JSON.parse(text);
    const objects = Array.isArray(data) ? data : data.objects || [];

    const colorByLabel = new Map();
    let colorIdx = 0;

    return objects
      .filter((o) => o && o.label && o.position)
      .map((o) => {
        if (!colorByLabel.has(o.label)) {
          colorByLabel.set(o.label, PALETTE[colorIdx++ % PALETTE.length]);
        }
        return new Annotation({
          label: o.label,
          parentId: o.parent_id || null,
          color: colorByLabel.get(o.label),
          ...toViewerSpace(o),
        });
      });
  }
}
