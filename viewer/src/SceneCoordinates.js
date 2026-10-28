// The same source-to-viewer rotation is used for the splat and its annotations.
export const SOURCE_ROTATION_X = Math.PI;

export function viewerPosition({ x, y, z }) { return { x, y: -y, z: -z }; }

export function viewerRotation({ x = 0, y = 0, z = 0, w = 1 } = {}) {
  // Quaternion composition: rotate the source frame by pi around viewer X.
  return { x: w, y: -z, z: y, w: -x };
}
