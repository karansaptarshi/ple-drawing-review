'use strict';
const fs = require('node:fs');
const {IfcAPI, LogLevel} = require('web-ifc');

function bounds() {
  return {low: [Infinity, Infinity, Infinity], high: [-Infinity, -Infinity, -Infinity]};
}

function include(box, xyz) {
  for (let axis = 0; axis < 3; axis++) {
    box.low[axis] = Math.min(box.low[axis], xyz[axis]);
    box.high[axis] = Math.max(box.high[axis], xyz[axis]);
  }
}

function dimensions(box) {
  return {...box, size: box.high.map((maximum, axis) => maximum - box.low[axis])};
}

function projectPoint(t, x, y, z) {
  // web-ifc maps IFC's +Z up to rendering +Y up. Convert the placed vertices
  // back to native project axes, leaving length scaling in the mesh transform.
  const a = t[0]*x + t[4]*y + t[8]*z + t[12];
  const b = t[1]*x + t[5]*y + t[9]*z + t[13];
  const c = t[2]*x + t[6]*y + t[10]*z + t[14];
  return [a, -c, b];
}

async function summarize(file, classFilter = "") {
  const api = new IfcAPI();
  await api.Init();
  api.SetLogLevel(LogLevel.LOG_LEVEL_OFF);
  const model = api.OpenModel(fs.readFileSync(file), {COORDINATE_TO_ORIGIN:false});
  if (model < 0) throw Error('Could not open the IFC model');
  try {
    const world = bounds();
    const elements = [];
    const counts = {};
    let matches = 0;
    api.StreamAllMeshes(model, mesh => {
      const box = bounds();
      let n = 0;
      for (let i=0; i<mesh.geometries.size(); i++) {
        const placed = mesh.geometries.get(i);
        if (placed.geomType !== 0) continue;
        const geom = api.GetGeometry(model, placed.geometryExpressID);
        try {
          const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
          for (let j=0; j<verts.length; j+=6) {
            const point = projectPoint(placed.flatTransformation, verts[j], verts[j+1], verts[j+2]);
            if (!point.every(Number.isFinite)) throw Error('Nonfinite vertex');
            include(box, point);
            n++;
          }
        } finally { geom.delete(); }
      }
      if (!n) return;
      const product = api.GetLine(model, mesh.expressID);
      include(world,box.low);
      include(world,box.high);
      const type = api.GetNameFromTypeCode(product.type);
      counts[type] = (counts[type] || 0) + 1;
      if (classFilter && type.toLowerCase() !== classFilter.toLowerCase()) return;
      matches++;
      if (elements.length < 40) {
        elements.push({id:mesh.expressID, type, name:(product.Name?.value ?? '').slice(0,120), ...dimensions(box)});
      }
    },true);
    return {schema:api.GetModelSchema(model), units:'m', axes:'IFC project XYZ',
            classes:counts, matches, world:Object.keys(counts).length ? dimensions(world) : null, elements};
  } finally { api.CloseModel(model); }
}

if (require.main === module) {
  summarize(0, process.argv[2]).then(info=>process.stdout.write(JSON.stringify(info)+'\n'))
    .catch(err=>{process.stderr.write(err.message+'\n'); process.exitCode=1;});
}
module.exports = {summarize};
