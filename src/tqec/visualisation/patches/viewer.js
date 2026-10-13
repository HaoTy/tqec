// All state lives in the viewer's document, which is isolated in a notebook iframe.
async function startPatchViewer() {
  const THREE = await import("three");
  const {OrbitControls} = await import("three-orbitcontrols");
  const data = JSON.parse(document.getElementById("patch-scene").textContent);
  const get = (id) => document.getElementById(id);
  const container = get("viewport");
  const status = get("status");
  const tooltip = get("tooltip");
  const colors = {X: 0xff7f7f, Y: 0x63c676, Z: 0x7396ff};
  const operatorColors = {X: 0xd92335, Y: 0x148b48, Z: 0x174dde};
  const cycles = data.cycles || [];
  const observables = data.observables || [];
  const times = [...new Set(cycles.map((cycle) => cycle.time))].sort((a, b) => a - b);
  const cycleById = new Map(cycles.map((cycle) => [cycle.id, cycle]));
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f7fa);
  const world = new THREE.Group();
  scene.add(world);
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  container.appendChild(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100000);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.screenSpacePanning = true;
  scene.add(new THREE.AmbientLight(0xffffff, 0.85));
  const light = new THREE.DirectionalLight(0xffffff, 0.6);
  light.position.set(10, -15, 25);
  scene.add(light);

  const geometries = new Set();
  const materials = new Set();
  const opacityMaterials = [];
  const fixedThicknessObjects = [];
  const pickable = [];
  const records = [];
  const materialCache = new Map();
  const geometry = (value) => { geometries.add(value); return value; };
  const sphereGeometry = geometry(new THREE.SphereGeometry(0.115, 10, 7));
  const ancillaGeometry = geometry(new THREE.OctahedronGeometry(0.13));
  const boundaryGeometry = geometry(new THREE.OctahedronGeometry(0.16));
  const operatorGeometry = geometry(new THREE.SphereGeometry(0.185, 12, 8));
  const unitBoxGeometry = geometry(new THREE.BoxGeometry(1, 1, 1));
  const ringGeometry = geometry(new THREE.TorusGeometry(0.26, 0.035, 5, 16));

  function material(color, opacity = 1, scaleOpacity = false, line = false, dashed = false) {
    const key = [color, opacity, scaleOpacity, line, dashed].join(":");
    if (materialCache.has(key)) return materialCache.get(key);
    const options = {color: new THREE.Color(color).convertSRGBToLinear(),
      transparent: opacity < 1 || scaleOpacity, opacity,
      depthWrite: opacity === 1 && !scaleOpacity, side: THREE.DoubleSide};
    let value;
    if (dashed) value = new THREE.LineDashedMaterial({...options, dashSize: 0.15, gapSize: 0.1});
    else if (line) value = new THREE.LineBasicMaterial(options);
    else value = new THREE.MeshBasicMaterial(options);
    materials.add(value);
    materialCache.set(key, value);
    if (scaleOpacity) opacityMaterials.push({material: value, coefficient: opacity});
    return value;
  }

  function annotate(object, description) {
    object.userData.description = description;
    if ([sphereGeometry, ancillaGeometry, boundaryGeometry, operatorGeometry, ringGeometry]
      .includes(object.geometry)) {
      fixedThicknessObjects.push({object, scaleZ: object.scale.z});
    }
    pickable.push(object);
    return object;
  }

  function group(category, timeMin, timeMax = timeMin, observable = null) {
    const value = new THREE.Group();
    world.add(value);
    records.push({object: value, category, timeMin, timeMax, observable});
    return value;
  }

  function line(points, color, closed = false, dashed = false, opacity = 1) {
    const vertices = points.map((point) => new THREE.Vector3(...point));
    if (closed && vertices.length) vertices.push(vertices[0].clone());
    const value = new THREE.Line(
      geometry(new THREE.BufferGeometry().setFromPoints(vertices)),
      material(color, opacity, false, true, dashed),
    );
    if (dashed) value.computeLineDistances();
    return value;
  }

  function supportText(support) {
    return (support || []).map(([x, y, basis]) => `${basis}(${x}, ${y})`).join(" ") || "identity";
  }

  function signedSupportText(support, sign = "+") {
    const phase = sign === -1 ? "−" : sign === 1 ? "+" : sign;
    return `${phase}${supportText(support)}`;
  }

  function measurementText(measurements, prefix = "Segment-local measurement parity") {
    return measurements?.length ? `${prefix}: ${measurements.join(" ⊕ ")}` : "";
  }

  function cycleText(cycle) {
    return cycle.label || `Block time ${cycle.block_z}, local round ${cycle.round}`;
  }

  function checkText(check, cycle, change = "") {
    return [change ? `${change} stabilizer` : "Measured stabilizer", cycleText(cycle),
      signedSupportText(check.support, check.sign),
      check.output_support ? `Output: ${signedSupportText(check.output_support, check.output_sign)}` : "",
      measurementText(check.measurements, "Round-local measurement parity"),
      check.id != null ? `Check: ${check.id}` : ""]
      .filter(Boolean).join("\n");
  }

  // Monotone-chain hulls avoid crossing polygon edges for arbitrary support order.
  function hull(points) {
    const ordered = [...new Map(points.map(([x, y]) => [`${x},${y}`, [x, y]])).values()]
      .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (ordered.length < 3) return ordered;
    const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lower = [];
    for (const point of ordered) {
      while (lower.length > 1 && cross(lower.at(-2), lower.at(-1), point) <= 0) lower.pop();
      lower.push(point);
    }
    const upper = [];
    for (const point of [...ordered].reverse()) {
      while (upper.length > 1 && cross(upper.at(-2), upper.at(-1), point) <= 0) upper.pop();
      upper.push(point);
    }
    return lower.slice(0, -1).concat(upper.slice(0, -1));
  }

  function polygonPoints(support) {
    let points = hull(support);
    if (points.length === 1) {
      const [x, y] = points[0];
      return Array.from({length: 16}, (_, i) => [x + 0.3 * Math.cos(i * Math.PI / 8),
        y + 0.3 * Math.sin(i * Math.PI / 8)]);
    }
    if (points.length === 2) {
      const [[x1, y1], [x2, y2]] = points;
      const length = Math.hypot(x2 - x1, y2 - y1);
      const dx = -(y2 - y1) / length * 0.24;
      const dy = (x2 - x1) / length * 0.24;
      return [[x1 + dx, y1 + dy], [x2 + dx, y2 + dy],
        [x2 - dx, y2 - dy], [x1 - dx, y1 - dy]];
    }
    if (points.length > 2) {
      const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
      const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
      points = points.map(([x, y]) => [cx + 0.9 * (x - cx), cy + 0.9 * (y - cy)]);
    }
    return points;
  }

  function polygon(points, time, faceMaterial) {
    const shape = new THREE.Shape();
    points.forEach(([x, y], i) => i ? shape.lineTo(x, y) : shape.moveTo(x, y));
    shape.closePath();
    const value = new THREE.Mesh(geometry(new THREE.ShapeGeometry(shape)), faceMaterial);
    value.position.z = time;
    return value;
  }

  function addCheck(parent, check, cycle, change = "") {
    const support = check.support || [];
    const points = polygonPoints(support);
    if (!points.length) return;
    const bases = new Set(support.map((entry) => entry[2]));
    const color = bases.size === 1 ? colors[support[0][2]] || 0x8b91a0 : 0xa28bbb;
    const time = cycle.time + (change ? 0.06 : 0.015);
    const info = checkText(check, cycle, change);
    const mesh = polygon(points, time, material(color, change ? 1.35 : 1, true));
    annotate(mesh, info);
    parent.add(mesh);
    const edgeColor = change === "Removed" ? 0xbb3e31 : change === "Added" ? 0x12814b : color;
    parent.add(annotate(line(points.map(([x, y]) => [x, y, time]), edgeColor, true,
      change === "Removed", change ? 1 : 0.5), info));
    if (bases.size > 1) {
      for (const [x, y, basis] of support) {
        const marker = new THREE.Mesh(sphereGeometry, material(colors[basis] || color));
        marker.position.set(x, y, time + 0.04);
        marker.scale.setScalar(0.8);
        parent.add(annotate(marker, info));
      }
    }
  }

  // Connect support sites using a minimum spanning tree, independent of their input order.
  function supportEdges(support, time) {
    if (support.length < 2) return [];
    const linked = new Set([0]);
    const remaining = new Set(support.slice(1).map((_, index) => index + 1));
    const edges = [];
    while (remaining.size) {
      let best = null;
      for (const a of linked) for (const b of remaining) {
        const distance = (support[a][0] - support[b][0]) ** 2 + (support[a][1] - support[b][1]) ** 2;
        if (best === null || distance < best.distance) best = {a, b, distance};
      }
      // Distant components of a logical operator need not be joined physically.
      if (best.distance <= 8.01) {
        edges.push([[support[best.a][0], support[best.a][1], time],
          [support[best.b][0], support[best.b][1], time]]);
      }
      remaining.delete(best.b);
      linked.add(best.b);
    }
    return edges;
  }

  function addSupport(parent, support, time, description, factor = false) {
    for (const [x, y, basis] of support) {
      const marker = new THREE.Mesh(factor ? ringGeometry : operatorGeometry,
        material(factor ? 0xdea521 : operatorColors[basis] || 0x75509c));
      marker.position.set(x, y, time);
      parent.add(annotate(marker, `${description}\n${basis}(${x}, ${y})`));
    }
    const basis = support.length ? support[0][2] : "";
    for (const edge of supportEdges(support, time)) {
      parent.add(annotate(line(edge, factor ? 0xc29013 : operatorColors[basis] || 0x75509c),
        description));
    }
  }

  for (const cycle of cycles) {
    const dataGroup = group("data", cycle.time);
    const syndromeGroup = group("syndrome", cycle.time);
    for (const qubit of cycle.qubits || []) {
      const isData = qubit.role === "data";
      const marker = new THREE.Mesh(isData ? sphereGeometry : ancillaGeometry,
        material(isData ? 0x263747 : 0xa5abb5));
      marker.position.set(qubit.x, qubit.y, cycle.time);
      const info = [isData ? "Data qubit" : "Syndrome ancilla", cycleText(cycle),
        `Position: (${qubit.x}, ${qubit.y})`,
        qubit.reset ? `Preparation: ${qubit.reset}` : "",
        qubit.measurement ? `Measurement: ${qubit.measurement}` : ""].filter(Boolean).join("\n");
      (isData ? dataGroup : syndromeGroup).add(annotate(marker, info));
      if (isData && qubit.reset) {
        const radius = 0.24;
        dataGroup.add(annotate(line([
          [qubit.x - radius, qubit.y - radius, cycle.time + 0.02],
          [qubit.x + radius, qubit.y - radius, cycle.time + 0.02],
          [qubit.x + radius, qubit.y + radius, cycle.time + 0.02],
          [qubit.x - radius, qubit.y + radius, cycle.time + 0.02],
        ], operatorColors[qubit.reset] || 0x263747, true), info));
      }
      if (isData && qubit.measurement) {
        const ring = new THREE.Mesh(ringGeometry, material(operatorColors[qubit.measurement] || 0x263747));
        ring.position.set(qubit.x, qubit.y, cycle.time + 0.02);
        ring.scale.setScalar(0.8);
        dataGroup.add(annotate(ring, info));
      }
    }
    const checks = group("checks", cycle.time);
    const changed = group("changed", cycle.time);
    for (const check of cycle.checks || []) {
      addCheck(checks, check, cycle);
      if (check.added) addCheck(changed, check, cycle, "Added");
    }
    for (const check of cycle.removed_checks || []) addCheck(changed, check, cycle, "Removed");
  }

  function box(center, size, kind, info, parent, openAxis = -1) {
    const bases = kind === "Y" ? ["Y", "Y", "Y"] : (kind || "").split("");
    const faceMaterials = [];
    for (let axis = 0; axis < 3; axis++) {
      const color = colors[bases[axis]] || 0xb1bac4;
      const face = material(color, axis === openAxis ? 0 : 1.7, true);
      faceMaterials.push(face, face);
    }
    const value = new THREE.Mesh(unitBoxGeometry, faceMaterials);
    value.position.set(...center);
    value.scale.set(...size);
    parent.add(annotate(value, info));
    return value;
  }

  function boxEdges(center, size, parent, color, dashed = false, info = "") {
    const edges = new THREE.LineSegments(
      geometry(new THREE.EdgesGeometry(unitBoxGeometry)), material(color, 1, false, true, dashed));
    edges.position.set(...center);
    edges.scale.set(...size);
    if (dashed) edges.computeLineDistances();
    parent.add(info ? annotate(edges, info) : edges);
  }

  function conditionalFrame(center, size, parent) {
    // Cylinders give the exterior white frame a real thickness on all WebGL implementations.
    const radius = Math.max(0.025, Math.min(...size) * 0.012);
    for (let axis = 0; axis < 3; axis++) {
      const other = [0, 1, 2].filter((value) => value !== axis);
      for (const a of [-1, 1]) for (const b of [-1, 1]) {
        const position = [...center];
        position[other[0]] += a * size[other[0]] / 2;
        position[other[1]] += b * size[other[1]] / 2;
        const edge = new THREE.Mesh(
          geometry(new THREE.CylinderGeometry(radius, radius, size[axis], 6)), material(0xffffff));
        edge.position.set(...position);
        if (axis === 0) edge.rotation.z = Math.PI / 2;
        if (axis === 2) edge.rotation.x = Math.PI / 2;
        parent.add(edge);
      }
    }
  }

  for (const coarse of data.coarse || []) {
    const {center, size} = coarse;
    const parent = group("coarse", center[2] - size[2] / 2, center[2] + size[2] / 2);
    const info = [coarse.label || `Block-level ${coarse.type}`, `Kind: ${coarse.kind}`,
      coarse.reason || "Physical representation unavailable."].join("\n");
    const pipeAxis = coarse.type === "pipe" ? coarse.kind.indexOf("O") : -1;
    if (coarse.faces?.length) {
      for (const face of coarse.faces) {
        const vertices = face.vertices || [];
        if (vertices.length < 3) continue;
        const positions = [];
        for (let i = 1; i < vertices.length - 1; i++) {
          positions.push(...vertices[0], ...vertices[i], ...vertices[i + 1]);
        }
        const faceGeometry = geometry(new THREE.BufferGeometry());
        faceGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        const isFrame = ["#ffffff", "#fff", "white"].includes(face.color?.toLowerCase());
        parent.add(annotate(new THREE.Mesh(faceGeometry,
          material(face.color || 0xb1bac4, isFrame ? 1 : 1.7, !isFrame)), info));
      }
      boxEdges(center, size, parent, 0x6d7988, true, info);
    } else if (coarse.type === "port") {
      boxEdges(center, size, parent, 0x6d7988, true, info);
    } else if (coarse.conditional) {
      const branches = coarse.conditional;
      const differing = [0, 1, 2].filter((i) => branches[0][i] !== branches[1][i]);
      const splitAxis = coarse.conditional_axis ?? ((differing.length === 1 ? differing[0] : 2) + 1) % 3;
      for (let branch = 0; branch < 2; branch++) {
        const branchSize = [...size];
        const branchCenter = [...center];
        branchSize[splitAxis] /= 2;
        branchCenter[splitAxis] += (branch ? 1 : -1) * size[splitAxis] / 4;
        box(branchCenter, branchSize, branches[branch],
          `${info}\nCondition ${branch}: ${branches[branch]}`, parent);
      }
      conditionalFrame(center, size, parent);
    } else if (pipeAxis >= 0 && coarse.kind.endsWith("H")) {
      for (let half = 0; half < 2; half++) {
        const halfSize = [...size];
        const halfCenter = [...center];
        halfSize[pipeAxis] /= 2;
        halfCenter[pipeAxis] += (half ? 1 : -1) * size[pipeAxis] / 4;
        const kind = half ? coarse.kind.replace(/[XZ]/g, (basis) => basis === "X" ? "Z" : "X")
          : coarse.kind;
        box(halfCenter, halfSize, kind, info, parent, pipeAxis);
      }
      const bandSize = [...size];
      bandSize[pipeAxis] = Math.max(0.035, size[pipeAxis] * 0.04);
      const band = new THREE.Mesh(unitBoxGeometry, material(0xffff65, 0.8));
      band.position.set(...center);
      band.scale.set(...bandSize);
      parent.add(annotate(band, `${info}\nHadamard transition`));
      boxEdges(center, size, parent, 0x6d7988, true, info);
    } else {
      box(center, size, coarse.kind, info, parent, pipeAxis);
      boxEdges(center, size, parent, 0x6d7988, true, info);
    }
  }

  for (const outline of data.outlines || []) {
    const {center, size} = outline;
    const parent = group("outlines", center[2] - size[2] / 2, center[2] + size[2] / 2);
    boxEdges(center, size, parent, 0xc1cbd4, false);
  }

  observables.forEach((observable, observableIndex) => {
    for (const snapshot of observable.cycles || []) {
      const cycle = cycleById.get(snapshot.cycle_id);
      if (!cycle) continue;
      const logical = group("logical", cycle.time, cycle.time, observableIndex);
      const info = [observable.name, cycleText(cycle), snapshot.description,
        `Support: ${signedSupportText(snapshot.support, snapshot.sign)}`,
        measurementText(snapshot.measurements, "Segment-local logical measurement parity"),
        snapshot.kind === "circuit_transport"
          ? `Circuit transport support: ${supportText(snapshot.transport_support)}` : "",
        snapshot.kind === "circuit_transport"
          ? measurementText(snapshot.transport_measurements,
            "Segment-local circuit transport measurement parity") : "",
      ].filter(Boolean).join("\n");
      addSupport(logical, snapshot.support || [], cycle.time + 0.14, info);
      for (const boundary of snapshot.boundaries || []) {
        const towardPast = boundary.side === "past";
        const time = cycle.time + (towardPast ? -0.38 : 0.38);
        const description = [observable.name, cycleText(cycle),
          `Physical tracking boundary toward ${towardPast ? "earlier" : "later"} rounds.`,
          boundary.reason, "No physical circuit flow is inferred across this boundary."]
          .filter(Boolean).join("\n");
        for (const [x, y] of snapshot.support || []) {
          const marker = new THREE.Mesh(boundaryGeometry, material(0xad6700));
          marker.position.set(x, y, time);
          logical.add(annotate(marker, description));
        }
        for (const edge of supportEdges(snapshot.support || [], time)) {
          logical.add(annotate(line(edge, 0xad6700, false, true), description));
        }
      }
      const deformation = group("deformation", cycle.time, cycle.time, observableIndex);
      for (const factor of snapshot.factors || []) {
        addSupport(deformation, factor.support || [], cycle.time + 0.25,
          [observable.name, factor.label || "Logical deformation factor", cycleText(cycle),
            snapshot.description, signedSupportText(factor.support, factor.sign),
            measurementText(factor.measurements)].filter(Boolean).join("\n"),
          true);
      }
    }
    for (const surface of observable.surfaces || []) {
      const vertices = surface.vertices || [];
      if (vertices.length < 2) continue;
      const parent = group("logical", Math.min(...vertices.map((v) => v[2])),
        Math.max(...vertices.map((v) => v[2])), observableIndex);
      const info = [observable.name, `${surface.basis} topological correlation surface`,
        surface.reason || "Physical operator tracking stops at this region."].join("\n");
      if (vertices.length > 2) {
        const positions = [];
        for (let i = 1; i < vertices.length - 1; i++) {
          positions.push(...vertices[0], ...vertices[i], ...vertices[i + 1]);
        }
        const surfaceGeometry = geometry(new THREE.BufferGeometry());
        surfaceGeometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
        parent.add(annotate(new THREE.Mesh(surfaceGeometry,
          material(operatorColors[surface.basis] || 0x75509c, 1.8, true)), info));
      }
      parent.add(annotate(line(vertices, operatorColors[surface.basis] || 0x75509c,
        vertices.length > 2, true), info));
    }
  });

  get("title").textContent = data.name ? `${data.name} · surface-code patches` : "Surface-code patches";
  get("summary").textContent = `d = ${data.d} · ${cycles.length} physical round${cycles.length === 1 ? "" : "s"}`
    + ` · ${(data.coarse || []).length} block-level region${(data.coarse || []).length === 1 ? "" : "s"}`;
  function updateNotices() {
    const index = Number(get("observable").value);
    const notices = [...(data.notices || []), ...(observables[index]?.notices || [])];
    get("notices").replaceChildren(...[...new Set(notices)].map((notice) => {
      const item = document.createElement("li");
      item.textContent = notice;
      return item;
    }));
  }
  for (const [index, observable] of observables.entries()) {
    const option = document.createElement("option");
    option.value = index;
    option.textContent = observable.name || `Observable ${index}`;
    get("observable").appendChild(option);
  }
  if (!observables.length) {
    const option = document.createElement("option");
    option.textContent = "No correlation surfaces";
    get("observable").appendChild(option);
    for (const id of ["observable", "show-logical", "show-deformation"]) get(id).disabled = true;
  }
  for (const id of ["first-round", "last-round"]) {
    get(id).max = Math.max(0, times.length - 1);
    get(id).disabled = !times.length;
  }
  get("single-round").disabled = !times.length;
  get("last-round").value = Math.max(0, times.length - 1);

  function render() { renderer.render(scene, camera); }

  function update() {
    const first = Number(get("first-round").value);
    const last = Number(get("last-round").value);
    const lower = times.length ? times[first] : -Infinity;
    const upper = times.length ? times[last] : Infinity;
    const isFullRange = first === 0 && last === times.length - 1 && !get("single-round").checked;
    const selected = Number(get("observable").value);
    world.scale.z = Number(get("spacing").value);
    for (const item of fixedThicknessObjects) item.object.scale.z = item.scaleZ / world.scale.z;
    const opacity = Number(get("opacity").value);
    for (const item of opacityMaterials) item.material.opacity = Math.min(0.95, opacity * item.coefficient);
    for (const item of records) {
      const inRange = !times.length || isFullRange || item.timeMax >= lower && item.timeMin <= upper;
      item.object.visible = get(`show-${item.category}`).checked && inRange
        && (item.observable === null || item.observable === selected);
    }
    const describeTime = (time) => [...new Set(cycles.filter((cycle) => cycle.time === time)
      .map(cycleText))].join("; ");
    get("round-label").textContent = !times.length ? "No physically compiled rounds."
      : first === last ? describeTime(lower) : `${describeTime(lower)} → ${describeTime(upper)}`;
    const observable = observables[selected];
    get("observable-detail").textContent = observable
      ? `${(observable.cycles || []).length} physical snapshots · ${(observable.surfaces || []).length} topological segments`
      : "";
    tooltip.style.display = "none";
    render();
  }

  function fitCamera() {
    world.updateMatrixWorld(true);
    const bounds = new THREE.Box3();
    for (const record of records) if (record.object.visible) bounds.expandByObject(record.object);
    if (bounds.isEmpty()) bounds.setFromCenterAndSize(new THREE.Vector3(), new THREE.Vector3(2, 2, 2));
    const center = bounds.getCenter(new THREE.Vector3());
    const size = bounds.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 1);
    const angle = Math.atan(Math.tan(camera.fov * Math.PI / 360) * Math.min(1, camera.aspect));
    const distance = radius / Math.sin(angle) * 1.15;
    camera.near = Math.max(0.005, distance / 10000);
    camera.far = Math.max(1000, distance * 100);
    camera.position.copy(center).addScaledVector(new THREE.Vector3(1, -1.4, 1.1).normalize(), distance);
    controls.target.copy(center);
    camera.updateProjectionMatrix();
    controls.update();
    render();
  }

  function resize() {
    const width = Math.max(container.clientWidth, 1);
    const height = Math.max(container.clientHeight, 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    render();
  }
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  controls.addEventListener("change", render);
  get("first-round").addEventListener("input", () => {
    if (get("single-round").checked || Number(get("first-round").value) > Number(get("last-round").value)) {
      get("last-round").value = get("first-round").value;
    }
    update();
  });
  get("last-round").addEventListener("input", () => {
    if (get("single-round").checked || Number(get("last-round").value) < Number(get("first-round").value)) {
      get("first-round").value = get("last-round").value;
    }
    update();
  });
  get("single-round").addEventListener("change", () => {
    if (get("single-round").checked) get("last-round").value = get("first-round").value;
    update();
  });
  get("all-rounds").addEventListener("click", () => {
    get("single-round").checked = false;
    get("first-round").value = 0;
    get("last-round").value = Math.max(0, times.length - 1);
    update();
  });
  get("fit").addEventListener("click", fitCamera);
  for (const input of document.querySelectorAll("input[id^=show-], #spacing, #opacity")) {
    input.addEventListener("input", update);
  }
  get("observable").addEventListener("change", () => { updateNotices(); update(); });

  const raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = 0.1;
  const pointer = new THREE.Vector2();
  const visible = (object) => {
    for (let current = object; current; current = current.parent) if (!current.visible) return false;
    return true;
  };
  renderer.domElement.addEventListener("pointermove", (event) => {
    if (event.buttons) { tooltip.style.display = "none"; return; }
    const rectangle = renderer.domElement.getBoundingClientRect();
    const x = event.clientX - rectangle.left;
    const y = event.clientY - rectangle.top;
    pointer.set(x / rectangle.width * 2 - 1, -(y / rectangle.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const intersection = raycaster.intersectObjects(pickable.filter(visible), false)[0];
    if (!intersection) { tooltip.style.display = "none"; return; }
    tooltip.textContent = intersection.object.userData.description;
    tooltip.style.display = "block";
    tooltip.style.left = `${Math.max(5, Math.min(x + 14, rectangle.width - tooltip.offsetWidth - 8))}px`;
    tooltip.style.top = `${Math.max(5, Math.min(y + 14, rectangle.height - tooltip.offsetHeight - 8))}px`;
  });
  renderer.domElement.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });
  renderer.domElement.addEventListener("webglcontextlost", (event) => {
    event.preventDefault();
    status.textContent = "The browser lost its WebGL context. Reload this viewer to restore the scene.";
    status.hidden = false;
  });
  window.addEventListener("pagehide", () => {
    resizeObserver.disconnect();
    controls.dispose();
    for (const value of geometries) value.dispose();
    for (const value of materials) value.dispose();
    renderer.dispose();
  }, {once: true});
  updateNotices();
  resize();
  update();
  fitCamera();
  status.hidden = true;
}

startPatchViewer().catch((error) => {
  const status = document.getElementById("status");
  status.hidden = false;
  status.textContent = `Unable to display the 3D scene: ${error.message || error}. `
    + "Check that WebGL is enabled and the pinned Three.js CDN is accessible.";
  console.error(error);
});
