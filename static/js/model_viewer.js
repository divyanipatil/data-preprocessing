class ModelViewer {
    constructor(containerId, width = 400, height = 400) {
        this.container = document.getElementById(containerId);
        this.width = width;
        this.height = height;

        // Scene setup
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0xf0f0f0);

        // Camera setup
        this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
        this.camera.position.set(0, 0, 500);  // Move camera further back
        this.camera.lookAt(0, 0, 0);

        // Renderer setup
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.renderer.setSize(width, height);
        this.container.appendChild(this.renderer.domElement);

        // Lights
        this.scene.add(new THREE.AmbientLight(0x404040));
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        this.scene.add(directionalLight);

        // Grid helper (larger size)
        const gridHelper = new THREE.GridHelper(500, 50);
        this.scene.add(gridHelper);

        // Axes helper (longer axes)
        const axesHelper = new THREE.AxesHelper(250);
        this.scene.add(axesHelper);

        // Orbit controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = true;

        // Start animation
        this.animate();
    }

    loadModel(jsonData) {
        // Remove existing model
        if (this.mesh) {
            this.scene.remove(this.mesh);
        }

        try {
            // Parse the base64 encoded JSON data
            const modelData = JSON.parse(atob(jsonData));
            console.log('Model data:', {
                bounds: modelData.bounds,
                centroid: modelData.centroid,
                vertices: modelData.vertices.length,
                faces: modelData.faces.length
            });

            // Create geometry
            const geometry = new THREE.BufferGeometry();

            // Set vertices directly without modification
            const vertices = new Float32Array(modelData.vertices.flat());
            geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

            // Set faces
            const indices = new Uint32Array(modelData.faces.flat());
            geometry.setIndex(new THREE.BufferAttribute(indices, 1));

            // Compute normals
            geometry.computeVertexNormals();

            // Create material
            const material = new THREE.MeshPhongMaterial({
                color: 0x808080,
                side: THREE.DoubleSide,
                flatShading: true,
                wireframe: false
            });

            // Create mesh
            this.mesh = new THREE.Mesh(geometry, material);

            // Add to scene without repositioning
            this.scene.add(this.mesh);

            // Update camera to see the whole scene but don't center
            const boundingBox = new THREE.Box3().setFromObject(this.mesh);
            console.log('Mesh bounds:', boundingBox.min, boundingBox.max);

        } catch (error) {
            console.error('Error loading model:', error);
        }
    }

    animate = () => {
        requestAnimationFrame(this.animate);
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
}
