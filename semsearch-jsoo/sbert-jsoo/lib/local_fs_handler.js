// A TensorFlow.js IOHandler that doesn't require loading tfjs-node.

class LocalFSHandler {
  constructor(path, loadOptions = {}) {
    this.path = path;
    this.weightPathPrefix = loadOptions.weightPathPrefix;

    // Lazily load fs.
    this.fs = require('fs').promises;
  }

  async load() {
    const modelConfigPath = this.path + '/model.json';
    let modelConfig;
    try {
      const modelConfigContent = await this.fs.readFile(modelConfigPath, 'utf8');
      modelConfig = JSON.parse(modelConfigContent);
    } catch (e) {
      throw new Error(`Failed to read or parse model JSON from ${modelConfigPath}: ${e}`);
    }

    const { modelTopology, weightsManifest } = modelConfig;

    if (modelTopology == null && weightsManifest == null) {
      throw new Error(`The JSON from path ${this.path} contains neither model topology or manifest for weights.`);
    }

    let weightSpecs;
    let weightData;
    if (weightsManifest != null) {
      const results = await this.loadWeights(weightsManifest);
      [weightSpecs, weightData] = results;
    }

    return { modelTopology, weightSpecs, weightData };
  }

  async loadWeights(weightsManifest) {
    const weightSpecs = weightsManifest[0].weights;
    const shardPaths = weightsManifest[0].paths.map(p => this.path + '/' + p);
    const buffers = await Promise.all(shardPaths.map(p => this.fs.readFile(p)));

    const weightData = Buffer.concat(buffers);
    return [weightSpecs, weightData.buffer];
  }
}

globalThis.LocalFSHandler = LocalFSHandler;
