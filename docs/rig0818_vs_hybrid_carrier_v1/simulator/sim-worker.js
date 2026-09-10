import {runSimulation} from './simulator-core.js';

self.onmessage = event => {
  try {
    const {targets, config} = event.data;
    const result = runSimulation(targets, config, progress => self.postMessage({type:'progress', progress}));
    self.postMessage({type:'result', result});
  } catch (error) {
    self.postMessage({type:'error', message:error?.stack || String(error)});
  }
};
