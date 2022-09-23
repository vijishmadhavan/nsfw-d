
const NSFWNET_MODEL_PATH ='model/tensorflowjs_model.pb';
const NSFWNET_WEIGHTS_PATH ='model/weights_manifest.json';

const IMAGE_SIZE = 256;
const IMAGE_CROP_SIZE = 224;
const TOPK_PREDICTIONS = 5;

const NSFW_CLASSES = {
  0: 'drawing',
  1: 'hentai',
  2: 'neural',
  3: 'porn',
  4: 'sexy',
};



let nsfwnet;
const nsfwnetDemo = async () => {
  //status('Loading model...');

  // nsfwnet = await tf.loadModel(MOBILENET_MODEL_PATH);
  nsfwnet = await tf.loadGraphModel(NSFWNET_MODEL_PATH, NSFWNET_WEIGHTS_PATH);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
  nsfwnet.predict(tf.zeros([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])).dispose();

  console.log('Model Warm complete');

  // Make a prediction through the locally hosted test_draw.jpg.
  const image_Element = document.getElementById('test_draw');
  if (image_Element.complete && image_Element.naturalHeight !== 0) {

    predict(image_Element);
    image_Element.style.display = '';
  } else {

    image_Element.onload = () => {
      predict(image_Element);
      image_Element.style.display = '';
    }
  }

  document.getElementById('file-container').style.display = '';
};


/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement) {
  //status('Predicting...');

  const startTime = performance.now();
  const logits = tf.tidy(() => {

    // tf.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();
    const crop_image = tf.slice(img, [16, 16, 0], [224, 224, -1]);
    const img_reshape = tf.reverse(crop_image, [-1]);

    let imagenet_mean = tf.expandDims([103.94, 116.78, 123.68], 0);
    imagenet_mean = tf.expandDims(imagenet_mean, 0);

    // Normalize the image from [0, 255] to [-1, 1].
    const normalized = img_reshape.sub(imagenet_mean);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3]);

    // Make a prediction through mobilenet.
    return nsfwnet.predict(batched);
  });

  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime = performance.now() - startTime;
  //status(`Done in ${Math.floor(totalTime)}ms`);

  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
async function getTopKClasses(logits, topK) {
  const values = await logits.data();

  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }
  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });
  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < 1; i++) {
    topClassesAndProbs.push({
      className: NSFW_CLASSES[topkIndices[0]]
      //probability: topkValues[0]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//


function showResults(imgElement, classes) {

  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';
  

  const imgContainer = document.createElement('div');
  //imgContainer.appendChild(imgElement);
  //predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row';

    const classElement = document.createElement('div');
    classElement.className = 'cell';
    //classElement.innerText = classes[i].className;    
    
    if(classes[i].className == "hentai")
    {const textnode = document.createTextNode("NSFW");
    classElement.appendChild(textnode);
    row.appendChild(classElement);}
    else if(classes[i].className == "porn")
    {const textnode = document.createTextNode("NSFW");
    classElement.appendChild(textnode)
    row.appendChild(classElement);}
    else 
    {const textnode = document.createTextNode("");
    //classElement.appendChild(textnode)
    imgContainer.appendChild(imgElement);}
    predictionContainer.appendChild(imgContainer);
    
    row.appendChild(imgContainer);

    
    //const probsElement = document.createElement('div');
    //probsElement.className = 'cell';
    //probsElement.innerText = classes[i].probability.toFixed(3);
    //row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);

  // remove predictionsElement's last child node (if it exists) and append the new one 
  if (predictionsElement.childNodes.length > 0) {
    predictionsElement.removeChild(predictionsElement.childNodes[0]);
  }

  predictionsElement.appendChild(
      predictionContainer);
}


const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});


const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;

const predictionsElement = document.getElementById('predictions');




nsfwnetDemo();
