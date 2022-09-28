
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

  nsfwnet = await tf.loadGraphModel(NSFWNET_MODEL_PATH, NSFWNET_WEIGHTS_PATH);

  nsfwnet.predict(tf.zeros([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3])).dispose();

  console.log('Model Warm complete');

  const image_Element = document.getElementById('test_draw');
  if (image_Element.complete && image_Element.naturalHeight !== 0) {

    predict(image_Element);
    image_Element.style.display = '';
  } 
  document.getElementById('file-container').style.display = '';
};


async function predict(imgElement) {
  

  const logits = tf.tidy(() => {

    const img = tf.browser.fromPixels(imgElement).toFloat();
    const crop_image = tf.slice(img, [16, 16, 0], [224, 224, -1]);
    const img_reshape = tf.reverse(crop_image, [-1]);

    let imagenet_mean = tf.expandDims([103.94, 116.78, 123.68], 0);
    imagenet_mean = tf.expandDims(imagenet_mean, 0);

    const normalized = img_reshape.sub(imagenet_mean);

    const batched = normalized.reshape([1, IMAGE_CROP_SIZE, IMAGE_CROP_SIZE, 3]);

    return nsfwnet.predict(batched);
  });

  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);

  imageIsSfw(classes);
}


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
    })
  }
  return topClassesAndProbs;
}



function imageIsSfw(classes) {
  if(classes[0].className == "hentai")
  {console.log("NSFW");}
  else if(classes[0].className =="porn")
  {console.log("NSFW");}
  else
  {console.log("SFW");}
}


 
const predictionsElement = document.getElementById('predictions');



nsfwnetDemo();