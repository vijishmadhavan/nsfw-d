import * as nsfwjs from 'nsfwjs'

let imageLoaded = false;
$("#image-selector").change(function () {
	imageLoaded = false;
	let reader = new FileReader();
	reader.onload = function () {
		let dataURL = reader.result;
		$("#selected-image").attr("src", dataURL);
		$("#prediction-list").empty();
		imageLoaded = true;
	}
	
	let file = $("#image-selector").prop('files')[0];
	reader.readAsDataURL(file);
});

let model;
let modelLoaded = false;
$( document ).ready(async function () {
	modelLoaded = false;
	$('.progress-bar').show();
    console.log( "Loading model..." );
    //model = await tf.loadLayersModel('model/model.json');
    const model = await nsfwjs.load('/model/')
    console.log( "Model loaded." );
	$('.progress-bar').hide();
	modelLoaded = true;
});

$("#predict-button").click(async function () {
	if (!modelLoaded) { alert("The model must be loaded first"); return; }
	if (!imageLoaded) { alert("Please select an image first"); return; }
	
	//let image = $('#selected-image').get(0);
	const img = document.getElementById('#selected-image')

	// Pre-process the image
	console.log( "Loading image..." );
	//let tensor = tf.browser.fromPixels(image)
	//let predictions = await model.predict(tensor).data();
	const predictions = await model.classify(img)
	console.log('Predictions: ', predictions);

});
