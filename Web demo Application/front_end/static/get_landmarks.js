const videoElement = document.getElementsByClassName('input_video')[0];
const canvas_element = document.getElementsByClassName('output_canvas')[0];
const canvas = canvas_element.getContext('2d');
console.log(canvas);
var left_marks=new Array(1000);
var right_marks=new Array(1000);
var empty=[-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];
var right_index=0;
var left_index=0;
window.start = function start(){
    console.log("start");
    hands.onResults(onResults);
}


function output_landmark(rightmarks,leftmarks){
    console.log(rightmarks);
    console.log(leftmarks);
}

const hands = new Hands({locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }});
hands.setOptions({
    maxNumHands: 2,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});


const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 1280,
    height: 720
});
function onResults(results) {
    setTimeout(function(){output_landmark(right_marks,left_marks);alert("录制完毕！");},3000);
    canvas.save();
    console.log(canvas);
    console.log(canvas_element)
    canvas.clearRect(0, 0, canvas_element.width, canvas_element.height);
    canvas.drawImage(results.image, 0, 0, canvas_element.width, canvas_element.height);
    if (results.multiHandLandmarks) {
        for (const landmarks of results.multiHandLandmarks) {
            console.log(landmarks);
            console.log(results);
            if(results.multiHandedness[0].label=="Left"&&results.multiHandedness[1]==null){
                left_marks.push(results.multiHandLandmarks[0]);
                right_marks.push(empty);
            }
            else if(results.multiHandedness[0].label=="Right"&&results.multiHandedness[1]==null){
                right_marks.push(results.multiHandLandmarks[0]);
                left_marks.push(empty);
            }
            else if(results.multiHandedness[0].label!=null&&results.multiHandedness[1]!=null){
                if(results.multiHandedness[0].label=="Left"){
                    left_index=results.multiHandedness[0].index;
                    right_index=results.multiHandedness[1].index;
                }
                else{
                    left_index=results.multiHandedness[1].index;
                    right_index=results.multiHandedness[0].index;
                }
                right_marks.push(results.multiHandLandmarks[right_index]);
                left_marks.push(results.multiHandLandmarks[left_index]);
            }
            output_landmark(landmarks);
            drawConnectors(canvas, landmarks, HAND_CONNECTIONS,
                {color: '#0000FF', lineWidth: 5});
            drawLandmarks(canvas, landmarks, {color: '#FF0000', lineWidth: 2});
        }
    }
    canvas.restore();
}
camera.start();