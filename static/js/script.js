document.addEventListener('DOMContentLoaded', () => {
    const video = document.getElementById('video');
    const startButton = document.getElementById('start');
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const faceNameElement = document.createElement('div');
    const registerButton = document.getElementById('register');

    startButton.addEventListener('click', () => {
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.play();
                startRecognition();
            })
            .catch(err => {
                console.error("Error accessing webcam: ", err);
            });
    });

    registerButton.addEventListener('click', () => {
        const name = prompt("Enter your name:");
        if (name) {
            $.ajax({
                url: '/register',
                type: 'POST',
                data: { name: name },
                success: (response) => {
                    alert(response.message);
                },
                error: (error) => {
                    console.error("Error registering user: ", error);
                }
            });
        }
    });


    function startRecognition() {
        setInterval(() => {
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/jpeg');
            const blob = dataURItoBlob(dataURL);
            const formData = new FormData();
            formData.append('image', blob);

            fetch('/recognize', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                drawResults(data);
            })
            .catch(err => {
                console.error("Error recognizing face: ", err);
            });
        }, 1000); // Adjust the interval for smoother fps
    }

    function drawResults(results) {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        results.forEach(result => {
            const { name, box } = result;
            const [x, y, w, h] = box;
            context.strokeStyle = 'blue';
            context.lineWidth = 2;
            context.strokeRect(x, y, w, h);
            context.font = '18px Arial';
            context.fillStyle = 'blue';
            context.fillText(name, x, y - 10);
        });
    }

    function dataURItoBlob(dataURI) {
        const byteString = atob(dataURI.split(',')[1]);
        const mimeString = dataURI.split(',')[0].split(':')[1].split(';')[0];
        const ab = new ArrayBuffer(byteString.length);
        const ia = new Uint8Array(ab);
        for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
        }
        return new Blob([ab], { type: mimeString });
    }
});
