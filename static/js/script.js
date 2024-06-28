document.addEventListener('DOMContentLoaded', () => {
    const startButton = document.getElementById('start');
    const registerButton = document.getElementById('register');
    const trainButton = document.getElementById('train');
    const analyzeButton = document.getElementById('analyze');
    const attendanceButton = document.getElementById('attendance');
    const passwordButton = document.getElementById('password');
    const video = document.getElementById('video');
    const flash = document.getElementById('flash');
    const imageCount = document.getElementById('imageCount');
    const cameraSelect = document.getElementById('cameraList');
    const fileSelect = document.getElementById('fileSelect');
    flashInterval = null;

    // Load models on page load to disable unessesary  buttons
    fetch('/load_models')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                startButton.disabled = true;
                analyzeButton.disabled = true;
                alert(data.error);
            }
        });
    
    // Load attendance files on page load
    fetch('/list_attendance_files')
    .then(response => response.json())
    .then(files => {
        files.forEach(file => {
            const option = document.createElement('option');
            option.value = file;
            option.textContent = file;
            fileSelect.appendChild(option);
        });
    });

    fileSelect.addEventListener('change', () => {
        const selectedFile = fileSelect.value;
        if (selectedFile) {
            fetch(`/read_attendance/${selectedFile}`)
                .then(response => response.json())
                .then(data => {
                    const tableBody = document.getElementById('logTable').getElementsByTagName('tbody')[0];
                    tableBody.innerHTML = '';  // Clear any existing rows

                    if (data.error) {
                        alert(data.error);
                    } else {
                        data.forEach((row, index) => {
                            const newRow = tableBody.insertRow();
                            newRow.insertCell(0).textContent = index + 1;
                            newRow.insertCell(1).textContent = row.Name;
                            newRow.insertCell(2).textContent = row.Time;
                            newRow.insertCell(3).textContent = row.Probability;
                        });
                    }
                })
                .catch(error => alert(error));
        }
    });
    
    fetch('/list_cameras')
        .then(response => response.json())
        .then(cameras => {
            cameras.forEach((camera, index) => {
                const option = document.createElement('option');
                option.value = camera;
                option.textContent = index;
                cameraSelect.appendChild(option);
            });
        });

    startButton.addEventListener('click', () => {
        startButton.disabled = true;
        if (startButton.style.backgroundColor === 'maroon') {
            // Stop facial recognition
            startButton.style.backgroundColor = '#6a3acb';
            startButton.style.backgroundImage = "url('static/image/recognize-start.png')";
            registerButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.disabled = false;
            video.onerror = video.src='static/image/null.png';
            fetch('/stop_feed');
        } else {
            fetch('/load_models')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert(data.error);
                        return;
                    }
                });
            // Start facial recognition
            startButton.style.backgroundColor = 'maroon';
            startButton.style.backgroundImage = "url('static/image/recognize-stop.png')";
            video.src = "/face_recognition";
            registerButton.disabled  = true;
            trainButton.disabled = true;
            analyzeButton.disabled = true;
        }
        timeout(startButton);
    });

    registerButton.addEventListener('click', () => {
        registerButton.disabled = true;
        if (registerButton.style.backgroundColor === 'maroon') {
            clearInterval(flashInterval);
            registerButton.style.backgroundColor = '#6a3acb';
            registerButton.style.backgroundImage = "url('static/image/register.png')";
            imageCount.textContent = '';
            video.onerror = video.src='static/image/null.png';
            startButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.disabled = false;
            fetch('/stop_feed');
            video.removeEventListener('load', handleVideoLoad);
        } else {
            registerButton.style.backgroundColor = 'maroon';
            registerButton.style.backgroundImage = "url('static/image/stop.png')";
            video.src = "/face_capturing";
            startButton.disabled = true;
            trainButton.disabled = true;
            analyzeButton.disabled = true;
            // Wait for video to load before capturing images
            video.addEventListener('load', handleVideoLoad);
        }
        timeout(registerButton);
    });

    trainButton.addEventListener('click', () => {
        const password = prompt("Enter password:");
        if (password) {
            verifyPassword(password).then(isVerified => {
                if (isVerified) {
                    trainButton.disabled = true;
                    startButton.disabled = true;
                    registerButton.disabled = true;
                    analyzeButton.disabled = true;
                    trainButton.style.backgroundImage = "url('static/image/training.png')";
                    // Training takes time depending on the number of images
                    fetch('/training')
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                alert(data.error);
                            } else {
                                startButton.disabled = false;
                                analyzeButton.disabled = false;
                                trainButton.disabled = false;
                                registerButton.disabled = false;
                                trainButton.style.backgroundImage = "url('static/image/train.png')";
                                alert(data.message);
                            }
                        });
                } else {
                    alert("Incorrect password!");
                }
            });
        }
    });

    analyzeButton.addEventListener('click', () => {
        if (analyzeButton.style.backgroundColor === 'maroon') {
            startButton.disabled = false;
            registerButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.style.backgroundColor = '#6a3acb'
            video.onerror = video.src='static/image/null.png';
        } else {
            const password = prompt("Enter password:");
            if (password) {
                verifyPassword(password).then(isVerified => {
                    if (isVerified) {
                        // Analyzing model takes time depending on the number of data points
                        fetch('/load_models')
                            .then(response => response.json())
                            .then(data => {
                                if (data.error) {
                                    alert(data.error);
                                    return;
                                }
                            });
                        startButton.disabled = true;
                        registerButton.disabled = true;
                        trainButton.disabled = true;
                        analyzeButton.disabled = true;
                        analyzeButton.style.backgroundColor = 'maroon'
                        analyzeButton.style.backgroundImage = "url('static/image/training.png')";
                        video.src = '/analyze_model';
                        video.addEventListener('load', handleVisualizerLoad);
                    } else {
                        alert("Incorrect password!");
                    }
                });
            }
        }
    });

    attendanceButton.addEventListener('click', () => {
        const tableBody = document.getElementById('logTable').getElementsByTagName('tbody')[0];
        const table = document.getElementById('attendance-table')

        if (attendanceButton.style.backgroundColor === 'maroon') {
            // Hide attendance table
            table.style.visibility = 'collapse';
            attendanceButton.style.backgroundImage = "url('static/image/attendance-view.png')";
            attendanceButton.style.backgroundColor = '#6a3acb';
            tableBody.innerHTML = '';  // Clear any existing rows
        }
        else {
            const password = prompt("Enter password:");
            if (password) {
                verifyPassword(password).then(isVerified => {
                    if (isVerified) {
                        fetch('/read_attendance')
                        .then(response => response.json())
                        .then(data => {
                            if (data.error) {
                                alert(data.error);
                                // Display other attendance if today's attendance is not available
                                const selectedFile = fileSelect.value;
                                if (selectedFile) {
                                    table.style.visibility = 'visible';
                                    attendanceButton.style.backgroundImage = "url('static/image/attendance-close.png')";
                                    attendanceButton.style.backgroundColor = 'maroon';
                                    selectAttendance(selectedFile);
                                }
                            }
                            else {
                                // Display today's attendance
                                table.style.visibility = 'visible';
                                attendanceButton.style.backgroundImage = "url('static/image/attendance-close.png')";
                                attendanceButton.style.backgroundColor = 'maroon';
                                fileSelect.value = fileSelect.options[fileSelect.options.length - 1].value;
                                tableBody.innerHTML = '';  // Clear any existing rows

                                data.forEach((row, index) => {
                                    const newRow = tableBody.insertRow();
                                    newRow.insertCell(0).textContent = index + 1;
                                    newRow.insertCell(1).textContent = row.Name;
                                    newRow.insertCell(2).textContent = row.Time;
                                    newRow.insertCell(3).textContent = row.Probability
                                });
                            }
                        })
                        .catch(error => alert(error));
                    } else {
                        alert("Incorrect password!");
                    }
                });
            }
        }
    });

    passwordButton.addEventListener('click', () => {
        const currentPassword = prompt("Enter current password:");
        if (currentPassword) {
            verifyPassword(currentPassword).then(isVerified => {
                if (isVerified) {
                    const newPassword = prompt("Enter new password:");
                    if (newPassword) {
                        updatePassword(currentPassword, newPassword).then(isUpdated => {
                            if (isUpdated) {
                                alert("Password changed successfully!");
                            } else {
                                alert("Failed to change password!");
                            }
                        });
                    }
                } else {
                    alert("Incorrect password!");
                }
            });
        }
    });

    fileSelect.addEventListener('change', () => {
        const selectedFile = fileSelect.value;
        if (selectedFile)
            selectAttendance(selectedFile);
    });

    cameraSelect.addEventListener('change', () => {
        const selectedCamera = cameraSelect.value;
        if (selectedCamera)
            fetch(`/change_camera/${selectedCamera}`)
    });

    function timeout(button) {
        // Enable button after 3 seconds to prevent spamming
        setTimeout(() => {
            button.disabled = false;
        }, 3000);
    }

    async function handleVideoLoad() {
        video.removeEventListener('load', handleVideoLoad);
        const userName = await prompt("Enter Name:");
        if (!userName) {
            alert("User name is required.");
            registerButton.style.backgroundColor = '#6a3acb';
            registerButton.style.backgroundImage = "url('static/image/register.png')";
            startButton.disabled = false;
            trainButton.disabled = false;
            analyzeButton.disabled = false;
            video.onerror = video.src='static/image/null.png';
            fetch('/stop_feed');
            return;
        } else {
            fetch('/capture_images', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ user_name: userName })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    alert(data.message);
                }
            });
        
            // Flash effect
            let count = 0;
            flashInterval = setInterval(() => {
                flash.style.opacity = 1;
                setTimeout(() => {
                    flash.style.opacity = 0;
                }, 100);
                
                count++;
                imageCount.textContent = `${count}/50`;
                if (count > 50) {
                    clearInterval(flashInterval);
                    registerButton.style.backgroundImage = "url('static/image/register.png')";
                    imageCount.textContent = '';
                    video.onerror = video.src='static/image/null.png';
                    fetch('/stop_feed');
                    return;
                }
            }, 800);
        }
    }

    function handleVisualizerLoad() {
        video.removeEventListener('load', handleVisualizerLoad);
        analyzeButton.disabled = false;
        analyzeButton.style.backgroundImage = "url('static/image/analyze.png')";
    }

    function selectAttendance(selectedFile) {
        // Display attendance based on selected file
        fetch(`/read_attendance/${selectedFile}`)
            .then(response => response.json())
            .then(data => {
                const tableBody = document.getElementById('logTable').getElementsByTagName('tbody')[0];
                tableBody.innerHTML = '';  // Clear any existing rows
                if (data.error) {
                    alert(data.error);
                } else {
                    data.forEach((row, index) => {
                        const newRow = tableBody.insertRow();
                        newRow.insertCell(0).textContent = index + 1;
                        newRow.insertCell(1).textContent = row.Name;
                        newRow.insertCell(2).textContent = row.Time;
                        newRow.insertCell(3).textContent = row.Probability;
                    });
                }
            })
            .catch(error => alert(error));
    }

    async function verifyPassword(password) {
        const response = await fetch('/verify_password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ password })
        });
        const result = await response.json();
        return result.status === 'success';
    }

    async function updatePassword(currentPassword, newPassword) {
        const response = await fetch('/update_password', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ current_password: currentPassword, new_password: newPassword })
        });
        const result = await response.json();
        return result.status === 'success';
    }
});